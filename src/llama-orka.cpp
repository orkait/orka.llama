#include "llama-orka.h"
#include "ggml-backend.h"

#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <vector>

// keyed by the stage-0 index tensor pointer (reused as the layer weight pointer)
static std::unordered_map<const ggml_tensor *, llama_orka_weight> g_orka;
static ggml_context * g_orka_ctx = nullptr;
static ggml_backend_buffer_t g_orka_buf = nullptr;
static ggml_context * g_orka_idx_ctx = nullptr;
static ggml_backend_buffer_t g_orka_idx_buf = nullptr;

void llama_orka_register(const ggml_tensor * key, const llama_orka_weight & w) {
    g_orka[key] = w;
}
const llama_orka_weight * llama_orka_lookup(const ggml_tensor * key) {
    auto it = g_orka.find(key);
    return it == g_orka.end() ? nullptr : &it->second;
}
void llama_orka_clear() {
    g_orka.clear();
    if (g_orka_buf) { ggml_backend_buffer_free(g_orka_buf); g_orka_buf = nullptr; }
    if (g_orka_ctx) { ggml_free(g_orka_ctx); g_orka_ctx = nullptr; }
    if (g_orka_idx_buf) { ggml_backend_buffer_free(g_orka_idx_buf); g_orka_idx_buf = nullptr; }
    if (g_orka_idx_ctx) { ggml_free(g_orka_idx_ctx); g_orka_idx_ctx = nullptr; }
}

// Unpack bit-plane indices (lo uint8 + hi packed) into resident I32 idx tensors. MSB-first
// bitstream per the python _pack_indices: idx = lo | (hi_val << 8), hi_val read from `hi`.
void llama_orka_finalize() {
    // idempotent: -fit param fitting calls load_tensors (and this) multiple times.
    if (g_orka_idx_buf) { ggml_backend_buffer_free(g_orka_idx_buf); g_orka_idx_buf = nullptr; }
    if (g_orka_idx_ctx) { ggml_free(g_orka_idx_ctx); g_orka_idx_ctx = nullptr; }
    // skip the -fit dry-run load where tensors exist but have no backend buffer yet
    if (!g_orka.empty() && g_orka.begin()->second.lo[0] && g_orka.begin()->second.lo[0]->buffer) {
        size_t n_idx = 0;
        for (auto & kv : g_orka) n_idx += kv.second.n_stages;
        ggml_init_params ip = { ggml_tensor_overhead() * (n_idx + 8), nullptr, true };
        g_orka_idx_ctx = ggml_init(ip);
        for (auto & kv : g_orka) {
            llama_orka_weight & w = kv.second;
            for (int s = 0; s < w.n_stages; s++) {
                int64_t count = ggml_nelements(w.lo[s]);
                w.idx[s] = ggml_new_tensor_1d(g_orka_idx_ctx, GGML_TYPE_I32, count);
            }
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(g_orka.begin()->second.lo[0]->buffer);
        g_orka_idx_buf = ggml_backend_alloc_ctx_tensors_from_buft(g_orka_idx_ctx, buft);

        std::vector<uint8_t> lo, hi; std::vector<int32_t> idx;
        for (auto & kv : g_orka) {
            llama_orka_weight & w = kv.second;
            for (int s = 0; s < w.n_stages; s++) {
                int64_t count = ggml_nelements(w.lo[s]);
                int hb = w.idx_bits[s] - 8;            // hi bits per index (0 if <=8)
                lo.resize(count); ggml_backend_tensor_get(w.lo[s], lo.data(), 0, count);
                hi.resize(ggml_nbytes(w.hi[s])); ggml_backend_tensor_get(w.hi[s], hi.data(), 0, hi.size());
                idx.resize(count);
                for (int64_t j = 0; j < count; j++) {
                    uint32_t hival = 0;
                    for (int b = 0; b < hb; b++) {
                        size_t pos = (size_t) j * hb + b;
                        int bit = (hi[pos >> 3] >> (7 - (pos & 7))) & 1;
                        hival = (hival << 1) | (uint32_t) bit;   // MSB first
                    }
                    idx[j] = (int32_t) (lo[j] | (hival << 8));
                }
                ggml_backend_tensor_set(w.idx[s], idx.data(), 0, count * sizeof(int32_t));
            }
            // warp GEMV args (N=1 decode): device pointers straight to the bit-planes.
            const char * bn = ggml_backend_buft_name(ggml_backend_buffer_get_type(w.lo[0]->buffer));
            if (bn && strstr(bn, "CUDA")) {
                for (int s = 0; s < 3; s++) {
                    int s0 = s < w.n_stages ? s : 0;     // unused stages alias stage 0
                    w.warp.lo[s] = w.lo[s0]->data;
                    w.warp.hi[s] = w.hi[s0]->data;
                    w.warp.cb[s] = w.cb[s0]->data;
                }
                w.warp.scale = w.scales->data;
                w.warp.M = w.M; w.warp.G = w.group_size; w.warp.GPR = w.K / w.group_size;
                w.warp.BPR = w.K / w.block_size; w.warp.GPB = w.block_size / w.group_size;
                w.warp.HI_BITS = w.idx_bits[0] > 8 ? w.idx_bits[0] - 8 : 0;
                w.warp.N_STAGES = w.n_stages;
                w.warp_ready = true;
            }
        }
    }
    llama_orka_materialize();
}

// Reconstruct W^T [K,M] from the RVQ side tensors using native ggml ops, then mul_mat.
// All ops (get_rows / reshape / mul / mul_mat) run on whatever backend the graph is on
// (CUDA included) - no custom CPU-only op. Verified bit-exact to dequant_linear.
//
//   gather:  WT = sum_s get_rows(cb_s[G,cbsz], idx_s[M*GPR])   -> [G, M*GPR]
//   reshape: WT -> [K, M]   (k = g*G+e, since K = GPR*G; this is W transposed)
//   scale:   WT[B,BPR,M] *= scales[1,BPR,M]   (per-block, broadcast over block_size)
//   matmul:  y = mul_mat(WT[K,M], cur[K,N]) -> [M,N]
// ---- load-time decompress: materialize W^T [K,M] once ----
void llama_orka_materialize() {
    if (g_orka_buf) { ggml_backend_buffer_free(g_orka_buf); g_orka_buf = nullptr; }
    if (g_orka_ctx) { ggml_free(g_orka_ctx); g_orka_ctx = nullptr; }
    if (!getenv("ORKA_DECOMPRESS") || g_orka.empty()) return;

    const size_t n = g_orka.size();
    ggml_init_params ip = { ggml_tensor_overhead() * (n + 8), nullptr, /*no_alloc*/ true };
    g_orka_ctx = ggml_init(ip);
    for (auto & kv : g_orka) {
        llama_orka_weight & w = kv.second;
        // Q8_0 so decode uses llama.cpp's fast quantized mat-vec kernel (F16 GEMV is slower)
        w.Wmat = ggml_new_tensor_2d(g_orka_ctx, GGML_TYPE_Q8_0, w.K, w.M); // W^T [K,M]
    }
    // one buffer on the side tensors' backend (assume uniform placement)
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(g_orka.begin()->second.idx[0]->buffer);
    g_orka_buf = ggml_backend_alloc_ctx_tensors_from_buft(g_orka_ctx, buft);

    std::vector<int32_t> idxh[3]; std::vector<ggml_fp16_t> cbh[3], sch;
    std::vector<float> Wf; std::vector<char> Wq;
    for (auto & kv : g_orka) {
        llama_orka_weight & w = kv.second;
        const int M = w.M, K = w.K, G = w.group_size, B = w.block_size, S = w.n_stages;
        const int GPR = K / G, BPR = K / B;
        for (int s = 0; s < S; s++) {
            idxh[s].resize(ggml_nelements(w.idx[s]));
            ggml_backend_tensor_get(w.idx[s], idxh[s].data(), 0, ggml_nbytes(w.idx[s]));
            cbh[s].resize(ggml_nelements(w.cb[s]));
            ggml_backend_tensor_get(w.cb[s], cbh[s].data(), 0, ggml_nbytes(w.cb[s]));
        }
        sch.resize(ggml_nelements(w.scales));
        ggml_backend_tensor_get(w.scales, sch.data(), 0, ggml_nbytes(w.scales));

        Wf.assign((size_t) K * M, 0.0f);
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                int g = k / G, e = k % G, blk = k / B;
                float acc = 0.0f;
                for (int s = 0; s < S; s++)
                    acc += ggml_fp16_to_fp32(cbh[s][(size_t) idxh[s][(size_t) m * GPR + g] * G + e]);
                Wf[(size_t) m * K + k] = acc * ggml_fp16_to_fp32(sch[(size_t) m * BPR + blk]); // W^T[k,m]
            }
        }
        Wq.resize(ggml_nbytes(w.Wmat));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, Wf.data(), Wq.data(), 0, M, K, nullptr); // M rows of K
        ggml_backend_tensor_set(w.Wmat, Wq.data(), 0, Wq.size());
    }
}

ggml_tensor * llama_orka_build_mm(ggml_context * ctx, const llama_orka_weight & w, ggml_tensor * cur) {
    if (w.Wmat) {
        return ggml_mul_mat(ctx, w.Wmat, cur);   // decompressed-resident: plain GPU mul_mat
    }
#ifdef GGML_USE_CUDA
    // N=1 decode on GPU: warp GEMV straight off the bit-planes (compressed-resident, no W).
    if (cur->ne[1] == 1 && w.warp_ready) {
        ggml_tensor * xf = ggml_cast(ctx, cur, GGML_TYPE_F16);
        return ggml_orka_warp(ctx, xf, &w.warp);
    }
#endif
    const int M = w.M, K = w.K, G = w.group_size, B = w.block_size, S = w.n_stages;
    const int GPR = K / G, BPR = K / B;

    ggml_tensor * WT = nullptr;
    for (int s = 0; s < S; s++) {
        int64_t cbsz = ggml_nelements(w.cb[s]) / G;
        // F16 cb (smaller -> better bandwidth) cast to F32 so get_rows output is F32
        // (avoids F16 stride asserts on CUDA binops); the cast is cheap vs the bandwidth win.
        ggml_tensor * cb2 = ggml_cast(ctx, ggml_reshape_2d(ctx, (ggml_tensor *) w.cb[s], G, cbsz), GGML_TYPE_F32);
        // idx is filled by llama_orka_finalize after load; during the -fit graph-reserve it is
        // still null, so use a correctly-shaped dummy (reserve only needs shapes, not data).
        ggml_tensor * idx = w.idx[s];
        if (idx == nullptr) idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ggml_nelements(w.lo[s]));
        ggml_tensor * gr  = ggml_get_rows(ctx, cb2, idx);            // [G, M*GPR] f32
        WT = WT ? ggml_add(ctx, WT, gr) : gr;
    }
    WT = ggml_reshape_3d(ctx, ggml_cont(ctx, WT), B, BPR, M);                  // [block, BPR, M]
    ggml_tensor * sc = ggml_cast(ctx, (ggml_tensor *) w.scales, GGML_TYPE_F32);
    sc = ggml_reshape_3d(ctx, sc, 1, BPR, M);                                 // broadcast over block
    WT = ggml_mul(ctx, WT, sc);
    WT = ggml_reshape_2d(ctx, WT, K, M);                                      // [K, M] = W^T
    return ggml_mul_mat(ctx, WT, cur);                                        // [M, N]
}
