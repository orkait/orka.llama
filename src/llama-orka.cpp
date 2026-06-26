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
    const int M = w.M, K = w.K, G = w.group_size, B = w.block_size, S = w.n_stages;
    const int GPR = K / G, BPR = K / B;

    ggml_tensor * WT = nullptr;
    for (int s = 0; s < S; s++) {
        int64_t cbsz = ggml_nelements(w.cb[s]) / G;
        // F16 cb (smaller -> better bandwidth) cast to F32 so get_rows output is F32
        // (avoids F16 stride asserts on CUDA binops); the cast is cheap vs the bandwidth win.
        ggml_tensor * cb2 = ggml_cast(ctx, ggml_reshape_2d(ctx, (ggml_tensor *) w.cb[s], G, cbsz), GGML_TYPE_F32);
        ggml_tensor * gr  = ggml_get_rows(ctx, cb2, (ggml_tensor *) w.idx[s]); // [G, M*GPR] f32
        WT = WT ? ggml_add(ctx, WT, gr) : gr;
    }
    WT = ggml_reshape_3d(ctx, ggml_cont(ctx, WT), B, BPR, M);                  // [block, BPR, M]
    ggml_tensor * sc = ggml_cast(ctx, (ggml_tensor *) w.scales, GGML_TYPE_F32);
    sc = ggml_reshape_3d(ctx, sc, 1, BPR, M);                                 // broadcast over block
    WT = ggml_mul(ctx, WT, sc);
    WT = ggml_reshape_2d(ctx, WT, K, M);                                      // [K, M] = W^T
    return ggml_mul_mat(ctx, WT, cur);                                        // [M, N]
}
