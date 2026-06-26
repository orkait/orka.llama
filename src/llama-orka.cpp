#include "llama-orka.h"

#include <unordered_map>
#include <cstring>
#include <vector>

// keyed by the stage-0 index tensor pointer (reused as the layer weight pointer)
static std::unordered_map<const ggml_tensor *, llama_orka_weight> g_orka;

void llama_orka_register(const ggml_tensor * key, const llama_orka_weight & w) {
    g_orka[key] = w;
}
const llama_orka_weight * llama_orka_lookup(const ggml_tensor * key) {
    auto it = g_orka.find(key);
    return it == g_orka.end() ? nullptr : &it->second;
}
void llama_orka_clear() { g_orka.clear(); }

// Reconstruct W^T [K,M] from the RVQ side tensors using native ggml ops, then mul_mat.
// All ops (get_rows / reshape / mul / mul_mat) run on whatever backend the graph is on
// (CUDA included) - no custom CPU-only op. Verified bit-exact to dequant_linear.
//
//   gather:  WT = sum_s get_rows(cb_s[G,cbsz], idx_s[M*GPR])   -> [G, M*GPR]
//   reshape: WT -> [K, M]   (k = g*G+e, since K = GPR*G; this is W transposed)
//   scale:   WT[B,BPR,M] *= scales[1,BPR,M]   (per-block, broadcast over block_size)
//   matmul:  y = mul_mat(WT[K,M], cur[K,N]) -> [M,N]
ggml_tensor * llama_orka_build_mm(ggml_context * ctx, const llama_orka_weight & w, ggml_tensor * cur) {
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
