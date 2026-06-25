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

// Fused dequant-matmul over output rows; parallelised by ith/nth. cur=[K,N] -> dst=[M,N].
// Reference math: y[m,n] = sum_g scale * sum_{e<G} (sum_s cb_s[idx_s[m,g]][e]) * cur[g*G+e, n]
static void orka_rvq_op(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const llama_orka_weight * p = (const llama_orka_weight *) ud;
    const int M = p->M, K = p->K, G = p->group_size, S = p->n_stages, gm = p->group_major;
    const int GPR = K / G, BPR = K / p->block_size, GPB = p->block_size / G;
    const struct ggml_tensor * xt = dst->src[0];
    const int N = (int) xt->ne[1];
    const float * x = (const float *) xt->data;          // [K, N]
    const int16_t * idx[3]; const ggml_fp16_t * cb[3];
    for (int s = 0; s < S; s++) { idx[s] = (const int16_t *)     dst->src[1 + s]->data;
                                  cb[s]  = (const ggml_fp16_t *) dst->src[1 + S + s]->data; }
    const ggml_fp16_t * scale = (const ggml_fp16_t *) dst->src[1 + 2 * S]->data;
    float * y = (float *) dst->data;                     // [M, N]

    for (int m = ith; m < M; m += nth) {
        for (int n = 0; n < N; n++) y[(size_t) n * M + m] = 0.0f;
        for (int g = 0; g < GPR; g++) {
            int ip  = gm ? (g * M + m) : (m * GPR + g);
            int blk = g / GPB;
            float s = ggml_fp16_to_fp32(scale[gm ? (blk * M + m) : (m * BPR + blk)]);
            for (int e = 0; e < G; e++) {
                float w = 0.0f;
                for (int st = 0; st < S; st++)
                    w += ggml_fp16_to_fp32(cb[st][(int)(uint16_t)idx[st][ip] * G + e]);
                float sw = s * w;
                int kcol = g * G + e;
                for (int n = 0; n < N; n++)
                    y[(size_t) n * M + m] += sw * x[(size_t) n * K + kcol];
            }
        }
    }
}

ggml_tensor * llama_orka_build_mm(ggml_context * ctx, const llama_orka_weight & w, ggml_tensor * cur) {
    // userdata must outlive graph compute; the registry entry does (lives on the model).
    const llama_orka_weight * ud = llama_orka_lookup(w.idx[0]);
    struct ggml_tensor * args[8]; int n = 0;
    args[n++] = cur;
    for (int s = 0; s < w.n_stages; s++) args[n++] = (ggml_tensor *) w.idx[s];
    for (int s = 0; s < w.n_stages; s++) args[n++] = (ggml_tensor *) w.cb[s];
    args[n++] = (ggml_tensor *) w.scales;
    int N = (int) cur->ne[1];
    return ggml_custom_4d(ctx, GGML_TYPE_F32, w.M, N, 1, 1, args, n,
                          orka_rvq_op, GGML_N_TASKS_MAX, (void *) ud);
}
