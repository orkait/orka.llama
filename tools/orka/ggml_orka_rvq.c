// GGML op for orka RVQ matmul, via ggml_custom_4d (no new op enum / ggml.c dispatch edit).
//
//   ggml_orka_rvq_mul_mat(ctx, x, idx[], cb[], scales, M,K,G,B,n_stages,group_major) -> y[M,N]
//
// Builds a custom-op node whose srcs are {x, idx_0..idx_{S-1}, cb_0..cb_{S-1}, scales}; the
// callback computes the fused dequant-matmul (the verified kernel) over the output rows,
// parallelised by ith/nth. This runs the orka type inside a ggml compute graph - the step
// between the standalone kernel refs and full model-graph wiring. CPU here; CUDA next.
//
// The test main reads the Python reference dump (cref/) and checks the ggml-op output
// equals expected_y, i.e. the op works in a real ggml graph.
//
// Build: gcc -O2 -I ../../ggml/include ggml_orka_rvq.c \
//        ../../build_cuda/ggml/src/libggml.so ... -lm   (see build cmd in PR)

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

struct orka_params { int M, K, G, B, n_stages, group_major; };

// dst->src layout: [0]=x(f32,[K]), [1..S]=idx(i16), [S+1..2S]=cb(f32), [2S+1]=scales(f32)
static void orka_rvq_op(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const struct orka_params * p = (const struct orka_params *) ud;
    const int M = p->M, K = p->K, G = p->G, S = p->n_stages, gm = p->group_major;
    const int GPR = K / G, BPR = K / p->B, GPB = p->B / G;
    const float * x = (const float *) dst->src[0]->data;
    const int16_t * idx[3]; const float * cb[3];
    for (int s = 0; s < S; s++) { idx[s] = (const int16_t *) dst->src[1 + s]->data;
                                  cb[s]  = (const float *)   dst->src[1 + S + s]->data; }
    const float * scale = (const float *) dst->src[1 + 2 * S]->data;
    float * y = (float *) dst->data;

    for (int m = ith; m < M; m += nth) {
        float acc = 0.0f;
        for (int g = 0; g < GPR; g++) {
            int ip = gm ? (g * M + m) : (m * GPR + g);
            int blk = g / GPB;
            float s = scale[gm ? (blk * M + m) : (m * BPR + blk)];
            for (int e = 0; e < G; e++) {
                float w = 0.0f;
                for (int st = 0; st < S; st++)
                    w += cb[st][(int)(uint16_t)idx[st][ip] * G + e];
                acc += s * w * x[g * G + e];
            }
        }
        y[m] = acc;
    }
}

struct ggml_tensor * ggml_orka_rvq_mul_mat(
        struct ggml_context * ctx, struct ggml_tensor * x,
        struct ggml_tensor ** idx, struct ggml_tensor ** cb, struct ggml_tensor * scales,
        struct orka_params * p) {
    struct ggml_tensor * args[10];
    int n = 0; args[n++] = x;
    for (int s = 0; s < p->n_stages; s++) args[n++] = idx[s];
    for (int s = 0; s < p->n_stages; s++) args[n++] = cb[s];
    args[n++] = scales;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, p->M, 1, 1, 1, args, n, orka_rvq_op, 1, p);
}

// ---- standalone graph test against the Python reference dump ----
static void * rd(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void * b = malloc(sz); if (fread(b, 1, sz, f) != (size_t) sz) exit(1); fclose(f);
    if (n) *n = sz; return b;
}
static int mint(const char * j, const char * k) {
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\":", k); const char * q = strstr(j, pat);
    return q ? atoi(q + strlen(pat)) : (fprintf(stderr, "meta %s\n", k), exit(1), 0);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ref_dir>\n", argv[0]); return 1; }
    const char * dir = argv[1]; char path[1024];
    snprintf(path, sizeof(path), "%s/meta.json", dir); char * meta = rd(path, NULL);
    struct orka_params p = { mint(meta, "M"), mint(meta, "K"), mint(meta, "G"),
                             mint(meta, "B"), mint(meta, "n_stages"), mint(meta, "group_major") };
    int GPR = p.K / p.G, BPR = p.K / p.B;

    size_t mem = (size_t) 256 * 1024 * 1024;
    struct ggml_init_params ip = { mem, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.K);
    memcpy(x->data, rd((snprintf(path, sizeof(path), "%s/x.f32", dir), path), NULL), p.K * 4);
    struct ggml_tensor * idx[3]; struct ggml_tensor * cb[3];
    for (int s = 0; s < p.n_stages; s++) {
        idx[s] = ggml_new_tensor_1d(ctx, GGML_TYPE_I16, (int64_t) p.M * GPR);
        size_t isz; void * id = rd((snprintf(path, sizeof(path), "%s/idx%d.i16", dir, s), path), &isz);
        memcpy(idx[s]->data, id, isz);
        size_t csz; void * cd = rd((snprintf(path, sizeof(path), "%s/cb%d.f32", dir, s), path), &csz);
        cb[s] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, csz / 4); memcpy(cb[s]->data, cd, csz);
    }
    struct ggml_tensor * scales = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) p.M * BPR);
    memcpy(scales->data, rd((snprintf(path, sizeof(path), "%s/scales.f32", dir), path), NULL),
           (size_t) p.M * BPR * 4);

    struct ggml_tensor * y = ggml_orka_rvq_mul_mat(ctx, x, idx, cb, scales, &p);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    float * exp = rd((snprintf(path, sizeof(path), "%s/expected_y.f32", dir), path), NULL);
    float * yo = (float *) y->data; double me = 0.0;
    for (int m = 0; m < p.M; m++) { double e = fabs(yo[m] - exp[m]); if (e > me) me = e; }
    printf("ggml orka op [%d,%d] stages=%d  max_err=%.3e  %s\n",
           p.M, p.K, p.n_stages, me, me < 2e-3 ? "OK" : "FAIL");
    ggml_free(ctx);
    return me < 2e-3 ? 0 : 2;
}
