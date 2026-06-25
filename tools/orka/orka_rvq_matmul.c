// Fused dequant-matmul reference for GGML_TYPE_ORKA_RVQ (CPU).
//
// Computes y = W @ x WITHOUT materializing W:
//     y[m] = sum_g scale[m, g/(B/G)] * sum_{e<G} (sum_s codebook_s[idx_s[m,g]][e]) * x[g*G+e]
// This is exactly the compute a GGML mul_mat / mat_vec for the orka type performs (the
// per-tensor codebook is read as a side input - the reason orka needs a custom op rather
// than a self-contained quant type). Standalone: validates the kernel math against the
// Python reference (expected_y = dequant_linear(W) @ x) before the ggml-op / graph wiring.
//
// Build:  gcc -O2 orka_rvq_matmul.c -o orka_rvq_matmul -lm
// Run:    ./orka_rvq_matmul <ref_dir>   (meta.json, idx{s}.i16, cb{s}.f32, scales.f32,
//                                         x.f32, expected_y.f32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static void *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { exit(1); }
    fclose(f); return buf;
}
static int meta_int(const char *j, const char *k) {
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\":", k);
    const char *p = strstr(j, pat); if (!p) { fprintf(stderr, "meta %s?\n", k); exit(1); }
    return atoi(p + strlen(pat));
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ref_dir>\n", argv[0]); return 1; }
    const char *dir = argv[1]; char p[1024];
    snprintf(p, sizeof(p), "%s/meta.json", dir); char *meta = read_file(p);
    int M = meta_int(meta, "M"), K = meta_int(meta, "K"), G = meta_int(meta, "G"), B = meta_int(meta, "B");
    int n_stages = meta_int(meta, "n_stages"), group_major = meta_int(meta, "group_major");
    int GPR = K / G, BPR = K / B, GPB = B / G;

    int16_t *idx[3]; float *cb[3];
    for (int s = 0; s < n_stages; s++) {
        snprintf(p, sizeof(p), "%s/idx%d.i16", dir, s); idx[s] = read_file(p);
        snprintf(p, sizeof(p), "%s/cb%d.f32", dir, s);  cb[s]  = read_file(p);
    }
    snprintf(p, sizeof(p), "%s/scales.f32", dir);     float *scale = read_file(p);
    snprintf(p, sizeof(p), "%s/x.f32", dir);          float *x = read_file(p);
    snprintf(p, sizeof(p), "%s/expected_y.f32", dir); float *expected = read_file(p);

    float *y = calloc((size_t)M, sizeof(float));
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int g = 0; g < GPR; g++) {
            int ip = group_major ? (g * M + m) : (m * GPR + g);
            int blk = g / GPB;
            int sp = group_major ? (blk * M + m) : (m * BPR + blk);
            float s = scale[sp];
            for (int e = 0; e < G; e++) {
                float w = 0.0f;
                for (int st = 0; st < n_stages; st++)
                    w += cb[st][(int)(uint16_t)idx[st][ip] * G + e];
                acc += s * w * x[g * G + e];
            }
        }
        y[m] = acc;
    }

    double max_err = 0.0;
    for (int m = 0; m < M; m++) {
        double e = fabs((double)y[m] - (double)expected[m]);
        if (e > max_err) max_err = e;
    }
    printf("orka fused dequant-matmul [%d,%d] G=%d stages=%d gm=%d  max_err=%.3e  %s\n",
           M, K, G, n_stages, group_major, max_err, max_err < 2e-3 ? "OK" : "FAIL");
    return max_err < 2e-3 ? 0 : 2;
}
