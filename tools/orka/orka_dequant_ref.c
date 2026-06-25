// CPU dequant reference for GGML_TYPE_ORKA_RVQ.
//
// Reconstructs a linear weight W[M,K] from orka's RVQ tensors - per-stage indices,
// per-stage codebooks, and per-block scales - by the rule
//     W[m, g*G+e] = (sum_s codebook_s[idx_s[m,g]*G + e]) * block_scale[m, g/(B/G)]
// This is the C twin of orka.export_gguf.dequant_linear and the math the GGML to_float /
// mat_vec kernels must implement. Standalone (reads the raw dumps written by the Python
// reference) so it can be gated independent of the full GGML model-graph integration.
//
// Build:  gcc -O2 orka_dequant_ref.c -o orka_dequant_ref -lm
// Run:    ./orka_dequant_ref <ref_dir>     (expects meta.json, idx{s}.i16, cb{s}.f32,
//                                            scales.f32, expected.f32)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static void *read_file(const char *path, long *nbytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); if (nbytes) *nbytes = n; return buf;
}

// minimal int field reader from the flat meta.json ("key":value)
static int meta_int(const char *json, const char *key) {
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *p = strstr(json, pat);
    if (!p) { fprintf(stderr, "meta missing %s\n", key); exit(1); }
    return atoi(p + strlen(pat));
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ref_dir>\n", argv[0]); return 1; }
    const char *dir = argv[1];
    char path[1024];

    snprintf(path, sizeof(path), "%s/meta.json", dir);
    char *meta = read_file(path, NULL);
    int M = meta_int(meta, "M"), K = meta_int(meta, "K");
    int G = meta_int(meta, "G"), B = meta_int(meta, "B");
    int n_stages = meta_int(meta, "n_stages");
    int group_major = meta_int(meta, "group_major");
    int GPR = K / G, BPR = K / B, GPB = B / G;

    int16_t *idx[3]; float *cb[3];
    for (int s = 0; s < n_stages; s++) {
        snprintf(path, sizeof(path), "%s/idx%d.i16", dir, s); idx[s] = read_file(path, NULL);
        snprintf(path, sizeof(path), "%s/cb%d.f32", dir, s);  cb[s]  = read_file(path, NULL);
    }
    snprintf(path, sizeof(path), "%s/scales.f32", dir);   float *scales = read_file(path, NULL);
    snprintf(path, sizeof(path), "%s/expected.f32", dir); float *expected = read_file(path, NULL);

    float *W = malloc((size_t)M * K * sizeof(float));
    for (int m = 0; m < M; m++) {
        for (int g = 0; g < GPR; g++) {
            // row-major index p = m*GPR+g; group-major p = g*M+m
            int p = group_major ? (g * M + m) : (m * GPR + g);
            int blk = g / GPB;
            int sp = group_major ? (blk * M + m) : (m * BPR + blk);
            float s = scales[sp];
            for (int e = 0; e < G; e++) {
                float acc = 0.0f;
                for (int st = 0; st < n_stages; st++)
                    acc += cb[st][(int)(uint16_t)idx[st][p] * G + e];
                W[(size_t)m * K + g * G + e] = acc * s;
            }
        }
    }

    double max_err = 0.0;
    for (size_t i = 0; i < (size_t)M * K; i++) {
        double e = fabs((double)W[i] - (double)expected[i]);
        if (e > max_err) max_err = e;
    }
    printf("orka CPU dequant [%d,%d] G=%d stages=%d gm=%d  max_err=%.3e  %s\n",
           M, K, G, n_stages, group_major, max_err, max_err < 1e-3 ? "OK" : "FAIL");
    return max_err < 1e-3 ? 0 : 2;
}
