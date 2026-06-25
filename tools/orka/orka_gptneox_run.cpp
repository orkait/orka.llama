// End-to-end GPTNeoX (pythia) runner over an orka RVQ gguf, in ggml.
//
// Loads pythia_orka.gguf (quantized linears as idx/cb/scales side tensors + passthrough
// norms/biases + arch hparams in KV), builds the GPTNeoX forward graph in ggml using the
// orka custom op for all 48 linears, and runs a forward. Validates staged activations
// (embed / layer0 / final-norm / logits) against the transformers orka reference dump,
// then runs greedy generation. This is the "compressed model runs end to end in a real
// C++ engine" milestone - orka compression + the fused dequant-matmul inside ggml.
//
// Build (CPU):
//   g++ -O2 -I ../../ggml/include orka_gptneox_run.cpp \
//       ../../build_cuda/bin/libggml-base.so ../../build_cuda/bin/libggml-cpu.so \
//       ../../build_cuda/bin/libggml.so -lm -o orka_gptneox_run
//   ./orka_gptneox_run ../../../pythia_orka.gguf <ref_dir>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

// ---------------- orka RVQ custom op (N-column generalization) ----------------
struct orka_params { int M, K, G, B, n_stages, group_major; };

static void orka_rvq_op(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const orka_params * p = (const orka_params *) ud;
    const int M = p->M, K = p->K, G = p->G, S = p->n_stages, gm = p->group_major;
    const int GPR = K / G, BPR = K / p->B, GPB = p->B / G;
    const struct ggml_tensor * xt = dst->src[0];
    const int N = (int) xt->ne[1];
    const float * x = (const float *) xt->data;          // [K, N]
    const int16_t * idx[3]; const ggml_fp16_t * cb[3];
    for (int s = 0; s < S; s++) { idx[s] = (const int16_t *) dst->src[1 + s]->data;
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

// ---------------- gguf weight store ----------------
struct Model {
    struct ggml_context * ctx;   // holds all gguf tensors
    struct gguf_context * gguf;
    int n_layer, n_head, n_embd, n_ff, n_vocab, n_ctx, rot_dims, group_major;
    float rotary_base, ln_eps;
    int parallel_residual;
};

static struct ggml_tensor * T(Model & mo, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(mo.ctx, name.c_str());
    if (!t) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    return t;
}
static uint32_t kv_u32(struct gguf_context * g, const std::string & k) {
    int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { fprintf(stderr, "missing kv %s\n", k.c_str()); exit(1); }
    return gguf_get_val_u32(g, id);
}
static float kv_f32(struct gguf_context * g, const std::string & k) {
    int64_t id = gguf_find_key(g, k.c_str());
    if (id < 0) { fprintf(stderr, "missing kv %s\n", k.c_str()); exit(1); }
    return gguf_get_val_f32(g, id);
}

// One orka linear -> ggml node. params persisted in `store` (callback reads at compute).
static struct ggml_tensor * orka_linear(struct ggml_context * cc, Model & mo,
        const std::string & wname, struct ggml_tensor * x,
        std::vector<orka_params*> & store) {
    std::string meta = "orka.linear." + wname + ".";
    orka_params * p = new orka_params{
        (int) kv_u32(mo.gguf, meta + "out_features"),
        (int) kv_u32(mo.gguf, meta + "in_features"),
        (int) kv_u32(mo.gguf, meta + "group_size"),
        (int) kv_u32(mo.gguf, meta + "block_size"),
        (int) kv_u32(mo.gguf, meta + "n_stages"),
        (int) kv_u32(mo.gguf, meta + "group_major") };
    store.push_back(p);
    struct ggml_tensor * args[10]; int n = 0; args[n++] = x;
    for (int s = 0; s < p->n_stages; s++) args[n++] = T(mo, wname + ".idx" + std::to_string(s));
    for (int s = 0; s < p->n_stages; s++) args[n++] = T(mo, wname + ".cb"  + std::to_string(s));
    args[n++] = T(mo, wname + ".scales");
    int N = (int) x->ne[1];
    return ggml_custom_4d(cc, GGML_TYPE_F32, p->M, N, 1, 1, args, n, orka_rvq_op, GGML_N_TASKS_MAX, p);
}

static struct ggml_tensor * layernorm(struct ggml_context * cc, struct ggml_tensor * x,
        struct ggml_tensor * w, struct ggml_tensor * b, float eps) {
    x = ggml_norm(cc, x, eps);
    x = ggml_mul(cc, x, w);
    x = ggml_add(cc, x, b);
    return x;
}

static void * readf(const char * path, size_t * n) {
    FILE * f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void * b = malloc(sz); if (fread(b, 1, sz, f) != (size_t) sz) exit(1); fclose(f);
    if (n) *n = sz; return b;
}
static double maxerr(const float * a, const float * b, size_t n) {
    double m = 0; for (size_t i = 0; i < n; i++) { double e = fabs((double)a[i]-(double)b[i]); if (e>m) m=e; }
    return m;
}

struct Captures { struct ggml_tensor *embed,*layer0,*finalln,*ln1,*qkv,*attn,*mlp; };

// Build + run the GPTNeoX forward for `toks`, return logits of the last position
// (malloc'd, caller frees) and argmax. Fills `cap` with staged nodes if non-null.
static int forward(Model & mo, const std::vector<int32_t> & toks, float ** last_logits,
                   Captures * cap) {
    int seq = (int) toks.size();
    int head_dim = mo.n_embd / mo.n_head;
    size_t cmem = (size_t) 2 * 1024 * 1024 * 1024;
    struct ggml_init_params cp = { cmem, NULL, false };
    struct ggml_context * cc = ggml_init(cp);
    std::vector<orka_params*> store;

    struct ggml_tensor * tok = ggml_new_tensor_1d(cc, GGML_TYPE_I32, seq);
    memcpy(tok->data, toks.data(), seq * 4);
    struct ggml_tensor * pos = ggml_new_tensor_1d(cc, GGML_TYPE_I32, seq);
    for (int i = 0; i < seq; i++) ((int32_t*)pos->data)[i] = i;

    struct ggml_tensor * x = ggml_get_rows(cc, T(mo,"gpt_neox.embed_in.weight"), tok); // [E, seq]
    struct ggml_tensor * embed_node = x;

    struct ggml_tensor * layer0_node = NULL;
    struct ggml_tensor *dbg_ln1=NULL,*dbg_qkv=NULL,*dbg_attn=NULL,*dbg_mlp=NULL;
    for (int l = 0; l < mo.n_layer; l++) {
        std::string P = "gpt_neox.layers." + std::to_string(l) + ".";
        struct ggml_tensor * ln1 = layernorm(cc, x, T(mo,P+"input_layernorm.weight"),
                                              T(mo,P+"input_layernorm.bias"), mo.ln_eps);
        struct ggml_tensor * qkv = orka_linear(cc, mo, P+"attention.query_key_value.weight", ln1, store);
        qkv = ggml_add(cc, qkv, T(mo,P+"attention.query_key_value.bias")); // [3E, seq]
        if(l==0){dbg_ln1=ln1;dbg_qkv=qkv;}

        // split heads: qkv laid as [n_head, 3*head_dim] per column
        size_t es = ggml_element_size(qkv);
        struct ggml_tensor * q = ggml_view_3d(cc, qkv, head_dim, mo.n_head, seq,
                3*head_dim*es, qkv->nb[1], 0*head_dim*es);
        struct ggml_tensor * k = ggml_view_3d(cc, qkv, head_dim, mo.n_head, seq,
                3*head_dim*es, qkv->nb[1], 1*head_dim*es);
        struct ggml_tensor * v = ggml_view_3d(cc, qkv, head_dim, mo.n_head, seq,
                3*head_dim*es, qkv->nb[1], 2*head_dim*es);
        q = ggml_rope_ext(cc, ggml_cont(cc,q), pos, NULL, mo.rot_dims, GGML_ROPE_TYPE_NEOX,
                mo.n_ctx, mo.rotary_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(cc, ggml_cont(cc,k), pos, NULL, mo.rot_dims, GGML_ROPE_TYPE_NEOX,
                mo.n_ctx, mo.rotary_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * Q = ggml_cont(cc, ggml_permute(cc, q, 0,2,1,3)); // [hd, seq, H]
        struct ggml_tensor * K = ggml_cont(cc, ggml_permute(cc, k, 0,2,1,3));
        struct ggml_tensor * KQ = ggml_mul_mat(cc, K, Q);                     // [seq, seq, H]
        KQ = ggml_scale(cc, KQ, 1.0f/sqrtf((float)head_dim));
        KQ = ggml_diag_mask_inf(cc, KQ, 0);
        KQ = ggml_soft_max(cc, KQ);
        struct ggml_tensor * V = ggml_cont(cc, ggml_permute(cc, v, 1,2,0,3)); // [seq, hd, H]
        struct ggml_tensor * KQV = ggml_mul_mat(cc, V, KQ);                   // [hd, seq, H]
        KQV = ggml_cont(cc, ggml_permute(cc, KQV, 0,2,1,3));                  // [hd, H, seq]
        struct ggml_tensor * attn = ggml_reshape_2d(cc, KQV, mo.n_embd, seq);
        attn = orka_linear(cc, mo, P+"attention.dense.weight", attn, store);
        attn = ggml_add(cc, attn, T(mo,P+"attention.dense.bias"));
        if(l==0)dbg_attn=attn;

        // parallel residual MLP (uses original x via post_attention_layernorm)
        struct ggml_tensor * ln2 = layernorm(cc, x, T(mo,P+"post_attention_layernorm.weight"),
                                              T(mo,P+"post_attention_layernorm.bias"), mo.ln_eps);
        struct ggml_tensor * h = orka_linear(cc, mo, P+"mlp.dense_h_to_4h.weight", ln2, store);
        h = ggml_add(cc, h, T(mo,P+"mlp.dense_h_to_4h.bias"));
        h = ggml_gelu_erf(cc, h);
        h = orka_linear(cc, mo, P+"mlp.dense_4h_to_h.weight", h, store);
        h = ggml_add(cc, h, T(mo,P+"mlp.dense_4h_to_h.bias"));
        if(l==0)dbg_mlp=h;

        x = ggml_add(cc, ggml_add(cc, x, attn), h); // parallel residual
        if (l == 0) layer0_node = x;
    }
    struct ggml_tensor * finalln = layernorm(cc, x, T(mo,"gpt_neox.final_layer_norm.weight"),
                                             T(mo,"gpt_neox.final_layer_norm.bias"), mo.ln_eps);
    struct ggml_tensor * logits = ggml_mul_mat(cc, T(mo,"embed_out.weight"), finalln); // [V, seq]

    struct ggml_cgraph * gf = ggml_new_graph(cc);
    ggml_build_forward_expand(gf, logits);
    if (cap) for(auto*t:{embed_node,layer0_node,finalln,dbg_ln1,dbg_qkv,dbg_attn,dbg_mlp})
        ggml_build_forward_expand(gf,t);
    ggml_graph_compute_with_ctx(cc, gf, 8);

    const float * lg = (const float *) logits->data;
    const float * last = lg + (size_t)(seq-1) * mo.n_vocab;
    int am = 0; for (int i = 1; i < mo.n_vocab; i++) if (last[i] > last[am]) am = i;
    if (last_logits) { *last_logits = (float*) malloc(mo.n_vocab*4); memcpy(*last_logits, last, mo.n_vocab*4); }
    if (cap) {
        auto dup=[&](ggml_tensor*t,int dim){ float*b=(float*)malloc((size_t)seq*dim*4); memcpy(b,t->data,(size_t)seq*dim*4); return b; };
        cap->embed=(ggml_tensor*)dup(embed_node,mo.n_embd); cap->layer0=(ggml_tensor*)dup(layer0_node,mo.n_embd);
        cap->finalln=(ggml_tensor*)dup(finalln,mo.n_embd); cap->ln1=(ggml_tensor*)dup(dbg_ln1,mo.n_embd);
        cap->qkv=(ggml_tensor*)dup(dbg_qkv,3*mo.n_embd); cap->attn=(ggml_tensor*)dup(dbg_attn,mo.n_embd);
        cap->mlp=(ggml_tensor*)dup(dbg_mlp,mo.n_embd);
    }
    ggml_free(cc);
    for (auto*p:store) delete p;
    return am;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf> [ref_dir]\n", argv[0]); return 1; }
    Model mo{};
    struct gguf_init_params gp = { false, &mo.ctx };
    mo.gguf = gguf_init_from_file(argv[1], gp);
    if (!mo.gguf) { fprintf(stderr, "gguf load failed\n"); return 1; }
    std::string A = "orka.linear.arch.";
    mo.n_layer = kv_u32(mo.gguf, A+"n_layer"); mo.n_head = kv_u32(mo.gguf, A+"n_head");
    mo.n_embd  = kv_u32(mo.gguf, A+"n_embd");  mo.n_ff   = kv_u32(mo.gguf, A+"n_ff");
    mo.n_vocab = kv_u32(mo.gguf, A+"n_vocab"); mo.n_ctx  = kv_u32(mo.gguf, A+"n_ctx");
    mo.rotary_base = kv_f32(mo.gguf, A+"rotary_base"); mo.ln_eps = kv_f32(mo.gguf, A+"ln_eps");
    mo.parallel_residual = kv_u32(mo.gguf, A+"parallel_residual");
    int head_dim = mo.n_embd / mo.n_head;
    mo.rot_dims = (int) lround(kv_f32(mo.gguf, A+"rotary_pct") * head_dim);
    printf("arch: L=%d H=%d E=%d FF=%d V=%d hd=%d rot=%d base=%.0f eps=%.1e par=%d\n",
           mo.n_layer, mo.n_head, mo.n_embd, mo.n_ff, mo.n_vocab, head_dim,
           mo.rot_dims, mo.rotary_base, mo.ln_eps, mo.parallel_residual);

    std::vector<int32_t> toks = {12,13,247,2456,273};
    if (argc >= 3) {
        char p[1024]; snprintf(p, sizeof(p), "%s/tokens.i32", argv[2]);
        size_t tn; int32_t * td = (int32_t*) readf(p, &tn); toks.assign(td, td + tn/4);
    }
    int seq = (int) toks.size();

    // ---- staged validation against the transformers reference ----
    if (argc >= 3) {
        Captures cap; float* lglast;
        int am = forward(mo, toks, &lglast, &cap);
        const char * d = argv[2]; char p[1024]; size_t n;
        auto chk=[&](const char*nm, void*t, int dim){ snprintf(p,sizeof(p),"%s/%s.f32",d,nm); float*r=(float*)readf(p,&n); printf("%-8s max_err = %.3e\n", nm, maxerr((float*)t,r,(size_t)seq*dim)); free(r); };
        chk("l0_ln1",cap.ln1,mo.n_embd); chk("l0_qkv",cap.qkv,3*mo.n_embd);
        chk("l0_attn",cap.attn,mo.n_embd); chk("l0_mlp",cap.mlp,mo.n_embd);
        chk("embed",cap.embed,mo.n_embd); chk("layer0",cap.layer0,mo.n_embd);
        chk("finalln",cap.finalln,mo.n_embd);
        snprintf(p,sizeof(p),"%s/logits.f32",d); float* rl=(float*)readf(p,&n);
        const float* rlast=rl+(size_t)(seq-1)*mo.n_vocab;
        printf("logits  max_err = %.3e\n", maxerr(lglast,rlast,mo.n_vocab));
        int ram=0; for(int i=1;i<mo.n_vocab;i++) if(rlast[i]>rlast[ram]) ram=i;
        printf("argmax = %d  ref = %d  %s\n", am, ram, ram==am?"MATCH":"MISMATCH");
        free(lglast); free(rl);
    }

    // ---- greedy generation ----
    int n_gen = 20;
    printf("\ngenerate (%d tokens, greedy):\n  prompt:", n_gen);
    for (int t : toks) printf(" %d", t);
    printf("\n  cont:  ");
    for (int i = 0; i < n_gen; i++) {
        int nxt = forward(mo, toks, NULL, NULL);
        printf(" %d", nxt); fflush(stdout);
        toks.push_back(nxt);
        if ((int) toks.size() >= mo.n_ctx) break;
    }
    printf("\n");
    return 0;
}
