#include "models.h"
#include "llama-orka.h"

void llama_model_gptneox::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL,   hparams.use_par_res);
    switch (hparams.n_layer) {
        case 6:
            switch (hparams.n_ff()) {
                case 512:  type = LLM_TYPE_14M; break;
                case 2048: type = LLM_TYPE_70M; break;
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        case 12:
            switch (hparams.n_ff()) {
                case 3072: type = LLM_TYPE_160M; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 16:
            switch (hparams.n_ff()) {
                case 8192: type = LLM_TYPE_1B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 24:
            switch (hparams.n_ff()) {
                case 4096: type = LLM_TYPE_410M; break;
                case 8192: type = LLM_TYPE_1_4B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 32:
            switch (hparams.n_ff()) {
                case 10240: type = LLM_TYPE_2_8B; break;
                case 16384: type = LLM_TYPE_6_9B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 36:
            switch (hparams.n_ff()) {
                case 20480: type = LLM_TYPE_12B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 44:
            switch (hparams.n_ff()) {
                case 24576: type = LLM_TYPE_20B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gptneox::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // orka RVQ: linears are stored as side tensors (idx/cb/scales) + a fused custom op
    uint32_t orka_u = 0;
    ml.get_key(std::string("orka.rvq"), orka_u, false);
    const bool orka = orka_u != 0;
    if (orka) llama_orka_clear();
    uint32_t o_stages = 0, o_group = 0, o_block = 0, o_gmaj = 0;
    uint32_t o_cb[3] = {0,0,0};
    if (orka) {
        ml.get_key(std::string("orka.n_stages"),   o_stages);
        ml.get_key(std::string("orka.group_size"), o_group);
        ml.get_key(std::string("orka.block_size"), o_block);
        ml.get_key(std::string("orka.group_major"), o_gmaj, false);
        for (uint32_t s = 0; s < o_stages; s++)
            ml.get_key("orka.cb_size." + std::to_string(s), o_cb[s]);
    }
    // load one orka linear (M out, K in) as side tensors keyed by stage-0 index; returns key
    auto load_orka = [&](llm_tensor tid, int i, int64_t M, int64_t K) -> ggml_tensor * {
        const int64_t idx_len = M * K / o_group;
        const int64_t sc_len  = M * K / o_block;
        llama_orka_weight w{};
        w.M = (int) M; w.K = (int) K; w.group_size = o_group; w.block_size = o_block;
        w.n_stages = o_stages; w.group_major = (int) o_gmaj;
        for (uint32_t s = 0; s < o_stages; s++) {
            int bits = 0; while ((1u << bits) < o_cb[s]) bits++;        // log2(codebook size)
            w.idx_bits[s] = bits;
            int64_t hi_bytes = bits > 8 ? (idx_len * (bits - 8) + 7) / 8 : 1;
            w.lo[s] = create_tensor(tn(tid, ("weight.idxlo" + std::to_string(s)).c_str(), i), {idx_len}, 0);
            w.hi[s] = create_tensor(tn(tid, ("weight.idxhi" + std::to_string(s)).c_str(), i), {hi_bytes}, 0);
            w.cb[s]  = create_tensor(tn(tid, ("weight.cb"  + std::to_string(s)).c_str(), i), {(int64_t) o_cb[s] * o_group}, 0);
        }
        w.scales = create_tensor(tn(tid, "weight.scales", i), {sc_len}, 0);
        llama_orka_register(w.lo[0], w);
        return (ggml_tensor *) w.lo[0];
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        if (orka) {
            layer.wqkv     = load_orka(LLM_TENSOR_ATTN_QKV,  i, n_embd + 2*n_embd_gqa, n_embd);
            layer.wo       = load_orka(LLM_TENSOR_ATTN_OUT,  i, n_embd,                n_embd);
            layer.ffn_down = load_orka(LLM_TENSOR_FFN_DOWN,  i, n_embd,                n_ff);
            layer.ffn_up   = load_orka(LLM_TENSOR_FFN_UP,    i, n_ff,                  n_embd);
        } else {
            layer.wqkv     = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
            layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        }
        layer.wqkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), {n_embd + 2*n_embd_gqa}, 0);
        layer.wo_b       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gptneox::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gptneox::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // ffn
        if (hparams.use_par_res) {
            // attention and ffn are computed in parallel
            // x = x + attn(ln1(x)) + ffn(ln2(x))

            ggml_tensor * attn_out = cur;

            cur = build_norm(inpL,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, inpL);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, attn_out);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        } else {
            // attention and ffn are computed sequentially
            // x = x + attn(ln1(x))
            // x = x + ffn(ln2(x))

            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    NULL,                      NULL,                        NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }
    }

    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
