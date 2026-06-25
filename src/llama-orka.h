#pragma once

// Orka RVQ weight support for llama.cpp.
//
// A quantized linear is stored as side tensors (per-stage int16 indices + fp16 codebooks
// + fp16 block scales) rather than one self-contained block tensor, because orka uses a
// per-tensor codebook. We register those side tensors against a key tensor (the stage-0
// index tensor, reused as layer.wqkv/wo/ffn_*), and build_lora_mm() routes any registered
// weight through the fused dequant-matmul custom op instead of ggml_mul_mat.

#include "ggml.h"
#include <cstdint>

struct llama_orka_weight {
    const ggml_tensor * idx[3];
    const ggml_tensor * cb[3];
    const ggml_tensor * scales;
    int M, K, group_size, block_size, n_stages, group_major;
};

// Register/lookup a weight by its key tensor (per model load; cleared on free).
void                       llama_orka_register(const ggml_tensor * key, const llama_orka_weight & w);
const llama_orka_weight *  llama_orka_lookup(const ggml_tensor * key);
void                       llama_orka_clear();

// Emit the fused dequant-matmul node: y[M, N] = W @ cur, cur is [K, N].
ggml_tensor * llama_orka_build_mm(ggml_context * ctx, const llama_orka_weight & w, ggml_tensor * cur);
