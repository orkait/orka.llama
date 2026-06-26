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
    const ggml_tensor * lo[3];      // bit-plane index low byte (uint8), from gguf
    const ggml_tensor * hi[3];      // bit-plane index high bits packed (uint8), from gguf
    int idx_bits[3];                // index width per stage (= log2 codebook size)
    ggml_tensor * idx[3] = {};      // unpacked I32 indices (allocated at finalize)
    const ggml_tensor * cb[3];
    const ggml_tensor * scales;
    int M, K, group_size, block_size, n_stages, group_major;
    ggml_tensor * Wmat = nullptr;   // materialized W^T [K,M] (load-time decompress), else null
};

// Register/lookup a weight by its key tensor (per model load; cleared on free).
void                       llama_orka_register(const ggml_tensor * key, const llama_orka_weight & w);
const llama_orka_weight *  llama_orka_lookup(const ggml_tensor * key);
void                       llama_orka_clear();

// Load-time decompress: reconstruct every registered weight into a persistent W^T tensor
// (on the same backend buffer type as its side tensors) so decode uses a plain mul_mat
// instead of rebuilding W per token. Keeps the gguf compressed on disk; W lives in memory.
// Enabled when env ORKA_DECOMPRESS is set. No-op otherwise. Call after tensor data loads.
void                       llama_orka_materialize();

// Called after tensor data loads: unpacks each weight's bit-plane indices (lo/hi) into a
// resident I32 idx tensor (needed by ggml_get_rows), then runs llama_orka_materialize.
void                       llama_orka_finalize();

// Emit the fused dequant-matmul node: y[M, N] = W @ cur, cur is [K, N].
ggml_tensor * llama_orka_build_mm(ggml_context * ctx, const llama_orka_weight & w, ggml_tensor * cur);
