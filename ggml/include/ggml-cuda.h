#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// orka RVQ warp GEMV (compressed-resident). args' pointers reference the loaded gguf
// tensors (device, persistent); the caller owns args for the graph's lifetime.
// HI_BITS is per stage (heterogeneous widths: e.g. 16-bit stage 0 + 8-bit stage 1).
// corr_* is an optional CSR delta (salient/outlier exact-value corrections):
// y = W_rvq @ x + corr @ x; all three null when the weight carries no correction.
struct ggml_orka_warp_args {
    const void * lo[3];
    const void * hi[3];
    const void * cb[3];
    const void * scale;
    const void * corr_ptr;   // int32 [M+1] or null
    const void * corr_col;   // int32 [nnz]
    const void * corr_val;   // fp16  [nnz]
    int M, GPR, BPR, GPB, G, N_STAGES;
    int HI_BITS[3];
};
// Build a node y[M,N] = W @ x (x must be F16 [K,N]; one warp per output row per column).
GGML_API struct ggml_tensor * ggml_orka_warp(struct ggml_context * ctx, struct ggml_tensor * x,
                                             const struct ggml_orka_warp_args * args);

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
