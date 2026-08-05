// orka RVQ warp-per-row GEMV for N=1 decode, consuming bit-planes directly on the GPU.
// Compressed-resident (no W materialization, no I32 index unpack): reads lo/hi index planes
// + fp16 codebooks + fp16 scales straight from the loaded gguf tensors. Wired into the ggml
// graph as a GGML_OP_CUSTOM node whose custom-fn is the sentinel `orka_warp_cpu`; ggml-cuda
// detects that fn and launches this kernel instead of the CPU fallback.
//
// Ported from orka/inference/cuda_planes.py (warp-per-row, shfl-reduce).

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"          // ggml_custom_op_params
#include "common.cuh"
#include <cuda_fp16.h>

// Per-stage index fetch. hb (hi bits per index) in {0, 2, 4, 8}: 0 is a plain byte,
// otherwise idx = lo[p] | (hi_bits << 8) with MSB-first packing (hb=8 degenerates to
// per=1, sh=0, bo=p - a full hi byte per index).
__device__ __forceinline__ int orka_idx(
    const unsigned char * __restrict__ lo, const unsigned char * __restrict__ hi,
    int p, int hb) {
    if (hb == 0) return lo[p];
    int per = 8 / hb, mask = (1 << hb) - 1;
    int sh = 8 - ((p % per) + 1) * hb;
    int bo = (p * hb) >> 3;
    return lo[p] | ((((int) __ldg(&hi[bo]) >> sh) & mask) << 8);
}

__global__ void orka_gemv_planes(
    const unsigned char * __restrict__ lo0, const unsigned char * __restrict__ hi0,
    const unsigned char * __restrict__ lo1, const unsigned char * __restrict__ hi1,
    const unsigned char * __restrict__ lo2, const unsigned char * __restrict__ hi2,
    const __half * __restrict__ cb0, const __half * __restrict__ cb1, const __half * __restrict__ cb2,
    const __half * __restrict__ scale,
    const int * __restrict__ corr_ptr, const int * __restrict__ corr_col,
    const __half * __restrict__ corr_val,
    const __half * __restrict__ x, float * __restrict__ y,
    int M, int GPR, int BPR, int GPB, int G,
    int HB0, int HB1, int HB2, int N_STAGES) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = gid >> 5, lane = gid & 31;
    if (m >= M) return;
    int GH = G >> 1;
    const int K = GPR * G;
    const __half * xn = x + (size_t) blockIdx.y * K;   // column n of x [K, N]
    const __half2 * x2 = (const __half2 *) xn;
    float acc = 0.0f;
    for (int g = lane; g < GPR; g += 32) {
        int p = m * GPR + g;
        int blk = g / GPB;
        float s = __half2float(scale[m * BPR + blk]);
        int i0 = orka_idx(lo0, hi0, p, HB0);
        int i1 = N_STAGES >= 2 ? orka_idx(lo1, hi1, p, HB1) : 0;
        int i2 = N_STAGES >= 3 ? orka_idx(lo2, hi2, p, HB2) : 0;
        const __half2 * c0 = (const __half2 *) (cb0 + i0 * G);
        const __half2 * c1 = (const __half2 *) (cb1 + i1 * G);
        const __half2 * c2 = (const __half2 *) (cb2 + i2 * G);
        const __half2 * xv = x2 + g * GH;
        float dot = 0.0f;
        for (int e = 0; e < GH; e++) {
            __half2 w = __ldg(&c0[e]);
            if (N_STAGES >= 2) w = __hadd2(w, __ldg(&c1[e]));
            if (N_STAGES >= 3) w = __hadd2(w, __ldg(&c2[e]));
            float2 wf = __half22float2(w), xf = __half22float2(__ldg(&xv[e]));
            dot += wf.x * xf.x + wf.y * xf.y;
        }
        acc += s * dot;
    }
    // CSR delta (salient/outlier exact-value corrections): y += corr @ x, lane-strided
    // over this row's nnz slice, reduced by the same shfl below.
    if (corr_ptr) {
        for (int i = corr_ptr[m] + lane; i < corr_ptr[m + 1]; i += 32) {
            acc += __half2float(__ldg(&corr_val[i])) * __half2float(__ldg(&xn[corr_col[i]]));
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) y[(size_t) blockIdx.y * M + m] = acc;
}

// sentinel custom-fn: identifies an orka-warp node. GPU-only - never executed (ggml-cuda
// intercepts before calling it); aborts if a CPU backend ever reaches it.
void orka_warp_cpu(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) dst; (void) ith; (void) nth; (void) ud;
    GGML_ABORT("orka warp op is GPU-only (built on a CUDA backend)");
}

bool ggml_cuda_is_orka_warp(const struct ggml_tensor * dst) {
    if (dst->op != GGML_OP_CUSTOM) return false;
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) dst->op_params;
    return p->fun == orka_warp_cpu;
}

void ggml_cuda_orka_warp(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const ggml_custom_op_params * p = (const ggml_custom_op_params *) dst->op_params;
    const ggml_orka_warp_args * a = (const ggml_orka_warp_args *) p->userdata;
    const __half * x = (const __half *) dst->src[0]->data;   // [K, N] fp16
    float * y = (float *) dst->data;                         // [M, N] f32
    const int N = (int) dst->ne[1];
    int th = 256;
    dim3 grid((a->M * 32 + th - 1) / th, N);
    orka_gemv_planes<<<grid, th, 0, ctx.stream()>>>(
        (const unsigned char *) a->lo[0], (const unsigned char *) a->hi[0],
        (const unsigned char *) a->lo[1], (const unsigned char *) a->hi[1],
        (const unsigned char *) a->lo[2], (const unsigned char *) a->hi[2],
        (const __half *) a->cb[0], (const __half *) a->cb[1], (const __half *) a->cb[2],
        (const __half *) a->scale,
        (const int *) a->corr_ptr, (const int *) a->corr_col, (const __half *) a->corr_val,
        x, y, a->M, a->GPR, a->BPR, a->GPB, a->G,
        a->HI_BITS[0], a->HI_BITS[1], a->HI_BITS[2], a->N_STAGES);
}

// graph builder (host): a custom node y[M,N] with the sentinel fn + args as userdata.
struct ggml_tensor * ggml_orka_warp(struct ggml_context * ctx, struct ggml_tensor * x,
                                    const struct ggml_orka_warp_args * args) {
    struct ggml_tensor * srcs[1] = { x };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, args->M, x->ne[1], 1, 1, srcs, 1,
                          orka_warp_cpu, 1, (void *) args);
}
