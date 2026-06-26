<div align="center">

# 🐋 orka.cpp

**A [llama.cpp](https://github.com/ggerganov/llama.cpp) fork that runs [orka](https://github.com/orkait/orka-compiler) RVQ-compressed models end to end - compressed on disk, compressed in VRAM, and decoded straight off the bit-planes.**

[![base](https://img.shields.io/badge/fork_of-llama.cpp-000000?logo=cplusplus&logoColor=white)](https://github.com/ggerganov/llama.cpp)
[![CUDA](https://img.shields.io/badge/CUDA-warp_GEMV-76B900?logo=nvidia&logoColor=white)](#-how-it-works)
[![format](https://img.shields.io/badge/format-12--bit_bit--planes-4f8ff7)](#-how-it-works)
[![decode](https://img.shields.io/badge/decode-404_tok%2Fs-7c3aed)](#-benchmarks)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

> Upstream llama.cpp docs: see [`docs/`](docs/) and the [original README in git history](https://github.com/ggerganov/llama.cpp/blob/master/README.md). This README covers the orka additions.

## 🧭 What it is

orka.cpp adds the **orka RVQ quant type** to llama.cpp. A model compressed by the
[orka-compiler](https://github.com/orkait/orka-compiler) (residual vector quantization:
per-tensor codebooks + bit-planed indices) loads and runs through the **real llama.cpp
engine** - KV cache, CUDA graphs, the lot. Decode reads the **12-bit bit-planes directly on
the GPU** via a warp-per-row kernel, so it never materializes the full-precision weights.

The headline: **because the weights stay compressed, decode reads fewer bytes - and decode
is memory-bound, so compression *is* the speed win.**

## ⚡ Benchmarks

RTX 3060 (12 GB), `-ngl 99`. Decode = autoregressive generation (N=1), the hot path.

### Speed - pythia-160m (gptneox), 3-stage RVQ @ ~4.5 bpw, 147 MB gguf

| Mode | Prefill tok/s | Decode tok/s | Notes |
|---|--:|--:|---|
| `mul_mat` (materialized Q8) | 10,236 | 110 | dequantizes to Q8, then GEMV |
| **warp kernel** (default) | 254 | **404** | reads bit-planes, no W materialized |
| **hybrid** (`ORKA_DECOMPRESS=1`) | **37,790** | **404** | Q8 GEMM prefill + warp decode |

Decode is **3.6x faster than the materialized path** - the compressed form is the fast form.

### Speed - SmolLM-135M (llama arch), same recipe, 170 MB gguf

| Mode | Prefill tok/s | Decode tok/s |
|---|--:|--:|
| warp (default) | 191 | **255** |
| hybrid | **22,491** | 255 |

Two architecture families (gptneox + llama) run the identical engine - the warp kernel,
loader hook, and bit-plane path are arch-agnostic.

### Quality - equal-bit perplexity vs bitsandbytes nf4

Same 4.5 bits/param on the linears, same eval text. Lower is better.

| Model | arch | fp16 | bnb-nf4 (4-bit) | **orka 3-stage** | winner |
|---|---|--:|--:|--:|---|
| pythia-160m | gptneox | 2.57 | 3.26 | **3.21** | **orka** |
| Qwen2.5-0.5B | qwen2 | 2.00 | 2.035 | 2.044 | bnb (+0.4%) |
| SmolLM-135M | llama | 1.954 | 2.144 | **1.977** | **orka (+8%)** |

At matched bits, orka's RVQ is **competitive-to-better than best-of-breed 4-bit** - it wins
2 of 3, decisively on SmolLM (within 1.2% of fp16 while nf4 is +9.7%).

## 🔬 How it works

```
weight [out, in]
  ──pack──▶  per-stage codebook (4096 entries) + indices
  ──store─▶  indices as 12-bit BIT-PLANES (lo uint8 + packed hi)   ← 2.7x tighter than int32
  ──decode▶  warp-per-row GEMV reads planes + cached codebook on GPU
```

- **Bit-planes**: each index is `lo (8 bits) | hi (width-8 bits, packed)`. 12-bit indices,
  not int32 - the disk and the DRAM traffic shrink with the compression.
- **Warp kernel** (`ggml/src/ggml-cuda/orka-rvq.cu`): one warp per output row, sums the RVQ
  stages, shfl-reduces. The codebook is tiny (~64 KB) and **L2-cached** - so decode streams
  only the indices from DRAM, never the weights. That is why it beats `mul_mat`.
- **Routing**: N=1 decode → warp kernel; N>1 prefill → materialized Q8 GEMM (cuBLAS-fast).
- **Output head stays Q8** - the head is RVQ-fragile (ppl explodes); int8 is lossless.

<details>
<summary>📐 The pieces added to llama.cpp</summary>

| File | Role |
|---|---|
| `src/llama-orka.{h,cpp}` | weight registry, bit-plane unpack (`llama_orka_finalize`), `build_mm` routing |
| `ggml/src/ggml-cuda/orka-rvq.cu` | warp-per-row GEMV CUDA kernel + `GGML_OP_CUSTOM` dispatch |
| `ggml/include/ggml-cuda.h` | `ggml_orka_warp` builder + args |
| `src/llama-graph.cpp` | `build_lora_mm` hook (routes any orka weight through the op) |
| `src/models/{gptneox,llama}.cpp` | per-arch loader branches (load bit-plane side tensors) |
| `src/llama-model-loader.cpp` | force orka side tensors onto the GPU buffer (the placement fix) |

</details>

## 🚀 Quick start

```bash
# 1. compress a model with orka-compiler -> a .orka artifact (see that repo)
# 2. convert .orka -> a llama.cpp-native bit-plane gguf
python scripts/export_gguf_llama.py  model.orka  <hf-model-dir>  model.gguf   # llama arch
# (or export_gguf_llamacpp.py for gptneox)

# 3. build with CUDA
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# 4. run
./build/bin/llama-cli -m model.gguf -ngl 99 -p "..."              # warp decode (default)
ORKA_DECOMPRESS=1 ./build/bin/llama-bench -m model.gguf -ngl 99   # hybrid (fast prefill)
```

> The `scripts/export_gguf_*.py` converters live in the
> [orka-compiler](https://github.com/orkait/orka-compiler) repo.

## 🔄 Staying current with llama.cpp

```bash
git fetch upstream && git rebase upstream/master && git push --force-with-lease origin main
```

## 📄 License

MIT, same as upstream llama.cpp. All credit to the
[llama.cpp / ggml](https://github.com/ggerganov/llama.cpp) authors; orka.cpp only adds the
RVQ quant type and kernels.
