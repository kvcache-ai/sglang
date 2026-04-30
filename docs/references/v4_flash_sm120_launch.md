# DeepSeek-V4-Flash on 8x RTX 5090 (SM_120) — Launch Recipes

This page collects the verified launch commands for running V4-Flash on
8x RTX 5090 (consumer Blackwell, SM_120) under the v4-trial branch.
Each recipe includes the role of every flag/env, the measured
single-request decode throughput on the reference hardware, and the
trade-offs.

The reference hardware is `yyj@192.168.200.5`: 2x AMD EPYC 9355 (128
logical cores, 2 NUMA nodes), 1.5 TB RAM, 8x RTX 5090 (32 GB each,
SM_120, CUDA 12.8). The reference model is `DeepSeek-V4-Flash` (149
GB, n_routed_experts=256, num_hidden_layers=43, MXFP4 routed experts +
FP8 e4m3 attention/shared).

Throughput numbers below are 5-run steady-state on the prompt
`"The capital of France is"` with `max_new_tokens=48,
temperature=0.0` (greedy), unless stated otherwise.

## Common environment

All recipes share these env vars to work around SM_120 + CUDA 12.8 +
flashinfer 0.6.8 quirks. They are all opt-in and have no effect on
non-V4 / non-SM_120 runs.

```bash
# JIT toggles - DeepGEMM is Hopper-only; tilelang indexer replaces
# the deep_gemm path; FP8 paged-MQA logits use a torch fallback to
# avoid DeepGEMM dispatch.
SGLANG_ENABLE_JIT_DEEPGEMM=0
SGLANG_OPT_USE_TILELANG_INDEXER=1
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1

# V4 attention dispatch - kernel path uses our triton sparse-FP8 MLA
# decode fallback (vLLM PR #40929 ported); flash_mla wheel does not
# support SM_120.
SGLANG_HACK_FLASHMLA_BACKEND=kernel

# Routed-MoE on GPU - drives the OAI triton_kernels matmul_ogs path
# (avoids the broken marlin and the unavailable DeepGEMM_MXFP4).
SGLANG_V4_USE_TRITON_KERNELS=1

# Bypass flashinfer 0.6.8 Python-to-C++ binding skew on T>0 +
# top_k/top_p/min_p sampling. See `Sampling on flashinfer 0.6.8` below.
SGLANG_FLASHINFER_SAMPLING_WORKAROUND=1

# flashinfer JIT capability gate - CUDA 12.8 misdetects SM_120 without
# these.
FLASHINFER_CUDA_ARCH_LIST=12.0a
TORCH_CUDA_ARCH_LIST=12.0+PTX
```

## Recipe A: Pure sglang + CUDA Graph (no kt) — 50.5 tok/s

The fastest path. All 256 routed experts on GPU via the OAI
matmul_ogs MXFP4 kernel; CUDA Graph captures the full forward.

```bash
env \
  SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_OPT_USE_TILELANG_INDEXER=1 \
  SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
  SGLANG_HACK_FLASHMLA_BACKEND=kernel \
  SGLANG_V4_USE_TRITON_KERNELS=1 \
  SGLANG_FLASHINFER_SAMPLING_WORKAROUND=1 \
  FLASHINFER_CUDA_ARCH_LIST=12.0a \
  TORCH_CUDA_ARCH_LIST=12.0+PTX \
  numactl --interleave=all python -m sglang.launch_server \
    --host 127.0.0.1 --port 30000 \
    --model /mnt/data/models/DeepSeek-V4-Flash \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --mem-fraction-static 0.85 \
    --context-length 4096 \
    --max-running-requests 4 \
    --max-total-tokens 8192 \
    --chunked-prefill-size 2048 \
    --watchdog-timeout 3000 \
    --attention-backend flashinfer \
    --disable-shared-experts-fusion \
    --cuda-graph-bs 1 2 4 \
    --cuda-graph-max-bs 4 \
    --moe-runner-backend flashinfer_mxfp4 \
    --fp8-gemm-backend triton
```

Result: 27 tok / 0.954 s = **50.5 tok/s** steady (5-run mean of
runs 2-5). T=0.7 + top_p=0.9 + top_k=50: 39 tok/s.

Critical flags:
- `--moe-runner-backend flashinfer_mxfp4` is required. Without it the
  dispatch picks Fp8MoEMethod and the first forward asserts in
  `fused_moe.py:386 Hidden size mismatch`.
- `SGLANG_V4_USE_TRITON_KERNELS=1` selects the OAI matmul_ogs MoE
  kernel; the sgl_kernel marlin MoE path on SM_120 has unresolved
  numerical issues (see `project_v4_marlin_root_cause.md`).

## Recipe B: Hybrid 144 GPU + 112 CPU + CUDA Graph — 12.44 tok/s

The lightest hybrid configuration. 144 routed experts on GPU
(matmul_ogs), 112 on CPU (kt-kernel AVX2). CUDA Graph captures the
GPU-side work; kt-kernel uses `cudaLaunchHostFunc` host nodes so the
CPU-expert submit/sync calls are graph-recordable too.

```bash
env \
  SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_OPT_USE_TILELANG_INDEXER=1 \
  SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
  SGLANG_HACK_FLASHMLA_BACKEND=kernel \
  SGLANG_V4_USE_TRITON_KERNELS=1 \
  SGLANG_FLASHINFER_SAMPLING_WORKAROUND=1 \
  FLASHINFER_CUDA_ARCH_LIST=12.0a \
  TORCH_CUDA_ARCH_LIST=12.0+PTX \
  numactl --interleave=all python -m sglang.launch_server \
    --host 127.0.0.1 --port 30000 \
    --model /mnt/data/models/DeepSeek-V4-Flash \
    --kt-weight-path /mnt/data/models/DeepSeek-V4-Flash \
    --kt-cpuinfer 8 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 144 \
    --kt-method MXFP4 \
    --kt-gpu-prefill-token-threshold 4096 \
    --kt-enable-dynamic-expert-update \
    --attention-backend flashinfer \
    --trust-remote-code \
    --mem-fraction-static 0.80 \
    --chunked-prefill-size 2048 \
    --max-running-requests 4 \
    --max-total-tokens 8192 \
    --watchdog-timeout 3000 \
    --tensor-parallel-size 8 \
    --disable-shared-experts-fusion \
    --cuda-graph-bs 1 2 4 \
    --cuda-graph-max-bs 4 \
    --fp8-gemm-backend triton
```

Result: 27 tok / 2.17 s = **12.44 tok/s** steady.

Key tuning:
- `--kt-cpuinfer 8` — number of AVX2 worker threads per kt-kernel
  thread-pool. Settings `>= 32` saturate L3 / cross-NUMA bandwidth on
  the dual-EPYC and starve the Python kernel-launch path; observed
  TP0 = 1118 % CPU at `--kt-cpuinfer 120`, dropping to 299 % at 8.
  CPU expert work for 1-token decode is small enough that 8 threads
  finish well before the GPU side, so the pool just needs to be
  large enough to handle the active expert count, not large in
  absolute terms.
- `--kt-num-gpu-experts 144` — 56 % of routed experts on GPU. Fewer
  GPU experts moves work to CPU but does not speed up steady-state
  decode (CPU was never the critical path); see Recipe C for higher
  GPU-expert counts.
- `--kt-enable-dynamic-expert-update` — required. Removing it drops
  steady-state throughput by ~17 % because the prefill-time hot
  expert reassignment is what makes the GPU mask cover the most-
  activated experts during decode.
- `--disable-shared-experts-fusion` — required. Re-enabling fusion
  drops throughput by ~15 % on this path.
- Do NOT pass `--enable-mixed-chunk` or `--enable-p2p-check`; both
  add ~10-15 % overhead in the hybrid path with no benefit at
  decode batch size 1.

## Recipe C: Hybrid 196 GPU + 60 CPU + CUDA Graph — 13.95 tok/s

Same as Recipe B but with more routed experts pinned to GPU. The
CPU contribution is smaller, so the per-layer cudaLaunchHostFunc +
H2D copy overhead shrinks. Steady-state crosses pure-no-CG (13.82)
without giving up the kt-kernel safety net.

Identical command to Recipe B except:

```bash
    --kt-num-gpu-experts 196 \
```

Result: 27 tok / 1.93 s = **13.95 tok/s** steady. Output drifts
slightly vs Recipe B because the routing differs (different
gpu_experts_mask), but both are valid.

## Recipe D: Hybrid no CUDA Graph (compatibility fallback) — 9.18 tok/s

If CUDA Graph capture fails for any reason (e.g. an upstream sglang
or kt-kernel change breaks host-node compatibility), this is the
fallback. Same as Recipe B but with `--disable-cuda-graph` instead
of the `--cuda-graph-*` flags.

```bash
# replace the --cuda-graph-bs / --cuda-graph-max-bs lines with:
    --disable-cuda-graph \
```

Result: 27 tok / 2.94 s = **9.18 tok/s** steady. Loses 26 % vs
Recipe B but is more robust to upstream churn.

## Recipe E: Plan C — 100 % CPU MoE (debug / minimum-GPU) — 5-6 tok/s

All 256 routed experts on CPU; GPU does only attention + shared
experts + lm_head. Use only when GPU MoE is broken or for debugging
the kt-kernel path in isolation.

```bash
    --kt-cpuinfer 120 \
    --kt-num-gpu-experts 0 \
    --kt-method MXFP4 \
    --disable-cuda-graph \
    # plus SGLANG_KT_BYPASS_GPU_MOE=1
```

Result: ~5-6 tok/s. Bottlenecked by CPU AVX2 throughput on 256
experts per token.

## Sampling on flashinfer 0.6.8

The flashinfer wheel installed in venv2 has a Python-to-C++ binding
skew at `flashinfer/sampling.py:425`. The Python wrapper passes 13
args (probs, samples, valid, indices, top_k_arr, top_k_val,
top_p_arr, top_p_val, deterministic, seed_arr, seed_val, offset_arr,
offset_val) but the C++ binding expects 10 (no valid, no seed_arr,
no offset_arr — the CG-friendly tensor seed/offset that the new
Python API added).

The first request with `temperature > 0` AND any of `top_k / top_p /
min_p` raises a TVM-FFI TypeError that kills every TP scheduler
process. T=0 (greedy `argmax`) and `T>0` without any filter
(`torch.multinomial` directly) both bypass the broken wrapper.

`SGLANG_FLASHINFER_SAMPLING_WORKAROUND=1` reroutes the
flashinfer-backend `top_k / top_p / min_p` path through
`top_k_renorm_prob + top_p_renorm_prob + (manual min_p mask) +
torch.multinomial`. The renorm kernels come from `sgl_kernel`
(CUDA, not torch); the final `torch.multinomial` is a single
PyTorch op. Cost is about 50-100 us per step — under 0.2 % of e2e
at our scale and CG-safe.

Default off so any environment with a matching wheel is unaffected.
All recipes above set the env, which is recommended for any
launch on this venv2 build until the binding is rebuilt.

## Quick perf summary

| Path | tok/s (T=0) | Notes |
|---|---:|---|
| Pure sglang + CG (Recipe A) | 50.5 | Fastest, no kt, all 256 GPU experts |
| Hybrid 196 + CG (Recipe C) | 13.95 | Beats pure-no-CG with kt safety |
| Hybrid 144 + CG (Recipe B) | 12.44 | Lighter GPU mem, same CG path |
| Hybrid no-CG (Recipe D) | 9.18 | Compatibility fallback |
| Plan C CPU-only (Recipe E) | 5-6 | Debug / GPU-MoE-broken-only |
