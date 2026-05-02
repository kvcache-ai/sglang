# SPDX-License-Identifier: Apache-2.0
"""
V4-Flash MXFP4 GPU MoE via OpenAI's `triton_kernels` package.

Default GPU MoE path for V4-Flash on every capability outside the trtllm
binary whitelist (`_TRTLLM_FP4_CAPS`, currently {(10,0)} = SM_100 datacenter
Blackwell only). Used on consumer Blackwell (SM_120, e.g. RTX 5090), Ada
(SM_89, e.g. L40S, RTX 4090), Hopper (SM_90), and Ampere — anywhere
flashinfer's `trtllm_fp4_block_scale_routed_moe` lacks a CUDA binary.

The OAI `triton_kernels` package's `matmul_ogs` (gather-or-scatter matmul)
provides a clean Triton MXFP4 path that:
- accepts FP4 packed weights + ue8m0 scales via either an upstream Hopper /
  Blackwell-DC swizzle (`_swizzle_mxfp4` from `mxfp4.py`) or a portable
  StridedLayout (`_swizzle_mxfp4_strided` here, used everywhere outside the
  trtllm whitelist),
- composes naturally with sglang's standard topk (we convert topk_ids /
  topk_weights → bitmatrix → routing_from_bitmatrix the same way the
  OAI vLLM port (PR #18595) does),
- runs the same Triton kernel that sglang already uses for unquantized
  bf16 MoE, so the kernel is on a tested code path; the only new
  ingredient is the FP4 weight + ue8m0 scale wiring.

Origin: sglang 本身.

Selection (default, capability-driven; see `mxfp4_deepseek.py`):
  cap == (10,0)             -> trtllm  (sm100f binary)
  cap not in whitelist      -> this module (StridedLayout + simulated MXFP)

Force-override env (diagnostic only):
  SGLANG_V4_USE_TRITON_KERNELS=1 -> force this module even on (10,0)
  SGLANG_V4_USE_TRITON_KERNELS=0 -> force trtllm even off-whitelist (fail loud)
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def use_v4_triton_kernels() -> bool:
    """Force-override gate for the V4 triton_kernels path.

    Default behavior (env unset): the dispatcher in
    `mxfp4_deepseek.process_weights_after_loading` chooses this path for any
    capability outside `_TRTLLM_FP4_CAPS`, so this returning False does not
    mean the path is disabled — only that no override was requested.

    SGLANG_V4_USE_TRITON_KERNELS=1: force this path even on whitelisted
    capabilities (numerical comparison / debugging).

    SGLANG_V4_USE_TRITON_KERNELS=0: force trtllm even off-whitelist (will
    fail loud with "Unsupported architecture" — kept as a diagnostic exit).

    Origin: sglang 本身."""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "1"


def force_disable_v4_triton_kernels() -> bool:
    """True when the user explicitly set SGLANG_V4_USE_TRITON_KERNELS=0 to
    force trtllm even off-whitelist (diagnostic only). Origin: sglang 本身."""
    return os.environ.get("SGLANG_V4_USE_TRITON_KERNELS") == "0"


# -----------------------------------------------------------------------------
# Bitmatrix construction (port of vLLM gpt_oss_triton_kernels_moe.pack_bitmatrix)
# -----------------------------------------------------------------------------


@triton.jit
def _pack_bitmatrix_v4(
    bitmatrix_ptr,      # uint32 [n_rows, bm_cols]
    topk_ids_ptr,       # int16 [n_rows, n_expts_act]
    n_rows,
    bm_cols: tl.constexpr,
    n_expts_act,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # 32 (bits per uint32)
):
    """For each token row, set the bits of expert ids selected in topk_ids.

    bitmatrix[r, e // 32] |= (1 << (e % 32)) for each e in topk_ids[r, :].
    Uses tl.atomic_or to handle multiple bits in the same uint32 word.
    """
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < n_rows

    # Iterate through topk slots (small, ≤ 16 typically)
    for k in range(n_expts_act):
        ids = tl.load(
            topk_ids_ptr + rows * n_expts_act + k,
            mask=row_mask,
            other=-1,
        ).to(tl.int32)
        valid = (ids >= 0) & row_mask
        col = ids // BLOCK_SIZE_K
        bit = ids - col * BLOCK_SIZE_K  # ids % 32
        ptrs = bitmatrix_ptr + rows * bm_cols + col
        tl.atomic_or(ptrs, (1 << bit).to(tl.uint32), mask=valid)


def _make_routing_data_v4(
    topk_ids: torch.Tensor,    # [M, n_topk] int (any int dtype)
    topk_weights: torch.Tensor,  # [M, n_topk] float
    num_local_experts: int,
):
    """Convert sglang's standard (topk_ids, topk_weights) to triton_kernels'
    (RoutingData, GatherIndx, ScatterIndx) via the bitmatrix path.

    Mirrors vLLM `make_routing_data` (gpt_oss_triton_kernels_moe.py).

    Note: triton_kernels' internal routing kernel
    (routing_details/_routing_compute._routing_compute_indx) does
    `tl.arange(0, N_EXPTS_ACT * BLOCK_M)` which requires the product to be
    a power of 2; equivalently `n_topk` must itself be a power of 2.
    V4-Flash uses top-6, which is not — so we pad to the next power of 2
    by appending invalid (-1) slots, which the kernel masks out via the
    `expt_indx == -1` check (also matches matmul_ogs's gammas == -1
    convention). Pad cost is ~33% extra slot bookkeeping; gemm work is
    unchanged because invalid slots are not routed.
    """
    from triton_kernels.routing import routing_from_bitmatrix
    from triton_kernels.tensor import Bitmatrix

    topk_ids_i16 = topk_ids.to(torch.int16).contiguous()
    topk_weights_bf = topk_weights.to(torch.bfloat16).contiguous()

    n_rows, n_topk_raw = topk_ids_i16.shape

    # Pad n_topk to next power of 2 (V4: 6 -> 8) for triton_kernels routing.
    n_topk = 1
    while n_topk < n_topk_raw:
        n_topk *= 2
    if n_topk != n_topk_raw:
        pad_len = n_topk - n_topk_raw
        pad_ids = torch.full(
            (n_rows, pad_len), -1, dtype=torch.int16, device=topk_ids_i16.device
        )
        pad_w = torch.full(
            (n_rows, pad_len), -1.0,
            dtype=torch.bfloat16, device=topk_ids_i16.device,
        )
        topk_ids_i16 = torch.cat([topk_ids_i16, pad_ids], dim=1).contiguous()
        topk_weights_bf = torch.cat([topk_weights_bf, pad_w], dim=1).contiguous()

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_local_experts, BLOCK_SIZE_K)
    bitmatrix_data = torch.zeros(
        (n_rows, bm_cols),
        dtype=torch.uint32,
        device=topk_ids_i16.device,
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    _pack_bitmatrix_v4[grid](
        bitmatrix_data,
        topk_ids_i16,
        n_rows,
        bm_cols,
        n_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bitmatrix = Bitmatrix(
        bitmatrix_data,
        shape=[n_rows, bm_cols * 32],
        shape_max=[n_rows, None],
        scratchpad=None,
    )

    # matmul_ogs convention: invalid topk weights are -1.0 (not 0).
    # Use masked_fill (Python-scalar) instead of torch.where(...,
    # torch.tensor(-1.0, ...), ...) so this is CUDA-graph-capture safe
    # (no H2D copy during capture).
    topk_weights_bf = topk_weights_bf.masked_fill(topk_ids_i16 == -1, -1.0)

    return routing_from_bitmatrix(
        bitmatrix, topk_weights_bf, topk_ids_i16, num_local_experts, n_topk
    )


# -----------------------------------------------------------------------------
# Weight conversion: raw V4 MXFP4 → triton_kernels-format
# -----------------------------------------------------------------------------


# Capabilities for which flashinfer's `trtllm_fp4_block_scale_routed_moe`
# ships a working binary. Verified on flashinfer 0.6.8: only sm100f.
# Outside this set, we use the StridedLayout + simulated-MXFP non-persistent
# matmul_ogs path here. Keep this in sync with the same constant in
# `mxfp4_deepseek.py`.
_TRTLLM_FP4_CAPS = {(10, 0)}


def _use_strided_layout() -> bool:
    """True when matmul_ogs must use StridedLayout + simulated MXFP rather
    than the upstream Hopper-TMA / Blackwell-DC swizzle. Origin: sglang 本身.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() not in _TRTLLM_FP4_CAPS


def _patch_strided_mxfp():
    """Strided-layout enablement for triton_kernels MXFP4 path. Two patches:

    (1) `target_info.has_native_mxfp()` must return False on non-trtllm-
        whitelist capabilities so matmul_ogs takes the simulated MXFP non-
        persistent code path instead of the native (TMA + cluster shared
        mem + tile::gather4) path that needs Hopper / DC-Blackwell features
        consumer / Ada / Ampere don't have. (On SM_89/SM_80 the upstream
        `has_native_mxfp` already returns False; force-False is a no-op
        but cheap. On SM_120 the upstream returns True since cap[0]==12
        is treated as Blackwell, hence the override.)

    (2) opt_flags must force `is_persistent=False`. The auto-selection in
        make_opt_flags chooses persistent for larger M (= prefill batches),
        which then collides with simulated-MXFP and raises 'Must use non-
        persistent kernel for simulated MXFP' at runtime. CG decode (small
        M) auto-selects non-persistent so it works without this; prefill
        needs the explicit constraint.

    Origin: sglang 本身 (triton_kernels package's Blackwell layout assumes
    SM_100 features; everything outside the trtllm whitelist falls in a
    similar gap)."""
    if not _use_strided_layout():
        return
    import triton_kernels.target_info as target_info
    if not getattr(target_info, "_v4_strided_patched", False):
        original = target_info.has_native_mxfp
        def has_native_mxfp_strided():
            if _use_strided_layout():
                return False
            return original()
        target_info.has_native_mxfp = has_native_mxfp_strided
        target_info._v4_strided_patched = True
    # opt_flags imports has_native_mxfp into its namespace at import time;
    # refresh the binding there too. Also force is_persistent=False so the
    # auto-selector in make_opt_flags can't pick the native path.
    try:
        import triton_kernels.matmul_ogs_details.opt_flags as _of
        if hasattr(_of, "has_native_mxfp"):
            _of.has_native_mxfp = target_info.has_native_mxfp
        _of.update_opt_flags_constraints({"is_persistent": False})
    except Exception:
        pass


def _swizzle_mxfp4_strided(quant_tensor: torch.Tensor, scale: torch.Tensor):
    """Wrap raw MXFP4 weight + ue8m0 scale into triton_kernels Tensor objects
    using StridedLayout (no swizzle), suitable for the simulated-MXFP non-
    persistent path on capabilities outside the trtllm whitelist. Origin:
    sglang 本身.

    Matches the API contract of sglang.srt.layers.quantization.mxfp4
    `_swizzle_mxfp4`: returns (Tensor[FP4], InFlexData(), Tensor[ue8m0])."""
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), StridedLayout
    )
    scale = convert_layout(wrap_torch_tensor(scale), StridedLayout)
    return quant_tensor, InFlexData(), scale


def convert_v4_weights_to_triton_kernels(
    w13: torch.Tensor,        # [E, 2*N_int, K//2] int8 packed FP4
    w13_scale: torch.Tensor,  # [E, 2*N_int, K//group] float8_e8m0fnu (or uint8)
    w2: torch.Tensor,         # [E, K, N_int//2] int8 packed FP4
    w2_scale: torch.Tensor,   # [E, K, N_int//group] float8_e8m0fnu
    *,
    num_warps: int = 4,
) -> Tuple:
    """Apply `_swizzle_mxfp4` from sglang.srt.layers.quantization.mxfp4 to
    each weight tensor and build the matching `PrecisionConfig`. The same
    swizzle is used by sglang's existing OAI MXFP4 path (see `mxfp4.py:136`).

    Returns:
        (w13_swiz, w13_pcg, w2_swiz, w2_pcg)
        where w13/w2 are `triton_kernels.Tensor` (FP4 layout) and
        pcg are `triton_kernels.matmul_ogs.PrecisionConfig`.
    """
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    _patch_strided_mxfp()

    # Wrap raw scale as float8_e8m0fnu if it came in as uint8/float32.
    if w13_scale.dtype != torch.float8_e8m0fnu:
        if w13_scale.dtype == torch.uint8:
            w13_scale = w13_scale.view(torch.float8_e8m0fnu)
        elif w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
    if w2_scale.dtype != torch.float8_e8m0fnu:
        if w2_scale.dtype == torch.uint8:
            w2_scale = w2_scale.view(torch.float8_e8m0fnu)
        elif w2_scale.dtype == torch.float32:
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)

    # The packed FP4 weight is stored as int8 in safetensors; the matmul_ogs
    # kernel asserts the underlying torch dtype is uint8 (or fp8). View
    # without copy.
    if w13.dtype != torch.uint8:
        w13 = w13.view(torch.uint8)
    if w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    if _use_strided_layout():
        # Non-trtllm-whitelist capability: bypass Blackwell DC swizzle (which
        # needs Hopper TMA / cluster shared mem) and use StridedLayout +
        # simulated MXFP non-persistent kernel.
        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4_strided(w13, w13_scale)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4_strided(w2, w2_scale)
    else:
        # SM_100 (cap=(10,0)): use the upstream swizzle from sglang.mxfp4.
        # Reached only via the SGLANG_V4_USE_TRITON_KERNELS=1 force-override;
        # the default dispatch routes this capability to the trtllm path.
        from sglang.srt.layers.quantization.mxfp4 import _swizzle_mxfp4
        w13_swiz, w13_flex, w13_scale_swiz = _swizzle_mxfp4(w13, w13_scale, num_warps)
        w2_swiz, w2_flex, w2_scale_swiz = _swizzle_mxfp4(w2, w2_scale, num_warps)

    w13_pcg = PrecisionConfig(
        weight_scale=w13_scale_swiz,
        flex_ctx=FlexCtx(rhs_data=w13_flex),
    )
    w2_pcg = PrecisionConfig(
        weight_scale=w2_scale_swiz,
        flex_ctx=FlexCtx(rhs_data=w2_flex),
    )

    return w13_swiz, w13_pcg, w2_swiz, w2_pcg


# -----------------------------------------------------------------------------
# Apply: V4-Flash MoE forward via matmul_ogs
# -----------------------------------------------------------------------------


def apply_v4_triton_kernels_moe(
    *,
    hidden_states: torch.Tensor,    # [M, K] bf16
    w13_swiz,                        # triton_kernels.Tensor (FP4) [E, K, 2*N]
    w13_pcg,                         # PrecisionConfig
    w2_swiz,                         # triton_kernels.Tensor (FP4) [E, N, K]
    w2_pcg,                          # PrecisionConfig
    topk_weights: torch.Tensor,     # [M, n_topk] bf16/float
    topk_ids: torch.Tensor,         # [M, n_topk] int
    intermediate_size: int,         # per-partition N
    num_experts: int,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Run V4 sparse MoE through `triton_kernels.matmul_ogs`.

    Deterministic across runs (no atomic-add reduction order issues) and
    byte-level reproducible under same input.

    Activation: silu_and_mul (V4 default), applied between the two GEMMs.
    """
    from triton_kernels.matmul_ogs import matmul_ogs
    from sgl_kernel import silu_and_mul

    # Refresh strided-layout patch (cheap idempotent guard) in case apply
    # runs in a process where target_info was re-imported.
    _patch_strided_mxfp()

    M, K = hidden_states.shape
    N = intermediate_size

    # Build routing data from sglang topk → triton_kernels (RoutingData,
    # GatherIndx, ScatterIndx). Note: this rebuilds per-call. Cheap
    # (O(M * n_topk)) compared to the gemms themselves.
    routing_data, gather_indx, scatter_indx = _make_routing_data_v4(
        topk_ids, topk_weights, num_experts
    )

    # gemm1: hidden_states (M, K) @ w13 → (M*topk, 2*N) bf16
    intermediate1 = matmul_ogs(
        hidden_states,
        w13_swiz,
        None,  # bias
        routing_data,
        gather_indx=gather_indx,
        precision_config=w13_pcg,
    )
    # intermediate1 shape: [M*topk, 2*N]
    M_topk = intermediate1.shape[0]
    intermediate2 = torch.empty(
        (M_topk, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    silu_and_mul(intermediate1.view(-1, 2 * N), intermediate2)

    # gemm2: (M*topk, N) @ w2 → (M, K), with gammas=topk_weights for combine
    output = matmul_ogs(
        intermediate2,
        w2_swiz,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_pcg,
        gammas=routing_data.gate_scal,
    )

    if routed_scaling_factor != 1.0:
        output = output * routed_scaling_factor

    return output
