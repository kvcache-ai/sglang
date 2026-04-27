# SPDX-License-Identifier: Apache-2.0
#
# DeepSeek V4 Flash MXFP4 MoE on hardware where the flashinfer TRT-LLM kernel
# is unavailable (notably SM_120 / RTX 5090 — the prebuilt
# `bmm_..._sm100f` binary in `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`
# only loads on Blackwell datacenter cards).
#
# sgl-kernel's `moe_wna16_marlin_gemm` natively understands
# `b_q_type == float4_e2m1f` with `group_size == 32` and reads scales as
# `Float8_e8m0fnu` (sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu:1126-1132,
# 1178-1184). That matches V4 exactly, so the only work here is wiring sglang
# to call the Marlin kernel and repacking the V4 raw weight layout into the
# layout Marlin expects.
#
# Origin: sglang 本身 (V4 quantizer is sglang trunk PR #23600). No kt-kernel
# changes.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import functools
import os

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import moe_sum_reduce
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

# Once-per-rank diagnostic prints. SGLANG_V4_MARLIN_DEBUG=1 turns on a
# numerical sanity log on the first call for each sub-step.
_DEBUG = os.environ.get("SGLANG_V4_MARLIN_DEBUG") == "1"
# SGLANG_V4_MARLIN_BYPASS=1 makes apply_v4_marlin_moe return zeros so we can
# isolate whether numerical garbage downstream comes from this MoE path or
# from somewhere else.
_BYPASS = os.environ.get("SGLANG_V4_MARLIN_BYPASS") == "1"
_diag_done = {"convert": False, "gemm1": False, "silu": False, "gemm2": False}


def _diag(name, t):
    if not _DEBUG or _diag_done.get(name):
        return
    _diag_done[name] = True
    flat = t.detach().to(torch.float32).flatten()
    sample = flat[: min(8, flat.numel())].tolist()
    print(
        f"[v4-marlin-diag] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"any_nan={torch.isnan(flat).any().item()} "
        f"any_inf={torch.isinf(flat).any().item()} "
        f"min={flat.min().item():.4g} max={flat.max().item():.4g} "
        f"absmean={flat.abs().mean().item():.4g} sample={sample}",
        flush=True,
    )

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

logger = logging.getLogger(__name__)

# V4 Flash uses the standard MXFP4 group of 32 fp4 weights per scale.
V4_FP4_GROUP_SIZE = 32
V4_NUM_BITS = 4
_PACK_FACTOR = 32 // V4_NUM_BITS  # 8 fp4 values per int32


def _v4_pack_to_gptq_int32(w_uint8: torch.Tensor) -> torch.Tensor:
    """V4 raw layout -> GPTQ-style packed int32 expected by gptq_marlin_moe_repack.

    V4 weights are stored as ``[E, N_out, K_in // 2]`` (int8) where each byte
    holds two fp4 values: low nibble = even k, high nibble = odd k
    (`mxfp4_tensor.py::fuse_uint4_to_uint8`).

    GPTQ Marlin expects ``[E, K // 8, N]`` (int32) with eight fp4 values per
    int32 packed K-major (val_k0 in bits 0-3, val_k1 in bits 4-7, ...).

    Both packings agree byte-for-byte, so we only have to (1) reinterpret as
    int32 along the K-stride and (2) transpose the resulting tensor so K
    becomes the outer of the last two dims.
    """
    assert w_uint8.dtype in (torch.int8, torch.uint8), (
        f"expected int8/uint8, got {w_uint8.dtype}"
    )
    assert w_uint8.ndim == 3
    E, N, K_half = w_uint8.shape
    assert K_half % 4 == 0, (
        f"V4 weight K_half={K_half} must be divisible by 4 to view as int32"
    )
    # Make sure the K dim is contiguous in memory before reinterpret.
    w_contig = w_uint8.contiguous()
    # Same byte order, view as int32: [E, N, K_half] -> [E, N, K_half // 4]
    w_i32 = w_contig.view(torch.int32).view(E, N, K_half // 4)
    # GPTQ format wants K-major: [E, K // 8, N].
    return w_i32.transpose(1, 2).contiguous()


def convert_v4_weights_to_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repack V4 raw FP4 weights + ue8m0 scales into Marlin's expected layout.

    Inputs:
      w13       : ``[E, 2 * intermediate_size, hidden_size // 2]`` int8 (fp4 packed).
      w13_scale : ``[E, 2 * intermediate_size, hidden_size // 32]`` Float8_e8m0fnu.
      w2        : ``[E, hidden_size, intermediate_size // 2]`` int8 (fp4 packed).
      w2_scale  : ``[E, hidden_size, intermediate_size // 32]`` Float8_e8m0fnu.

    Outputs (matching what `moe_wna16_marlin_gemm` consumes):
      w13_marlin       : repacked weights for w13 (int32 layout).
      w13_scale_marlin : permuted ue8m0 scales [E, K // group_size, N].
      w2_marlin        : repacked weights for w2.
      w2_scale_marlin  : permuted ue8m0 scales for w2.
    """
    E = w13.shape[0]

    K_w13 = hidden_size
    N_w13 = 2 * intermediate_size
    K_w2 = intermediate_size
    N_w2 = hidden_size

    # --- weights ---
    w13_gptq = _v4_pack_to_gptq_int32(w13)  # [E, K_w13 // 8, N_w13]
    w2_gptq = _v4_pack_to_gptq_int32(w2)    # [E, K_w2 // 8, N_w2]

    # gptq_marlin_moe_repack expects a per-expert `perm` (act-order indices).
    # We have no act-order; pass an empty int per expert which the repack
    # kernel treats as the identity permutation (same convention used by
    # compressed_tensors_wNa16_moe.py:286-301 / awq.py).
    perm_w13 = torch.empty((E, 0), dtype=torch.int32, device=w13.device)
    perm_w2 = torch.empty((E, 0), dtype=torch.int32, device=w13.device)
    w13_marlin = gptq_marlin_moe_repack(w13_gptq, perm_w13, K_w13, N_w13, V4_NUM_BITS)
    w2_marlin = gptq_marlin_moe_repack(w2_gptq, perm_w2, K_w2, N_w2, V4_NUM_BITS)

    # --- scales ---
    # V4 stores scale as [E, N, K // group_size]. Marlin wants K-major for
    # the per-expert reshape inside marlin_moe_permute_scales: pass it
    # transposed so the resulting view is [E, K // group_size, N].
    w13_scale_t = w13_scale.transpose(1, 2).contiguous()  # [E, K_w13/32, N_w13]
    w2_scale_t = w2_scale.transpose(1, 2).contiguous()    # [E, K_w2/32, N_w2]
    # marlin_moe_permute_scales operates on float dtype; ue8m0 is just a uint8
    # bit-pattern and the permutation is layout-only, so reinterpret as uint8
    # for the gather, then re-view as Float8_e8m0fnu on the output. The kernel
    # consumes it as Float8_e8m0fnu directly (ops.cu:1129-1130).
    w13_scale_u8 = w13_scale_t.view(torch.uint8)
    w2_scale_u8 = w2_scale_t.view(torch.uint8)
    w13_scale_marlin = marlin_moe_permute_scales(
        w13_scale_u8, K_w13, N_w13, V4_FP4_GROUP_SIZE
    ).view(torch.float8_e8m0fnu)
    w2_scale_marlin = marlin_moe_permute_scales(
        w2_scale_u8, K_w2, N_w2, V4_FP4_GROUP_SIZE
    ).view(torch.float8_e8m0fnu)

    return w13_marlin, w13_scale_marlin, w2_marlin, w2_scale_marlin


def apply_v4_marlin_moe(
    *,
    hidden_states: torch.Tensor,    # [M, K] bf16
    w13: torch.Tensor,              # marlin-packed weights for w13
    w2: torch.Tensor,               # marlin-packed weights for w2
    w13_scale: torch.Tensor,        # ue8m0 scales for w13 (Marlin layout)
    w2_scale: torch.Tensor,         # ue8m0 scales for w2 (Marlin layout)
    topk_weights: torch.Tensor,     # [M, top_k] bf16/fp16
    topk_ids: torch.Tensor,         # [M, top_k] int32
    intermediate_size: int,         # per-partition N of w13 (== 2*intermediate)/2 = intermediate
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Run V4 sparse MoE through sgl-kernel's `moe_wna16_marlin_gemm`.

    Mirrors the structure of `fused_marlin_moe()` but bypasses its
    `hidden_states.dtype == w_scale.dtype` assert, since for FP4 e2m1 +
    group_size=32 the kernel takes ue8m0 scales directly.
    """
    from sglang.srt.layers.moe.fused_moe_triton import (
        moe_align_block_size,
        try_get_optimal_moe_config,
    )

    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert w13.is_contiguous() and w2.is_contiguous()
    assert hidden_states.is_contiguous()

    M, K = hidden_states.shape
    E = w13.shape[0]
    N = intermediate_size
    topk = topk_ids.shape[1]

    # Sanitize any -1 expert ids (kt hybrid path may emit these); for the GPU
    # path we just zero them out, the Marlin kernel ignores zero topk weights.
    invalid = (topk_ids < 0) | (topk_ids >= E)
    topk_ids = torch.where(invalid, torch.zeros_like(topk_ids), topk_ids)
    topk_weights = torch.where(
        invalid, torch.zeros_like(topk_weights), topk_weights
    )

    if E == 0 or M == 0:
        return torch.zeros_like(hidden_states)

    if _BYPASS:
        return torch.zeros_like(hidden_states)

    _diag("input_hs", hidden_states)
    _diag("input_w13_scale", w13_scale)
    _diag("input_w2_scale", w2_scale)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w13.shape,
        w2.shape,
        topk,
        None,
        is_marlin=True,
    )
    config = get_config_func(M)
    block_size_m = config["BLOCK_SIZE_M"]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, E
    )

    sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    max_workspace_size = (max(2 * N, K) // 64) * (
        sorted_token_ids.size(0) // block_size_m
    )
    max_workspace_size = min(max_workspace_size, sms * 4)
    workspace = torch.zeros(
        max_workspace_size, dtype=torch.int, device=hidden_states.device
    )

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    )
    fp4_type = scalar_types.float4_e2m1f

    intermediate1 = moe_wna16_marlin_gemm(
        hidden_states,
        None,           # c_or_none -> kernel allocates [M*topk, 2*N]
        w13,
        None,           # b_bias_or_none
        w13_scale,
        None,           # global_scale_or_none
        None,           # b_zeros_or_none
        None,           # g_idx_or_none
        None,           # perm_or_none
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_k_full=True,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    _diag("gemm1_out", intermediate1)
    intermediate2 = torch.empty(
        (M * topk, N), device=hidden_states.device, dtype=hidden_states.dtype
    )
    silu_and_mul(intermediate1.view(-1, 2 * N), intermediate2)
    _diag("silu_out", intermediate2)

    intermediate3 = moe_wna16_marlin_gemm(
        intermediate2,
        None,
        w2,
        None,
        w2_scale,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=False,
        b_q_type=fp4_type,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=True,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    _diag("gemm2_out", intermediate3)
    output = torch.empty_like(hidden_states)
    moe_sum_reduce(intermediate3, output, routed_scaling_factor)
    _diag("moe_final", output)
    return output
