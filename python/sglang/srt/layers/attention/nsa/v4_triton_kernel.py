# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from vllm-project/vllm PR #40929 (head: bbbearxyz/vllm@10934adf,
# vllm/model_executor/layers/deepseek_v4_triton_kernels.py).
# Provides a portable Triton implementation of DeepSeek V4 sparse FP8
# MLA decode for hardware where the standalone flash_mla CUDA kernel is
# unavailable (notably SM_120 / RTX 5090).
"""Triton fallback for DeepSeek V4 sparse FP8 MLA decode."""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

DEEPSEEK_V4_MLA_HEAD_DIM = 512
FP8_DS_MLA_FP8_DIM = 448
FP8_DS_MLA_SCALE_GROUP = 64
FP8_DS_MLA_SCALE_BYTES = 8
FP8_DS_MLA_TOKEN_BYTES = 576


@triton.jit
def _decode_sparse_attention_fp8_kernel(
    q_ptr,
    swa_cache_fp8_ptr,
    swa_cache_bf16_ptr,
    swa_cache_u8_ptr,
    swa_indices_ptr,
    swa_lens_ptr,
    extra_cache_fp8_ptr,
    extra_cache_bf16_ptr,
    extra_cache_u8_ptr,
    extra_indices_ptr,
    extra_lens_ptr,
    sink_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    swa_index_topk: tl.constexpr,
    extra_index_topk: tl.constexpr,
    swa_num_blocks: tl.constexpr,
    extra_num_blocks: tl.constexpr,
    swa_block_size: tl.constexpr,
    extra_block_size: tl.constexpr,
    swa_stride_block_bytes: tl.constexpr,
    extra_stride_block_bytes: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_swa_indices_t: tl.constexpr,
    stride_swa_indices_k: tl.constexpr,
    stride_extra_indices_t: tl.constexpr,
    stride_extra_indices_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
    stride_out_d: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_DIM: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    TOKEN_BYTES: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
    LOG2E_CONST: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < num_heads

    q = tl.load(
        q_ptr
        + token_id * stride_qt
        + heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_h[:, None],
        other=0.0,
    )

    if HAS_SINK:
        sink = tl.load(sink_ptr + heads, mask=mask_h, other=-float("inf"))
        e_max = sink * LOG2E_CONST
        e_sum = tl.where(mask_h, 1.0, 0.0)
    else:
        e_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
        e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + token_id)
    extra_len = tl.load(extra_lens_ptr + token_id) if HAS_EXTRA else 0
    total_len = extra_len + swa_len

    for start in range(0, extra_index_topk + swa_index_topk, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        use_extra = HAS_EXTRA & (offs_n < extra_len)
        use_swa = (offs_n >= extra_len) & (offs_n < total_len)

        extra_cols = offs_n
        swa_cols = offs_n - extra_len
        extra_idx = tl.load(
            extra_indices_ptr
            + token_id * stride_extra_indices_t
            + extra_cols * stride_extra_indices_k,
            mask=HAS_EXTRA & (extra_cols < extra_index_topk),
            other=-1,
        )
        swa_idx = tl.load(
            swa_indices_ptr
            + token_id * stride_swa_indices_t
            + swa_cols * stride_swa_indices_k,
            mask=(swa_cols >= 0) & (swa_cols < swa_index_topk),
            other=-1,
        )
        idx = tl.where(use_extra, extra_idx, swa_idx)

        extra_block = idx // extra_block_size
        extra_pos = idx - extra_block * extra_block_size
        swa_block = idx // swa_block_size
        swa_pos = idx - swa_block * swa_block_size
        valid_extra = use_extra & (idx >= 0) & (extra_block < extra_num_blocks)
        valid_swa = use_swa & (idx >= 0) & (swa_block < swa_num_blocks)
        valid = valid_extra | valid_swa

        extra_token_base = extra_block * extra_stride_block_bytes
        extra_token_base += extra_pos * TOKEN_BYTES
        swa_token_base = swa_block * swa_stride_block_bytes
        swa_token_base += swa_pos * TOKEN_BYTES
        token_base = tl.where(use_extra, extra_token_base, swa_token_base)
        block_size = tl.where(use_extra, extra_block_size, swa_block_size)
        stride_block_bytes = tl.where(
            use_extra, extra_stride_block_bytes, swa_stride_block_bytes
        )
        pos = tl.where(use_extra, extra_pos, swa_pos)

        is_fp8 = offs_d < FP8_DIM
        scale_offsets = (
            tl.where(use_extra, extra_block, swa_block)[:, None]
            * stride_block_bytes[:, None]
            + block_size[:, None] * TOKEN_BYTES
            + pos[:, None] * SCALE_BYTES
            + (offs_d[None, :] // SCALE_GROUP)
        )
        encoded_scale = tl.load(
            tl.where(use_extra[:, None], extra_cache_u8_ptr, swa_cache_u8_ptr)
            + scale_offsets,
            mask=valid[:, None] & is_fp8[None, :],
            other=127,
        ).to(tl.float32)
        fp8_scale = tl.exp2(encoded_scale - 127.0)

        fp8_offsets = token_base[:, None] + offs_d[None, :]
        fp8_vals = (
            tl.load(
                tl.where(use_extra[:, None], extra_cache_fp8_ptr, swa_cache_fp8_ptr)
                + fp8_offsets,
                mask=valid[:, None] & is_fp8[None, :],
                other=0.0,
            ).to(tl.float32)
            * fp8_scale
        )

        bf16_offsets = (token_base[:, None] + FP8_DIM) // 2
        bf16_offsets += offs_d[None, :] - FP8_DIM
        bf16_vals = tl.load(
            tl.where(use_extra[:, None], extra_cache_bf16_ptr, swa_cache_bf16_ptr)
            + bf16_offsets,
            mask=valid[:, None] & (~is_fp8[None, :]),
            other=0.0,
        ).to(tl.float32)
        k = tl.where(is_fp8[None, :], fp8_vals, bf16_vals)

        qk = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale_log2
        qk = tl.where(
            mask_h[:, None] & valid[None, :],
            qk,
            -3.4028234663852886e38,
        )

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        p = tl.where(mask_h[:, None] & valid[None, :], p, 0.0)
        acc = acc * re_scale[:, None] + tl.dot(p.to(k.dtype), k)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    acc = acc / tl.maximum(e_sum, 1.0e-20)[:, None]
    tl.store(
        out_ptr
        + token_id * stride_out_t
        + heads[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        acc.to(tl.bfloat16),
        mask=mask_h[:, None],
    )


def decode_sparse_attention_triton(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lens: torch.Tensor | None = None,
) -> None:
    """Run V4 sparse FP8 MLA decode through a portable Triton kernel.

    The kernel expects:
      q          : (N_tokens, num_heads, 512) bf16
      swa_cache  : (num_blocks, swa_block_size, head_bytes) uint8 packed
                   layout = [block_size * 576 data] + [block_size * 8 scale]
      swa_indices: (N_tokens, topk_swa) int32 (gets squeezed if 3D)
      swa_lens   : (N_tokens,) int32
      out        : (N_tokens, num_heads, 512) bf16, written in place
    """
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices.squeeze(1)

    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return
    if head_dim != DEEPSEEK_V4_MLA_HEAD_DIM:
        raise ValueError(
            "DeepSeek V4 decode Triton fallback expects "
            f"D={DEEPSEEK_V4_MLA_HEAD_DIM}, got {head_dim}"
        )
    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_lens is not None
    )
    if not has_extra:
        extra_cache = swa_cache
        extra_indices = swa_indices[:, :1]
        extra_lens = swa_lens

    assert extra_cache is not None
    assert extra_indices is not None
    assert extra_lens is not None
    grid = (num_tokens, triton.cdiv(num_heads, 8))
    _decode_sparse_attention_fp8_kernel[grid](
        q,
        swa_cache.view(torch.float8_e4m3fn),
        swa_cache.view(torch.bfloat16),
        swa_cache,
        swa_indices,
        swa_lens,
        extra_cache.view(torch.float8_e4m3fn),
        extra_cache.view(torch.bfloat16),
        extra_cache,
        extra_indices,
        extra_lens,
        attn_sink if attn_sink is not None else q,
        out,
        num_tokens,
        num_heads,
        swa_indices.shape[-1],
        extra_indices.shape[-1] if has_extra else 0,
        swa_cache.shape[0],
        extra_cache.shape[0],
        swa_cache.shape[1],
        extra_cache.shape[1],
        swa_cache.stride(0),
        extra_cache.stride(0),
        scale * LOG2E,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_indices.stride(0),
        swa_indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=8,
        BLOCK_N=16,
        BLOCK_D=DEEPSEEK_V4_MLA_HEAD_DIM,
        FP8_DIM=FP8_DS_MLA_FP8_DIM,
        SCALE_GROUP=FP8_DS_MLA_SCALE_GROUP,
        SCALE_BYTES=FP8_DS_MLA_SCALE_BYTES,
        TOKEN_BYTES=FP8_DS_MLA_TOKEN_BYTES,
        HAS_EXTRA=has_extra,
        HAS_SINK=attn_sink is not None,
        LOG2E_CONST=LOG2E,
        num_warps=8,
    )
