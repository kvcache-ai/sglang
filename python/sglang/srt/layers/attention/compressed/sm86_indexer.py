"""Page-parallel BF16 indexer for DeepSeek V4 decode on SM86."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

DSV4_SM86_INDEXER_BLOCK_K = 32
DSV4_SM86_INDEXER_NUM_WARPS = 4
DSV4_SM86_INDEXER_NUM_STAGES = 2


@triton.jit
def _bf16_direct_paged_mqa_logits_triton_kernel(
    q_ptr,
    cache_ptr,
    weights_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    max_seq_len: tl.int32,
    num_cache_pages: tl.int32,
    stride_q_row: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_cache_page: tl.constexpr,
    stride_weights_row: tl.constexpr,
    stride_weights_head: tl.constexpr,
    stride_page_table_row: tl.constexpr,
    stride_page_table_col: tl.constexpr,
    stride_logits_row: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEADS)

    page_cols = key_offsets // PAGE_SIZE
    token_offsets = key_offsets % PAGE_SIZE
    pages = tl.load(
        page_table_ptr
        + row * stride_page_table_row
        + page_cols * stride_page_table_col,
        mask=key_offsets < max_seq_len,
        other=-1,
    ).to(tl.int64)
    page_valid = (pages >= 0) & (pages < num_cache_pages)
    safe_pages = tl.minimum(tl.maximum(pages, 0), num_cache_pages - 1)

    keys = tl.load(
        cache_ptr
        + safe_pages[:, None] * stride_cache_page
        + token_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :],
        mask=page_valid[:, None] & (key_offsets[:, None] < max_seq_len),
        other=0.0,
    )
    queries = tl.load(
        q_ptr
        + row * stride_q_row
        + head_offsets[:, None] * stride_q_head
        + dim_offsets[None, :] * stride_q_dim
    )
    head_scores = tl.dot(keys, tl.trans(queries), out_dtype=tl.float32)
    head_scores = tl.maximum(head_scores, 0.0)
    weights = tl.load(
        weights_ptr + row * stride_weights_row + head_offsets * stride_weights_head
    ).to(tl.float32)
    logits = tl.sum(head_scores * weights[None, :], axis=1)

    seq_len = tl.load(seq_lens_ptr + row)
    valid = page_valid & (key_offsets < seq_len) & (key_offsets < max_seq_len)
    tl.store(
        logits_ptr + row * stride_logits_row + key_offsets,
        tl.where(valid, logits, -float("inf")),
        mask=key_offsets < max_seq_len,
    )


def bf16_direct_paged_mqa_logits_triton(
    q_bf16: torch.Tensor,
    kvcache_bf16: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Compute DSV4 index logits with one CTA per 32-key block."""
    _ = deep_gemm_metadata
    if q_bf16.ndim != 4 or q_bf16.shape[1] != 1:
        raise ValueError(f"q must have shape [B,1,H,D], got {tuple(q_bf16.shape)}")
    batch_size, _, num_heads, head_dim = q_bf16.shape
    if kvcache_bf16.ndim != 3 or kvcache_bf16.shape[1:] != (64, 128):
        raise ValueError(
            "SM86 BF16 index cache must have shape [pages,64,128], got "
            f"{tuple(kvcache_bf16.shape)}"
        )
    if q_bf16.dtype != torch.bfloat16 or kvcache_bf16.dtype != torch.bfloat16:
        raise TypeError("SM86 direct indexer requires BF16 query and cache")
    if head_dim != 128 or num_heads != 64:
        raise ValueError(
            f"DSV4 indexer expects H=64,D=128, got H={num_heads},D={head_dim}"
        )
    if weight.shape != (batch_size, num_heads) or weight.dtype != torch.float32:
        raise TypeError("DSV4 indexer weights must have shape [B,64] and FP32 dtype")
    if seq_lens.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("seq_lens must contain one value per query")
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError("page_table must have shape [B,max_pages]")
    if max_seq_len <= 0 or max_seq_len > page_table.shape[1] * 64:
        raise ValueError("max_seq_len exceeds the index page-table capacity")
    if clean_logits is not False:
        raise ValueError("DSV4 indexer requires clean_logits=False")

    q = q_bf16[:, 0].contiguous()
    cache = kvcache_bf16.contiguous()
    weights = weight.contiguous()
    lengths = seq_lens.reshape(-1).to(torch.int32)
    pages = page_table.to(torch.int32)
    logits = torch.empty(
        (batch_size, max_seq_len), dtype=torch.float32, device=q.device
    )
    grid = (batch_size, triton.cdiv(max_seq_len, DSV4_SM86_INDEXER_BLOCK_K))
    _bf16_direct_paged_mqa_logits_triton_kernel[grid](
        q,
        cache,
        weights,
        lengths,
        pages,
        logits,
        max_seq_len,
        cache.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cache.stride(0),
        weights.stride(0),
        weights.stride(1),
        pages.stride(0),
        pages.stride(1),
        logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=64,
        BLOCK_K=DSV4_SM86_INDEXER_BLOCK_K,
        num_warps=DSV4_SM86_INDEXER_NUM_WARPS,
        num_stages=DSV4_SM86_INDEXER_NUM_STAGES,
    )
    return logits
