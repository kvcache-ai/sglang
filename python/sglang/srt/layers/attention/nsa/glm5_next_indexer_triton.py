"""Architecture-local Triton scorer for GLM-5-Next KPool.

The scorer keeps the model's exact contract: one ReLU dot product per index
head, followed by the learned head reduction and the optional per-key FP8
descale.  SM86 consumes BF16 Q/K directly; SM89 consumes E4M3 Q/K.  SM120
continues to use its accepted eager implementation.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
INDEX_PAGE_SIZE = 64


def use_glm5_next_triton_indexer(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) in ((8, 6), (8, 9))


def glm5_next_indexer_launch_config(
    capability: tuple[int, int],
) -> tuple[int, int, int]:
    """Return the offline-selected conservative config for SM86/SM89."""

    if capability == (8, 6):
        return 32, 4, 2
    if capability == (8, 9):
        return 64, 4, 3
    raise ValueError(
        f"GLM Triton indexer is unsupported on SM{capability[0]}{capability[1]}"
    )


def _validate_glm5_next_indexer_profile(
    query: torch.Tensor,
    index_k: torch.Tensor,
    k_scale: torch.Tensor | None,
) -> tuple[int, int]:
    """Fail closed if cache precision does not match the consumer GPU profile."""

    if not use_glm5_next_triton_indexer(query.device):
        raise RuntimeError("GLM Triton indexer requires SM86 or SM89")
    capability = torch.cuda.get_device_capability(query.device)
    if index_k.device != query.device or (
        k_scale is not None and k_scale.device != query.device
    ):
        raise ValueError("GLM Triton indexer inputs must share one CUDA device")
    if capability == (8, 6):
        if query.dtype != torch.bfloat16 or index_k.dtype != torch.bfloat16:
            raise TypeError("SM86 GLM indexer requires BF16 query and KPool cache")
        if k_scale is not None:
            raise TypeError("SM86 GLM indexer must not receive an FP8 K scale")
    else:
        if query.dtype != torch.float8_e4m3fn or index_k.dtype != torch.float8_e4m3fn:
            raise TypeError("SM89 GLM indexer requires E4M3 query and KPool cache")
        if k_scale is None:
            raise TypeError("SM89 GLM indexer requires the FP8 KPool scale")
        if k_scale.dtype != torch.float32:
            raise TypeError("SM89 GLM indexer requires an FP32 KPool scale")
    return capability


@triton.jit
def _glm5_next_flat_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    logits_ptr,
    num_keys: tl.int32,
    stride_q_row: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_row: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_weights_row: tl.constexpr,
    stride_weights_head: tl.constexpr,
    stride_logits_row: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    USE_K_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    key_block = tl.program_id(1)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEADS)

    keys = tl.load(
        k_ptr
        + key_offsets[:, None] * stride_k_row
        + dim_offsets[None, :] * stride_k_dim,
        mask=key_offsets[:, None] < num_keys,
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
    if USE_K_SCALE:
        key_scale = tl.load(
            k_scale_ptr + key_offsets,
            mask=key_offsets < num_keys,
            other=0.0,
        ).to(tl.float32)
        logits *= key_scale

    row_start = tl.load(ks_ptr + row)
    row_end = tl.load(ke_ptr + row)
    valid = (
        (key_offsets >= row_start) & (key_offsets < row_end) & (key_offsets < num_keys)
    )
    tl.store(
        logits_ptr + row * stride_logits_row + key_offsets,
        tl.where(valid, logits, -float("inf")),
        mask=key_offsets < num_keys,
    )


@triton.jit
def _glm5_next_paged_mqa_logits_kernel(
    q_ptr,
    cache_ptr,
    cache_scale_ptr,
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
    SCALE_OFFSET: tl.constexpr,
    SCALE_PAGE_STRIDE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    USE_K_SCALE: tl.constexpr,
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
        other=0,
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
    if USE_K_SCALE:
        scales = tl.load(
            cache_scale_ptr
            + safe_pages * SCALE_PAGE_STRIDE
            + SCALE_OFFSET
            + token_offsets,
            mask=page_valid & (key_offsets < max_seq_len),
            other=0.0,
        ).to(tl.float32)
        logits *= scales

    seq_len = tl.load(seq_lens_ptr + row)
    valid = page_valid & (key_offsets < seq_len) & (key_offsets < max_seq_len)
    tl.store(
        logits_ptr + row * stride_logits_row + key_offsets,
        tl.where(valid, logits, -float("inf")),
        mask=key_offsets < max_seq_len,
    )


def iter_glm5_next_triton_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    k_scale: torch.Tensor | None = None,
    query_chunk_size: int = 32,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    capability = _validate_glm5_next_indexer_profile(q, k, k_scale)
    if q.ndim != 3 or q.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"q must have shape [Q,H,128], got {tuple(q.shape)}")
    if k.ndim != 2 or k.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"k must have shape [K,128], got {tuple(k.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(f"weights must have shape {tuple(q.shape[:2])}")
    if weights.dtype != torch.float32:
        raise TypeError("GLM Triton indexer weights must use FP32")
    if k_scale is not None and k_scale.shape != (k.shape[0],):
        raise ValueError("k_scale must have shape [K]")
    if ks.dtype != torch.int32 or ke.dtype != torch.int32:
        raise TypeError("GLM Triton ragged bounds must use int32")
    if ks.shape != (q.shape[0],) or ke.shape != (q.shape[0],):
        raise ValueError("GLM Triton ragged bounds must have one entry per query")
    if weights.device != q.device or ks.device != q.device or ke.device != q.device:
        raise ValueError("GLM Triton indexer metadata must share the query device")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    block_k, num_warps, num_stages = glm5_next_indexer_launch_config(capability)
    dummy_scale = k if k_scale is None else k_scale
    for q_start in range(0, q.shape[0], query_chunk_size):
        q_end = min(q_start + query_chunk_size, q.shape[0])
        q_chunk = q[q_start:q_end].contiguous()
        weights_chunk = weights[q_start:q_end].contiguous()
        logits = torch.empty(
            (q_end - q_start, k.shape[0]), dtype=torch.float32, device=q.device
        )
        grid = (q_end - q_start, triton.cdiv(k.shape[0], block_k))
        _glm5_next_flat_mqa_logits_kernel[grid](
            q_chunk,
            k,
            dummy_scale,
            weights_chunk,
            ks[q_start:q_end],
            ke[q_start:q_end],
            logits,
            k.shape[0],
            q_chunk.stride(0),
            q_chunk.stride(1),
            q_chunk.stride(2),
            k.stride(0),
            k.stride(1),
            weights_chunk.stride(0),
            weights_chunk.stride(1),
            logits.stride(0),
            NUM_HEADS=q.shape[1],
            HEAD_DIM=INDEX_HEAD_DIM,
            USE_K_SCALE=k_scale is not None,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        yield q_start, q_end, logits


def glm5_next_triton_paged_mqa_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    *,
    use_k_scale: bool,
) -> torch.Tensor:
    if q.ndim == 4:
        if q.shape[1] != 1:
            raise ValueError("paged q next-token dimension must be one")
        q = q[:, 0]
    q = q.contiguous()
    weights = weights.contiguous()
    if q.ndim != 3 or q.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"q must have shape [B,H,128], got {tuple(q.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(f"weights must have shape {tuple(q.shape[:2])}")
    if weights.dtype != torch.float32:
        raise TypeError("GLM Triton indexer weights must use FP32")
    if cache.ndim != 2 or page_table.ndim != 2:
        raise ValueError("cache and page_table must be two-dimensional")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if max_seq_len > page_table.shape[1] * INDEX_PAGE_SIZE:
        raise ValueError("max_seq_len exceeds the KPool page-table capacity")
    if cache.shape[0] == 0 or cache.stride(1) != 1:
        raise ValueError("KPool cache must contain contiguous physical pages")
    if seq_lens.dtype != torch.int32 or page_table.dtype != torch.int32:
        raise TypeError("KPool lengths and page table must use int32")
    if seq_lens.shape not in ((q.shape[0],), (q.shape[0], 1)):
        raise ValueError("KPool lengths must have one entry per query")
    seq_lens = seq_lens.view(-1)
    if page_table.shape[0] != q.shape[0]:
        raise ValueError("KPool page table must have one row per query")
    if weights.device != q.device or seq_lens.device != q.device:
        raise ValueError("KPool scorer metadata must share the query device")
    if page_table.device != q.device or cache.device != q.device:
        raise ValueError("KPool cache metadata must share the query device")

    if use_k_scale:
        if cache.dtype != torch.uint8:
            raise TypeError("scaled KPool cache must use packed uint8 storage")
        cache_values = cache.view(torch.float8_e4m3fn)
        cache_scales = cache.view(torch.float32)
        scale_offset = INDEX_PAGE_SIZE * INDEX_HEAD_DIM // 4
        scale_page_stride = cache.shape[1] // 4
    else:
        if cache.dtype != torch.bfloat16:
            raise TypeError("unscaled KPool cache must be BF16")
        cache_values = cache
        cache_scales = cache
        scale_offset = 0
        scale_page_stride = 0

    capability = _validate_glm5_next_indexer_profile(
        q,
        cache_values,
        cache_scales if use_k_scale else None,
    )
    block_k, num_warps, num_stages = glm5_next_indexer_launch_config(capability)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    grid = (q.shape[0], triton.cdiv(max_seq_len, block_k))
    _glm5_next_paged_mqa_logits_kernel[grid](
        q,
        cache_values,
        cache_scales,
        weights,
        seq_lens,
        page_table,
        logits,
        max_seq_len,
        cache.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cache_values.stride(0),
        weights.stride(0),
        weights.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        SCALE_OFFSET=scale_offset,
        SCALE_PAGE_STRIDE=scale_page_stride,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        PAGE_SIZE=INDEX_PAGE_SIZE,
        USE_K_SCALE=use_k_scale,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits
