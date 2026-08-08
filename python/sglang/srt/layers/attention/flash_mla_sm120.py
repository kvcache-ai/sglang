"""FlashInfer sparse MLA decode adapter for DeepSeek V4 on SM120.

DeepSeek V4 stores each FP8 KV page as one data section followed by one
UE8M0-scale section.  FlashInfer's SM120 kernel requires 64-token pages, while
SGLang may use a larger physical SWA page.  The split buffer below is persistent
and only the source pages referenced by the current decode step are refreshed.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch
import triton
import triton.language as tl

_PBS_DST = 64
_NOPE_ROPE_STRIDE = 576
_SCALE_STRIDE = 8
_BYTES_PER_TOKEN = _NOPE_ROPE_STRIDE + _SCALE_STRIDE
_BYTES_PER_DST_PAGE = _PBS_DST * _BYTES_PER_TOKEN
_BYTES_PER_DST_PAGE_PADDED = (
    math.ceil(_BYTES_PER_DST_PAGE / _NOPE_ROPE_STRIDE) * _NOPE_ROPE_STRIDE
)

# The base branch predates sglang.srt.runtime_context.  Keep the same grow-only,
# per-device lifetime here.  One model process executes its layers on the same
# CUDA stream, so reusing the page-split workspace between layers is ordered.
_SPLIT_BUFFERS: dict[tuple[str, Optional[int]], torch.Tensor] = {}
_MASK_BUFFERS: dict[tuple[str, Optional[int]], torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple[str, Optional[int]]:
    return (device.type, device.index)


@functools.lru_cache(maxsize=1)
def is_flashinfer_dsv4_available() -> bool:
    """Return whether the pinned FlashInfer exposes its SM120 DSV4 API."""
    try:
        from flashinfer.mla._sparse_mla_sm120 import (  # noqa: F401
            _DECODE_MAX_TOKENS,
            _sparse_mla_sm120_paged_attention,
        )
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False
    return True


@triton.jit
def _page_mark_kernel(
    indices_ptr,
    mask_ptr,
    n_indices,
    n_pages,
    src_page_size: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_indices:
        return
    index = tl.load(indices_ptr + pid)
    if index < 0:
        return
    page = index // src_page_size
    if page < n_pages:
        # Concurrent stores of the same byte value are benign.
        tl.store(mask_ptr + page, 1)


@triton.jit
def _page_split_kernel(
    src_ptr,
    dst_ptr,
    n_pages,
    src_stride0: tl.constexpr,
    dst_stride0: tl.constexpr,
    data_per_subpage: tl.constexpr,
    scale_per_subpage: tl.constexpr,
    src_scale_offset: tl.constexpr,
    dst_scale_offset: tl.constexpr,
    ratio: tl.constexpr,
    block_size: tl.constexpr,
    page_mask_ptr,
    has_page_mask: tl.constexpr,
):
    pid = tl.program_id(0)
    page = pid // ratio
    subpage = pid % ratio
    if page >= n_pages:
        return
    if has_page_mask:
        if tl.load(page_mask_ptr + page) == 0:
            return

    src_base = src_ptr + page * src_stride0
    dst_base = dst_ptr + (page * ratio + subpage) * dst_stride0

    data_src_offset = subpage * data_per_subpage
    for start in tl.range(0, data_per_subpage, block_size):
        offsets = start + tl.arange(0, block_size)
        valid = offsets < data_per_subpage
        values = tl.load(src_base + data_src_offset + offsets, mask=valid)
        tl.store(dst_base + offsets, values, mask=valid)

    scale_src_offset = src_scale_offset + subpage * scale_per_subpage
    for start in tl.range(0, scale_per_subpage, block_size):
        offsets = start + tl.arange(0, block_size)
        valid = offsets < scale_per_subpage
        values = tl.load(src_base + scale_src_offset + offsets, mask=valid)
        tl.store(dst_base + dst_scale_offset + offsets, values, mask=valid)


def _split_kv_pages_to_64(
    kv_u8: torch.Tensor,
    src_page_size: int,
    touched_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a persistent 64-token-page view of a footer-layout KV pool."""
    assert src_page_size >= _PBS_DST and src_page_size % _PBS_DST == 0
    if src_page_size == _PBS_DST:
        return kv_u8

    num_src_pages = kv_u8.shape[0]
    ratio = src_page_size // _PBS_DST
    num_dst_pages = num_src_pages * ratio
    key = _device_key(kv_u8.device)

    split_buffer = _SPLIT_BUFFERS.get(key)
    if split_buffer is None or split_buffer.shape[0] < num_dst_pages:
        with torch.inference_mode(False):
            split_buffer = torch.empty(
                num_dst_pages,
                _BYTES_PER_DST_PAGE_PADDED,
                dtype=torch.uint8,
                device=kv_u8.device,
            )
        _SPLIT_BUFFERS[key] = split_buffer
    dst = split_buffer[:num_dst_pages]

    if kv_u8.ndim == 4:
        src_stride0 = kv_u8.stride(0)
        src = torch.as_strided(kv_u8, (num_src_pages, src_stride0), (src_stride0, 1))
    else:
        src = kv_u8
        src_stride0 = src.stride(0)

    use_mask = touched_indices is not None and touched_indices.numel() > 0
    page_mask_ptr = src  # Dummy pointer for the unmasked specialization.
    if use_mask:
        mask_buffer = _MASK_BUFFERS.get(key)
        if mask_buffer is None or mask_buffer.shape[0] < num_src_pages:
            # A tensor allocated while inference mode is active cannot later be
            # zeroed during CUDA-graph capture.  Force a normal tensor here.
            with torch.inference_mode(False):
                mask_buffer = torch.empty(
                    num_src_pages, dtype=torch.int8, device=kv_u8.device
                )
            _MASK_BUFFERS[key] = mask_buffer
        page_mask = mask_buffer[:num_src_pages]
        page_mask.zero_()
        flat_indices = touched_indices.reshape(-1).contiguous()
        if flat_indices.dtype != torch.int32:
            flat_indices = flat_indices.to(torch.int32)
        _page_mark_kernel[(flat_indices.numel(),)](
            flat_indices,
            page_mask,
            flat_indices.numel(),
            num_src_pages,
            src_page_size,
        )
        page_mask_ptr = page_mask

    data_per_subpage = _PBS_DST * _NOPE_ROPE_STRIDE
    scale_per_subpage = _PBS_DST * _SCALE_STRIDE
    _page_split_kernel[(num_dst_pages,)](
        src,
        dst,
        num_src_pages,
        src_stride0,
        _BYTES_PER_DST_PAGE_PADDED,
        data_per_subpage,
        scale_per_subpage,
        src_page_size * _NOPE_ROPE_STRIDE,
        data_per_subpage,
        ratio,
        1024,
        page_mask_ptr,
        use_mask,
    )

    return dst.as_strided(
        (num_dst_pages, _PBS_DST, 1, _BYTES_PER_TOKEN),
        (_BYTES_PER_DST_PAGE_PADDED, _BYTES_PER_TOKEN, _BYTES_PER_TOKEN, 1),
    )


def flash_mla_with_kvcache_sm120(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    is_fp8_kvcache: bool,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_unused,
):
    """Run FlashInfer's SM120 sparse MLA kernel for one DSV4 decode step."""
    if not is_flashinfer_dsv4_available():
        raise ImportError("FlashInfer SM120 DSV4 sparse MLA API is unavailable")
    if not is_fp8_kvcache:
        raise ValueError("FlashInfer SM120 DSV4 decode requires an FP8 KV cache")
    if head_dim_v != 512:
        raise ValueError(f"DeepSeek V4 head_dim_v must be 512, got {head_dim_v}")

    from flashinfer.mla._sparse_mla_sm120 import (
        _DECODE_MAX_TOKENS,
        _sparse_mla_sm120_paged_attention,
    )

    q_3d = q.squeeze(1) if q.ndim == 4 else q
    batch_size, num_heads, _ = q_3d.shape
    idx = indices.squeeze(1) if indices.ndim == 3 else indices

    kv_u8 = k_cache if k_cache.dtype == torch.uint8 else k_cache.view(torch.uint8)
    src_page_size = k_cache.shape[1]
    kv_64 = _split_kv_pages_to_64(kv_u8, src_page_size, idx)

    extra_kv = extra_k_cache
    if extra_kv is not None and extra_kv.dtype != torch.uint8:
        extra_kv = extra_kv.view(torch.uint8)
    extra_idx = extra_indices_in_kvcache
    if extra_idx is not None and extra_idx.ndim == 3:
        extra_idx = extra_idx.squeeze(1)

    output = torch.empty(
        batch_size, num_heads, head_dim_v, dtype=torch.bfloat16, device=q.device
    )
    output_lse = torch.empty(
        batch_size, num_heads, dtype=torch.float32, device=q.device
    )

    if batch_size <= _DECODE_MAX_TOKENS:
        split_size = 64
        num_splits = math.ceil(idx.shape[-1] / split_size)
        if extra_idx is not None:
            num_splits += math.ceil(extra_idx.shape[-1] / split_size)
        mid_output = torch.empty(
            batch_size,
            num_heads,
            num_splits,
            head_dim_v,
            dtype=torch.bfloat16,
            device=q.device,
        )
        mid_lse = torch.empty(
            batch_size,
            num_heads,
            num_splits,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        mid_output = None
        mid_lse = None

    _sparse_mla_sm120_paged_attention(
        q_3d,
        kv_64,
        idx,
        output,
        output_lse,
        float(softmax_scale),
        d_v=head_dim_v,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_kv_cache=extra_kv,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
        mid_out=mid_output,
        mid_lse=mid_lse,
    )
    return (output.unsqueeze(1), None)
