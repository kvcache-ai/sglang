from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


@dataclass
class NopeFp8RopeBf16Pack:
    k_nope_fp8: torch.Tensor
    k_rope_bf16: torch.Tensor
    scale_k_nope_ue8m0: torch.Tensor

    def __post_init__(self):
        assert self.k_nope_fp8.shape[-1] == 448
        assert self.k_rope_bf16.shape[-1] == 64
        assert self.scale_k_nope_ue8m0.shape[-1] == 7

    def slice_pack(self, _slice: Any) -> NopeFp8RopeBf16Pack:
        return NopeFp8RopeBf16Pack(
            k_nope_fp8=self.k_nope_fp8[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice],
            scale_k_nope_ue8m0=self.scale_k_nope_ue8m0[_slice],
        )


# ---------------------------------------------------------------------------
# BF16 cache pack (SM_86): nope + rope both stored as bfloat16, no scale.
# ---------------------------------------------------------------------------

@dataclass
class NopeBf16RopeBf16Pack:
    k_nope_bf16: torch.Tensor
    k_rope_bf16: torch.Tensor

    def __post_init__(self):
        assert self.k_nope_bf16.shape[-1] == 448
        assert self.k_rope_bf16.shape[-1] == 64

    def slice_pack(self, _slice: Any) -> "NopeBf16RopeBf16Pack":
        return NopeBf16RopeBf16Pack(
            k_nope_bf16=self.k_nope_bf16[_slice],
            k_rope_bf16=self.k_rope_bf16[_slice],
        )


class SetBf16KAndS:
    """Scatter all-bf16 KV cache (nope + rope) into the page buffer."""

    @classmethod
    def execute(cls, pool, buf, loc, pack: NopeBf16RopeBf16Pack):
        _set_bf16_k_and_s_triton(buf, loc, pack, pool.page_size)

    @classmethod
    def torch(cls, pool, buf, loc, pack: NopeBf16RopeBf16Pack):
        _set_bf16_k_and_s_torch(buf, loc, pack, pool.page_size)


@triton.jit
def _set_bf16_k_and_s_kernel(
    buf_bf16_ptr,
    loc_ptr,
    nope_ptr,
    rope_ptr,
    nope_stride_0,
    rope_stride_0,
    PAGE_SIZE: tl.constexpr,
    BF16_PER_PAGE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TOKEN_BF16: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(loc_ptr + token_id)
    page = loc // PAGE_SIZE
    off = loc % PAGE_SIZE
    base = page * BF16_PER_PAGE + off * TOKEN_BF16

    # Nope
    nope_offs = tl.arange(0, BLOCK_NOPE)
    nope_mask = nope_offs < NOPE_DIM
    nope_data = tl.load(
        nope_ptr + token_id * nope_stride_0 + nope_offs,
        mask=nope_mask, other=0.0,
    )
    tl.store(buf_bf16_ptr + base + nope_offs, nope_data, mask=nope_mask)

    # Rope
    rope_offs = tl.arange(0, BLOCK_ROPE)
    rope_data = tl.load(rope_ptr + token_id * rope_stride_0 + rope_offs)
    tl.store(buf_bf16_ptr + base + NOPE_DIM + rope_offs, rope_data)


def _set_bf16_k_and_s_triton(buf: torch.Tensor, loc: torch.Tensor,
                               pack: NopeBf16RopeBf16Pack, page_size: int):
    (num_tokens,) = loc.shape
    nope_dim, rope_dim = 448, 64
    token_bf16 = nope_dim + rope_dim  # 512
    buf_bf16 = buf.view(torch.bfloat16)
    bf16_per_page = buf_bf16.shape[1]
    loc = loc.to(torch.int64)

    _set_bf16_k_and_s_kernel[(num_tokens,)](
        buf_bf16, loc,
        pack.k_nope_bf16, pack.k_rope_bf16,
        pack.k_nope_bf16.stride(0), pack.k_rope_bf16.stride(0),
        PAGE_SIZE=page_size, BF16_PER_PAGE=bf16_per_page,
        NOPE_DIM=nope_dim, ROPE_DIM=rope_dim, TOKEN_BF16=token_bf16,
        BLOCK_NOPE=512, BLOCK_ROPE=64,
        num_warps=4,
    )


def _set_bf16_k_and_s_torch(buf: torch.Tensor, loc: torch.Tensor,
                              pack: NopeBf16RopeBf16Pack, page_size: int):
    """Torch scatter fallback."""
    (num_tokens,), nope_dim = loc.shape, 448
    rope_dim, buf_dim = 64, buf.shape[1]
    device = buf.device
    loc = loc.to(torch.int64)
    token_bf16 = nope_dim + rope_dim

    loc_page = loc // page_size
    loc_off = loc % page_size

    flat_bf16 = buf.view(torch.bfloat16).flatten()
    bf16_per_page = buf_dim // 2

    nope_base = loc_page * bf16_per_page + loc_off * token_bf16
    nope_idx = nope_base[:, None] + torch.arange(nope_dim, device=device, dtype=torch.int64)[None, :]
    nope_idx = nope_idx.reshape(-1)[: num_tokens * nope_dim]
    flat_bf16[nope_idx] = pack.k_nope_bf16.reshape(-1)

    rope_base = loc_page * bf16_per_page + loc_off * token_bf16 + nope_dim
    rope_idx = rope_base[:, None] + torch.arange(rope_dim, device=device, dtype=torch.int64)[None, :]
    rope_idx = rope_idx.reshape(-1)[: num_tokens * rope_dim]
    flat_bf16[rope_idx] = pack.k_rope_bf16.reshape(-1)


class SetKAndS:
    @classmethod
    def execute(cls, pool, buf, loc, nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack):
        # Triton's fp8e4nv type is only supported on SM >= 90 (Hopper+).
        # SM_86 (Ampere) and older fall back to the Torch implementation.
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            cls.torch(pool, buf, loc, nope_fp8_rope_bf16_pack)
        else:
            cls.triton(pool, buf, loc, nope_fp8_rope_bf16_pack)

    @classmethod
    def torch(cls, pool, buf, loc, nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack):
        _set_k_and_s_torch(buf, loc, nope_fp8_rope_bf16_pack, pool.page_size)

    @classmethod
    def triton(cls, pool, buf, loc, nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack):
        _set_k_and_s_triton(buf, loc, nope_fp8_rope_bf16_pack, pool.page_size)


def _set_k_and_s_triton(
    buf: torch.Tensor,
    loc: torch.Tensor,
    nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    page_size: int,
):
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape

    k_nope, k_rope, scale_k_nope = (
        nope_fp8_rope_bf16_pack.k_nope_fp8,
        nope_fp8_rope_bf16_pack.k_rope_bf16,
        nope_fp8_rope_bf16_pack.scale_k_nope_ue8m0,
    )

    num_tokens_to_write_nope, nope_dim = k_nope.shape
    num_tokens_to_write_rope, rope_dim = k_rope.shape
    num_tokens_to_write_scale, scale_dim = scale_k_nope.shape

    assert (
        num_tokens_to_write
        == num_tokens_to_write_nope
        == num_tokens_to_write_rope
        == num_tokens_to_write_scale
    )

    assert buf.dtype == torch.uint8
    assert loc.dtype in [torch.int64, torch.int32], f"{loc.dtype=}"

    assert k_nope.dtype == fp8_dtype
    assert k_rope.dtype == torch.bfloat16
    assert scale_k_nope.dtype == torch.uint8, f"{scale_k_nope.dtype=}"

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert k_nope.is_contiguous()
    assert k_rope.is_contiguous()
    assert scale_k_nope.is_contiguous()

    buf_fp8 = buf.view(fp8_dtype)
    buf_bf16 = buf.view(torch.bfloat16)
    buf_uint8 = buf.view(torch.uint8)

    nope_rope_bytes = nope_dim + rope_dim * 2
    s_offset_nbytes_in_page = page_size * (nope_dim + rope_dim * 2)

    _set_k_and_s_triton_kernel[(num_tokens_to_write,)](
        buf_fp8,
        buf_bf16,
        buf_uint8,
        loc,
        k_nope,
        k_rope,
        scale_k_nope,
        k_nope.stride(0),
        k_rope.stride(0),
        scale_k_nope.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_NOPE_ELEMS_PER_TOKEN=nope_dim,
        NUM_ROPE_ELEMS_PER_TOKEN=rope_dim,
        NUM_SCALE_ELEMS_PER_TOKEN=scale_dim,
        NUM_NOPE_ROPE_BYTES_PER_TOKEN=nope_rope_bytes,
        PADDED_SCALE_ELEMS_PER_TOKEN=scale_dim + 1,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        BLOCK_NOPE=512,
        BLOCK_ROPE=64,
        BLOCK_SCALE=8,
    )


@triton.jit
def _set_k_and_s_triton_kernel(
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    loc_ptr,
    k_nope_ptr,
    k_rope_ptr,
    scale_k_nope_ptr,
    k_nope_ptr_stride_0,
    k_rope_ptr_stride_0,
    scale_k_nope_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_NOPE_ELEMS_PER_TOKEN: tl.constexpr,
    NUM_ROPE_ELEMS_PER_TOKEN: tl.constexpr,
    NUM_NOPE_ROPE_BYTES_PER_TOKEN: tl.constexpr,
    NUM_SCALE_ELEMS_PER_TOKEN: tl.constexpr,
    PADDED_SCALE_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_SCALE: tl.constexpr,
):
    token_id = tl.program_id(0)
    loc = tl.load(loc_ptr + token_id)

    nope_range = tl.arange(0, BLOCK_NOPE)
    nope_mask = nope_range < NUM_NOPE_ELEMS_PER_TOKEN
    in_k_nope_offsets = token_id * k_nope_ptr_stride_0 + nope_range
    k_nope = tl.load(k_nope_ptr + in_k_nope_offsets, mask=nope_mask, other=0.0)

    rope_range = tl.arange(0, BLOCK_ROPE)
    in_k_rope_offsets = token_id * k_rope_ptr_stride_0 + rope_range
    k_rope = tl.load(k_rope_ptr + in_k_rope_offsets)

    scale_range = tl.arange(0, BLOCK_SCALE)
    scale_mask = scale_range < NUM_SCALE_ELEMS_PER_TOKEN
    in_scale_k_offsets = token_id * scale_k_nope_ptr_stride_0 + scale_range
    k_scale = tl.load(scale_k_nope_ptr + in_scale_k_offsets, mask=scale_mask, other=0)

    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    out_k_nope_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_NOPE_ROPE_BYTES_PER_TOKEN
        + nope_range
    )

    out_k_rope_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 2
        + loc_token_offset_in_page * (NUM_NOPE_ROPE_BYTES_PER_TOKEN // 2)
        + NUM_NOPE_ELEMS_PER_TOKEN // 2
        + rope_range
    )

    out_s_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + S_OFFSET_NBYTES_IN_PAGE
        + loc_token_offset_in_page * PADDED_SCALE_ELEMS_PER_TOKEN
        + scale_range
    )

    tl.store(buf_fp8_ptr + out_k_nope_offsets, k_nope, mask=nope_mask)
    tl.store(buf_bf16_ptr + out_k_rope_offsets, k_rope)
    tl.store(buf_uint8_ptr + out_s_offsets, k_scale, mask=scale_mask)


def _set_k_and_s_torch(
    buf: torch.Tensor,
    loc: torch.Tensor,
    nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    page_size: int,
):
    """Vectorized scatter that is CUDA-graph-capture safe."""
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens_to_write,) = loc.shape

    k_nope, k_rope, scale_k_nope = (
        nope_fp8_rope_bf16_pack.k_nope_fp8,
        nope_fp8_rope_bf16_pack.k_rope_bf16,
        nope_fp8_rope_bf16_pack.scale_k_nope_ue8m0,
    )

    num_tokens_to_write_nope, nope_dim = k_nope.shape
    num_tokens_to_write_rope, rope_dim = k_rope.shape
    num_tokens_to_write_scale, scale_dim = scale_k_nope.shape

    assert (
        num_tokens_to_write
        == num_tokens_to_write_nope
        == num_tokens_to_write_rope
        == num_tokens_to_write_scale
    ), f"{num_tokens_to_write=} {num_tokens_to_write_nope=} {num_tokens_to_write_rope=} {num_tokens_to_write_scale=}"

    assert buf.dtype == torch.uint8
    assert loc.dtype in [torch.int64, torch.int32], f"{loc.dtype=}"
    assert k_nope.dtype == fp8_dtype
    assert k_rope.dtype == torch.bfloat16
    assert scale_k_nope.dtype == torch.uint8
    assert buf.is_contiguous() and loc.is_contiguous()
    assert k_nope.is_contiguous() and k_rope.is_contiguous()
    assert scale_k_nope.is_contiguous()

    device = buf.device
    loc = loc.to(torch.int64)
    loc_page_index = (loc // page_size).to(torch.int64)
    loc_token_offset_in_page = (loc % page_size).to(torch.int64)

    nope_rope_bytes = nope_dim + rope_dim * 2
    s_offset_nbytes_in_page = page_size * nope_rope_bytes

    # --- Nope (FP8, 1 byte/el) via uint8 view ---
    flat_u8 = buf.flatten()
    nope_base = (
        loc_page_index * buf_numel_per_page
        + loc_token_offset_in_page * nope_rope_bytes
    )
    nope_idx = nope_base[:, None] + torch.arange(
        nope_dim, device=device, dtype=torch.int64
    )[None, :]
    nope_idx = nope_idx.reshape(-1)[: num_tokens_to_write * nope_dim]
    flat_u8[nope_idx] = k_nope.view(torch.uint8).reshape(-1)

    # --- Rope (BF16, 2 bytes/el) via bf16 view ---
    flat_bf16 = buf.view(torch.bfloat16).flatten()
    rope_base = (
        loc_page_index * (buf_numel_per_page // 2)
        + (loc_token_offset_in_page * nope_rope_bytes + nope_dim) // 2
    )
    rope_idx = rope_base[:, None] + torch.arange(
        rope_dim, device=device, dtype=torch.int64
    )[None, :]
    rope_idx = rope_idx.reshape(-1)[: num_tokens_to_write * rope_dim]
    flat_bf16[rope_idx] = k_rope.reshape(-1)

    # --- Scale (uint8, 1 byte/padded el) via uint8 view ---
    padded_scale_dim = scale_dim + 1
    scale_base = (
        loc_page_index * buf_numel_per_page
        + s_offset_nbytes_in_page
        + loc_token_offset_in_page * padded_scale_dim
    )
    scale_idx = scale_base[:, None] + torch.arange(
        scale_dim, device=device, dtype=torch.int64
    )[None, :]
    scale_idx = scale_idx.reshape(-1)[: num_tokens_to_write * scale_dim]
    flat_u8[scale_idx] = scale_k_nope.reshape(-1)
