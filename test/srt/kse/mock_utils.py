"""Shared mock utilities for KSE unit tests.

All mocks are lightweight Python objects with the minimal attributes
needed by the KSE components.  No sglang server or GPU is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Minimal ForwardMode stub
# ---------------------------------------------------------------------------

class MockForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()

    def is_decode(self):
        return self == MockForwardMode.DECODE

    def is_extend(self):
        return self == MockForwardMode.EXTEND


# ---------------------------------------------------------------------------
# Mock ReqToTokenPool
# ---------------------------------------------------------------------------

class MockReqToTokenPool:
    """Mimics ``ReqToTokenPool`` with a plain ``req_to_token`` tensor."""

    def __init__(self, size: int, max_context_len: int, device: str = "cpu"):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )


# ---------------------------------------------------------------------------
# Mock KVCache
# ---------------------------------------------------------------------------

class MockKVCache:
    """Mimics ``KVCache`` with simple k/v buffer tensors per layer."""

    def __init__(
        self,
        layer_num: int,
        pool_size: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 1,
        device: str = "cpu",
    ):
        self.layer_num = layer_num
        self.pool_size = pool_size
        self.page_size = page_size
        self.device = device
        self._k_buffers: Dict[int, torch.Tensor] = {}
        self._v_buffers: Dict[int, torch.Tensor] = {}
        for lid in range(layer_num):
            self._k_buffers[lid] = torch.randn(
                pool_size, num_kv_heads, head_dim, device=device
            )
            self._v_buffers[lid] = torch.randn(
                pool_size, num_kv_heads, head_dim, device=device
            )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._k_buffers[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self._v_buffers[layer_id]


# ---------------------------------------------------------------------------
# Mock ForwardBatch
# ---------------------------------------------------------------------------

@dataclass
class MockForwardBatch:
    """Mimics the subset of ``ForwardBatch`` fields used by KSE."""

    forward_mode: MockForwardMode
    batch_size: int
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    req_to_token_pool: MockReqToTokenPool
    token_to_kv_pool: MockKVCache
    out_cache_loc: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Mock Triton ForwardMetadata
# ---------------------------------------------------------------------------

@dataclass
class MockTritonMetadata:
    """Mimics the Triton backend ``ForwardMetadata``."""

    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor


# ---------------------------------------------------------------------------
# Mock FlashAttention ForwardMetadata
# ---------------------------------------------------------------------------

@dataclass
class MockFAMetadata:
    """Mimics ``FlashAttentionMetadata``."""

    page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    max_seq_len_k: int = 0


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def build_identity_req_to_token(
    pool: MockReqToTokenPool,
    req_idx: int,
    seq_len: int,
    offset: int = 0,
) -> None:
    """Fill ``req_to_token[req_idx, 0:seq_len]`` with ``offset, offset+1, ...``."""
    pool.req_to_token[req_idx, :seq_len] = torch.arange(
        offset, offset + seq_len, dtype=torch.int32, device=pool.device
    )


def make_decode_batch(
    seq_lens: list[int],
    pool: MockReqToTokenPool,
    kv_cache: MockKVCache,
    device: str = "cpu",
) -> MockForwardBatch:
    """Build a decode-mode ``MockForwardBatch`` for the given sequence lengths."""
    bs = len(seq_lens)
    return MockForwardBatch(
        forward_mode=MockForwardMode.DECODE,
        batch_size=bs,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device=device),
        req_to_token_pool=pool,
        token_to_kv_pool=kv_cache,
    )


def make_extend_batch(
    seq_lens: list[int],
    pool: MockReqToTokenPool,
    kv_cache: MockKVCache,
    device: str = "cpu",
) -> MockForwardBatch:
    """Build an extend-mode ``MockForwardBatch``."""
    bs = len(seq_lens)
    return MockForwardBatch(
        forward_mode=MockForwardMode.EXTEND,
        batch_size=bs,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device=device),
        req_to_token_pool=pool,
        token_to_kv_pool=kv_cache,
    )
