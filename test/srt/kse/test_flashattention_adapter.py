"""Tests for the FlashAttention MetadataAdapter."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.adapters.flashattention_adapter import FlashAttentionAdapter
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.types import Granularity, SelectionResult

from mock_utils import (
    MockFAMetadata,
    MockKVCache,
    MockReqToTokenPool,
    build_identity_req_to_token,
    make_decode_batch,
)

BACKEND_PAGE_SIZE = 16


def _make_adapter(sparse_page_size=16):
    cfg = KSEConfig(
        policy_name="quest",
        backend_name="flashattention",
        page_size=sparse_page_size,
    )
    return FlashAttentionAdapter(cfg, torch.device("cpu"))


def _make_fa_metadata(bs, num_pages_per_req, page_size=BACKEND_PAGE_SIZE):
    """Build a mock FlashAttention metadata with sequential page ids."""
    page_table = torch.zeros(bs, num_pages_per_req, dtype=torch.int32)
    cache_seqlens = torch.zeros(bs, dtype=torch.int32)
    for i in range(bs):
        base = i * num_pages_per_req
        page_table[i] = torch.arange(
            base, base + num_pages_per_req, dtype=torch.int32
        )
        cache_seqlens[i] = num_pages_per_req * page_size
    max_seq_len_k = int(cache_seqlens.max().item())
    return MockFAMetadata(
        page_table=page_table,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=max_seq_len_k,
    )


class TestSaveRestore(unittest.TestCase):
    def test_save_and_restore(self):
        adapter = _make_adapter()
        meta = _make_fa_metadata(bs=2, num_pages_per_req=8)
        orig_pt = meta.page_table.clone()
        orig_sl = meta.cache_seqlens_int32.clone()
        orig_max = meta.max_seq_len_k

        adapter.save_dense_metadata(meta)

        # Corrupt
        meta.page_table.zero_()
        meta.cache_seqlens_int32.zero_()
        meta.max_seq_len_k = 0

        adapter.restore_dense_metadata(meta)
        self.assertTrue(torch.equal(meta.page_table, orig_pt))
        self.assertTrue(torch.equal(meta.cache_seqlens_int32, orig_sl))
        self.assertEqual(meta.max_seq_len_k, orig_max)


class TestApplyPageSelection(unittest.TestCase):
    def test_select_pages(self):
        adapter = _make_adapter(sparse_page_size=BACKEND_PAGE_SIZE)
        pool = MockReqToTokenPool(2, 256)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 128)
        batch = make_decode_batch([128], pool, kv)

        # 128 tokens / 16 page_size = 8 pages
        meta = _make_fa_metadata(bs=1, num_pages_per_req=8)
        adapter.save_dense_metadata(meta)

        # Select sparse pages [1, 3] (pages_per_sparse = 1)
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[1, 3]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # page_table[0, 0:2] should be [1, 3] (the physical page ids)
        self.assertEqual(meta.page_table[0, 0].item(), 1)
        self.assertEqual(meta.page_table[0, 1].item(), 3)

    def test_cache_seqlens_updated(self):
        adapter = _make_adapter(sparse_page_size=BACKEND_PAGE_SIZE)
        pool = MockReqToTokenPool(2, 256)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 128)
        batch = make_decode_batch([128], pool, kv)

        meta = _make_fa_metadata(bs=1, num_pages_per_req=8)
        adapter.save_dense_metadata(meta)

        # Select 3 pages
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[0, 2, 5]], dtype=torch.int32),
            valid_lengths=torch.tensor([3], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)
        self.assertEqual(
            meta.cache_seqlens_int32[0].item(), 3 * BACKEND_PAGE_SIZE
        )

    def test_max_seq_len_k_updated(self):
        adapter = _make_adapter(sparse_page_size=BACKEND_PAGE_SIZE)
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 128)
        build_identity_req_to_token(pool, 1, 128, offset=128)
        batch = make_decode_batch([128, 128], pool, kv)

        meta = _make_fa_metadata(bs=2, num_pages_per_req=8)
        adapter.save_dense_metadata(meta)

        # Req 0: select 2 pages, Req 1: select 5 pages
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor(
                [[0, 1, -1, -1, -1], [0, 1, 2, 3, 4]], dtype=torch.int32
            ),
            valid_lengths=torch.tensor([2, 5], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)
        self.assertEqual(meta.max_seq_len_k, 5 * BACKEND_PAGE_SIZE)


class TestTokenGranularityRaises(unittest.TestCase):
    def test_raises_runtime_error(self):
        adapter = _make_adapter()
        pool = MockReqToTokenPool(2, 64)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 32)
        batch = make_decode_batch([32], pool, kv)

        meta = _make_fa_metadata(bs=1, num_pages_per_req=2)
        adapter.save_dense_metadata(meta)

        result = SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.tensor([[0, 1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        with self.assertRaises(RuntimeError):
            adapter.apply(result, meta, batch, layer_id=0)


class TestNonSparseUnchanged(unittest.TestCase):
    def test_dense_request_page_table_unchanged(self):
        adapter = _make_adapter(sparse_page_size=BACKEND_PAGE_SIZE)
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 64)
        build_identity_req_to_token(pool, 1, 64, offset=64)
        batch = make_decode_batch([64, 64], pool, kv)

        meta = _make_fa_metadata(bs=2, num_pages_per_req=4)
        adapter.save_dense_metadata(meta)
        orig_pt_1 = meta.page_table[1].clone()
        orig_sl_1 = meta.cache_seqlens_int32[1].clone()

        # Req 0: sparse, Req 1: dense
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[0, -1], [0, 0]], dtype=torch.int32),
            valid_lengths=torch.tensor([1, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # Request 1 should be unchanged
        self.assertTrue(torch.equal(meta.page_table[1], orig_pt_1))
        self.assertEqual(
            meta.cache_seqlens_int32[1].item(), orig_sl_1.item()
        )


class TestMultiPageSparse(unittest.TestCase):
    def test_sparse_page_is_multiple_of_backend(self):
        # sparse_page_size = 32 = 2 * backend_page_size(16)
        adapter = _make_adapter(sparse_page_size=32)
        pool = MockReqToTokenPool(2, 256)
        kv = MockKVCache(1, 256, 2, 64, page_size=BACKEND_PAGE_SIZE)
        build_identity_req_to_token(pool, 0, 128)
        batch = make_decode_batch([128], pool, kv)

        # 128 tokens / 16 = 8 backend pages
        meta = _make_fa_metadata(bs=1, num_pages_per_req=8)
        adapter.save_dense_metadata(meta)

        # Select sparse page 1 → backend pages [2, 3]
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[1]], dtype=torch.int32),
            valid_lengths=torch.tensor([1], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # page_table[0, 0:2] should be physical pages [2, 3]
        self.assertEqual(meta.page_table[0, 0].item(), 2)
        self.assertEqual(meta.page_table[0, 1].item(), 3)
        self.assertEqual(
            meta.cache_seqlens_int32[0].item(), 2 * BACKEND_PAGE_SIZE
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
