"""Tests for the Triton MetadataAdapter."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.adapters.triton_adapter import TritonAdapter
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.types import Granularity, SelectionResult

from mock_utils import (
    MockKVCache,
    MockReqToTokenPool,
    MockTritonMetadata,
    build_identity_req_to_token,
    make_decode_batch,
)


def _make_adapter(page_size=4):
    cfg = KSEConfig(policy_name="quest", backend_name="triton", page_size=page_size)
    return TritonAdapter(cfg, torch.device("cpu"))


def _make_dense_metadata(seq_lens, pool):
    """Build a dense Triton metadata with identity-mapped kv_indices."""
    bs = len(seq_lens)
    indptr = torch.zeros(bs + 1, dtype=torch.int64)
    for i, sl in enumerate(seq_lens):
        indptr[i + 1] = indptr[i] + sl

    total = int(indptr[-1].item())
    indices = torch.empty(total, dtype=torch.int64)
    for i, sl in enumerate(seq_lens):
        start = int(indptr[i].item())
        req_idx = i
        indices[start:start + sl] = pool.req_to_token[req_idx, :sl].long()

    return MockTritonMetadata(kv_indptr=indptr, kv_indices=indices)


class TestSaveRestore(unittest.TestCase):
    def test_save_and_restore(self):
        adapter = _make_adapter()
        pool = MockReqToTokenPool(2, 64)
        build_identity_req_to_token(pool, 0, 8, offset=100)
        build_identity_req_to_token(pool, 1, 6, offset=200)

        meta = _make_dense_metadata([8, 6], pool)
        orig_indptr = meta.kv_indptr.clone()
        orig_indices = meta.kv_indices.clone()

        adapter.save_dense_metadata(meta)

        # Corrupt metadata
        meta.kv_indptr = torch.zeros(3, dtype=torch.int64)
        meta.kv_indices = torch.zeros(1, dtype=torch.int64)

        adapter.restore_dense_metadata(meta)
        self.assertTrue(torch.equal(meta.kv_indptr, orig_indptr))
        self.assertTrue(torch.equal(meta.kv_indices, orig_indices))


class TestApplyTokenSelection(unittest.TestCase):
    def test_select_specific_tokens(self):
        adapter = _make_adapter()
        pool = MockReqToTokenPool(2, 64)
        kv = MockKVCache(1, 256, 2, 64)
        # Request 0: physical tokens [100, 101, 102, 103, 104, 105, 106, 107]
        build_identity_req_to_token(pool, 0, 8, offset=100)
        batch = make_decode_batch([8], pool, kv)

        meta = _make_dense_metadata([8], pool)
        adapter.save_dense_metadata(meta)

        # Select tokens at logical positions [0, 2, 5]
        result = SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.tensor([[0, 2, 5]], dtype=torch.int32),
            valid_lengths=torch.tensor([3], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # kv_indices should contain physical indices [100, 102, 105]
        expected = torch.tensor([100, 102, 105], dtype=torch.int64)
        self.assertTrue(torch.equal(meta.kv_indices, expected))
        # indptr: [0, 3]
        self.assertEqual(meta.kv_indptr[0].item(), 0)
        self.assertEqual(meta.kv_indptr[1].item(), 3)


class TestApplyPageSelection(unittest.TestCase):
    def test_select_pages(self):
        adapter = _make_adapter(page_size=4)
        pool = MockReqToTokenPool(2, 64)
        kv = MockKVCache(1, 256, 2, 64)
        # Request 0: 16 tokens → 4 pages
        build_identity_req_to_token(pool, 0, 16, offset=0)
        batch = make_decode_batch([16], pool, kv)

        meta = _make_dense_metadata([16], pool)
        adapter.save_dense_metadata(meta)

        # Select pages 0 and 2 (logical page indices)
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[0, 2]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # Page 0: tokens 0-3, Page 2: tokens 8-11
        expected = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11], dtype=torch.int64)
        self.assertTrue(torch.equal(meta.kv_indices, expected))
        self.assertEqual(meta.kv_indptr[1].item(), 8)

    def test_page_clamps_to_seq_len(self):
        adapter = _make_adapter(page_size=4)
        pool = MockReqToTokenPool(2, 64)
        kv = MockKVCache(1, 256, 2, 64)
        # Request 0: 6 tokens → page 0 (4 tokens) + page 1 (2 tokens)
        build_identity_req_to_token(pool, 0, 6, offset=50)
        batch = make_decode_batch([6], pool, kv)

        meta = _make_dense_metadata([6], pool)
        adapter.save_dense_metadata(meta)

        # Select page 1 (tokens 4-5, not 4-7)
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[1]], dtype=torch.int32),
            valid_lengths=torch.tensor([1], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # Only tokens 4,5 → physical 54, 55
        expected = torch.tensor([54, 55], dtype=torch.int64)
        self.assertTrue(torch.equal(meta.kv_indices, expected))


class TestMixedSparseDense(unittest.TestCase):
    def test_dense_request_unchanged(self):
        adapter = _make_adapter()
        pool = MockReqToTokenPool(4, 64)
        kv = MockKVCache(1, 256, 2, 64)
        build_identity_req_to_token(pool, 0, 8, offset=100)
        build_identity_req_to_token(pool, 1, 6, offset=200)
        batch = make_decode_batch([8, 6], pool, kv)

        meta = _make_dense_metadata([8, 6], pool)
        adapter.save_dense_metadata(meta)

        # Request 0: sparse, Request 1: dense
        result = SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.tensor([[0, 3, -1], [0, 0, 0]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 0], dtype=torch.int32),
            sparse_mask=torch.tensor([True, False]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        # Request 0: selected tokens [0, 3] → physical [100, 103]
        # Request 1: dense → all 6 tokens [200..205]
        req0_start = int(meta.kv_indptr[0].item())
        req0_end = int(meta.kv_indptr[1].item())
        req1_start = int(meta.kv_indptr[1].item())
        req1_end = int(meta.kv_indptr[2].item())

        self.assertEqual(req0_end - req0_start, 2)
        self.assertEqual(req1_end - req1_start, 6)

        req0_indices = meta.kv_indices[req0_start:req0_end]
        self.assertTrue(torch.equal(
            req0_indices, torch.tensor([100, 103], dtype=torch.int64)
        ))

        req1_indices = meta.kv_indices[req1_start:req1_end]
        expected_1 = torch.arange(200, 206, dtype=torch.int64)
        self.assertTrue(torch.equal(req1_indices, expected_1))


class TestIndptrConsistency(unittest.TestCase):
    def test_indptr_diffs_match_token_counts(self):
        adapter = _make_adapter(page_size=4)
        pool = MockReqToTokenPool(4, 64)
        kv = MockKVCache(1, 256, 2, 64)
        build_identity_req_to_token(pool, 0, 12, offset=0)
        build_identity_req_to_token(pool, 1, 8, offset=100)
        batch = make_decode_batch([12, 8], pool, kv)

        meta = _make_dense_metadata([12, 8], pool)
        adapter.save_dense_metadata(meta)

        # Request 0: select pages [0, 2] → 8 tokens
        # Request 1: select page [1] → 4 tokens
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.tensor([[0, 2], [1, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2, 1], dtype=torch.int32),
            sparse_mask=torch.tensor([True, True]),
        )

        adapter.apply(result, meta, batch, layer_id=0)

        diff0 = meta.kv_indptr[1].item() - meta.kv_indptr[0].item()
        diff1 = meta.kv_indptr[2].item() - meta.kv_indptr[1].item()
        self.assertEqual(diff0, 8)
        self.assertEqual(diff1, 4)
        self.assertEqual(meta.kv_indices.shape[0], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
