"""Tests for the Quest sparsity policy."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.policies.quest import QuestPolicy
from sglang.srt.layers.kse.types import Frequency, Granularity

from mock_utils import (
    MockKVCache,
    MockReqToTokenPool,
    build_identity_req_to_token,
    make_decode_batch,
)

PAGE_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
POOL_SIZE = 256
LAYER_NUM = 2


def _make_quest(
    min_seq_len=16,
    token_budget_ratio=0.5,
    num_recent_pages=1,
):
    cfg = KSEConfig(
        policy_name="quest",
        backend_name="triton",
        page_size=PAGE_SIZE,
        min_seq_len=min_seq_len,
        policy_kwargs={
            "token_budget_ratio": token_budget_ratio,
            "num_recent_pages": num_recent_pages,
            "max_pool_pages": 128,
        },
    )
    return QuestPolicy(cfg, torch.device("cpu"))


def _make_env(seq_lens):
    """Create pool, kv_cache, and batch with identity token mapping."""
    bs = len(seq_lens)
    max_ctx = max(seq_lens) + 64
    pool = MockReqToTokenPool(bs + 1, max_ctx)
    kv = MockKVCache(LAYER_NUM, POOL_SIZE, NUM_KV_HEADS, HEAD_DIM, page_size=1)
    offset = 0
    for i, sl in enumerate(seq_lens):
        build_identity_req_to_token(pool, i, sl, offset=offset)
        offset += sl
    batch = make_decode_batch(seq_lens, pool, kv)
    return pool, kv, batch


class TestQuestBasic(unittest.TestCase):
    def test_granularity(self):
        q = _make_quest()
        self.assertEqual(q.granularity(), Granularity.PAGE)

    def test_frequency(self):
        q = _make_quest()
        self.assertEqual(q.frequency(), Frequency.PER_LAYER)


class TestQuestPrefill(unittest.TestCase):
    def test_builds_bounding_boxes(self):
        quest = _make_quest()
        pool, kv, batch = _make_env([16])

        quest.on_prefill_complete(
            layer_id=0,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            k_buffer=kv.get_key_buffer(0),
            v_buffer=kv.get_value_buffer(0),
            forward_batch=batch,
        )

        self.assertIn(0, quest._page_k_min)
        self.assertIn(0, quest._page_k_max)

    def test_bounding_box_values(self):
        quest = _make_quest()
        pool, kv, batch = _make_env([8])

        # Overwrite k_buffer with known values: 2 pages of 4 tokens each
        k_buf = kv.get_key_buffer(0)
        # Page 0 (tokens 0-3): values from 1.0 to 4.0
        for t in range(4):
            k_buf[t] = torch.full((NUM_KV_HEADS, HEAD_DIM), float(t + 1))
        # Page 1 (tokens 4-7): values from 10.0 to 13.0
        for t in range(4, 8):
            k_buf[t] = torch.full((NUM_KV_HEADS, HEAD_DIM), float(t + 6))

        quest.on_prefill_complete(
            layer_id=0,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            k_buffer=k_buf,
            v_buffer=kv.get_value_buffer(0),
            forward_batch=batch,
        )

        # Physical page 0 (tokens 0-3): min=1.0, max=4.0
        phys_page_0 = 0  # token 0 // page_size(4) = 0
        self.assertAlmostEqual(
            quest._page_k_min[0][phys_page_0, 0, 0].item(), 1.0, places=4
        )
        self.assertAlmostEqual(
            quest._page_k_max[0][phys_page_0, 0, 0].item(), 4.0, places=4
        )

        # Physical page 1 (tokens 4-7): min=10.0, max=13.0
        phys_page_1 = 1  # token 4 // page_size(4) = 1
        self.assertAlmostEqual(
            quest._page_k_min[0][phys_page_1, 0, 0].item(), 10.0, places=4
        )
        self.assertAlmostEqual(
            quest._page_k_max[0][phys_page_1, 0, 0].item(), 13.0, places=4
        )


class TestQuestSelect(unittest.TestCase):
    def test_returns_page_granularity(self):
        quest = _make_quest(min_seq_len=8)
        pool, kv, batch = _make_env([16])
        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            kv.get_key_buffer(0), kv.get_value_buffer(0), batch,
        )
        query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)
        self.assertEqual(result.granularity, Granularity.PAGE)

    def test_short_seq_not_sparse(self):
        quest = _make_quest(min_seq_len=32)
        pool, kv, batch = _make_env([16])
        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            kv.get_key_buffer(0), kv.get_value_buffer(0), batch,
        )
        query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)
        self.assertFalse(result.sparse_mask[0].item())

    def test_always_includes_recent_pages(self):
        quest = _make_quest(min_seq_len=8, token_budget_ratio=0.1,
                            num_recent_pages=2)
        pool, kv, batch = _make_env([40])  # 10 pages
        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            kv.get_key_buffer(0), kv.get_value_buffer(0), batch,
        )
        query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)

        selected = result.selected_indices[0, :result.valid_lengths[0].item()]
        num_pages = 40 // PAGE_SIZE  # 10
        # Last 2 pages (indices 8, 9) must be present
        self.assertIn(8, selected.tolist())
        self.assertIn(9, selected.tolist())

    def test_budget_ratio_respected(self):
        quest = _make_quest(min_seq_len=8, token_budget_ratio=0.3,
                            num_recent_pages=1)
        pool, kv, batch = _make_env([40])  # 10 pages
        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            kv.get_key_buffer(0), kv.get_value_buffer(0), batch,
        )
        query = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)

        n_selected = result.valid_lengths[0].item()
        # budget = max(1, int(10 * 0.3)) = 3 pages from topk + 1 recent
        # After unique, should be between 3 and 4
        self.assertGreaterEqual(n_selected, 3)
        self.assertLessEqual(n_selected, 5)

    def test_scores_prefer_high_key_pages(self):
        quest = _make_quest(min_seq_len=8, token_budget_ratio=0.2,
                            num_recent_pages=0)
        pool, kv, batch = _make_env([20])  # 5 pages

        k_buf = kv.get_key_buffer(0)
        # Make page 2 (tokens 8-11) have very high keys
        for t in range(20):
            k_buf[t] = torch.full((NUM_KV_HEADS, HEAD_DIM), 0.1)
        for t in range(8, 12):
            k_buf[t] = torch.full((NUM_KV_HEADS, HEAD_DIM), 100.0)

        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            k_buf, kv.get_value_buffer(0), batch,
        )

        # Use a positive query so high keys → high scores
        query = torch.ones(1, NUM_KV_HEADS, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)

        selected = result.selected_indices[0, :result.valid_lengths[0].item()]
        self.assertIn(2, selected.tolist())

    def test_gqa_head_mismatch(self):
        quest = _make_quest(min_seq_len=8)
        pool, kv, batch = _make_env([16])
        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            kv.get_key_buffer(0), kv.get_value_buffer(0), batch,
        )
        # 4 query heads, 2 kv heads → GQA ratio = 2
        query = torch.randn(1, 4, HEAD_DIM)
        result = quest.select(query, 0, batch.req_pool_indices,
                              batch.seq_lens, batch)
        self.assertTrue(result.sparse_mask[0].item())


class TestQuestIncrementalUpdate(unittest.TestCase):
    def test_bbox_expands_on_new_token(self):
        quest = _make_quest(min_seq_len=8)
        pool, kv, batch = _make_env([8])

        k_buf = kv.get_key_buffer(0)
        # All keys = 5.0
        for t in range(8):
            k_buf[t] = torch.full((NUM_KV_HEADS, HEAD_DIM), 5.0)

        quest.on_prefill_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            k_buf, kv.get_value_buffer(0), batch,
        )

        # Now simulate a new token at position 7 (last page) with value 20.0
        k_buf[7] = torch.full((NUM_KV_HEADS, HEAD_DIM), 20.0)

        quest.on_attention_complete(
            0, batch.req_pool_indices, batch.seq_lens,
            k_buf, kv.get_value_buffer(0), batch,
        )

        phys_page_1 = 1  # tokens 4-7
        self.assertAlmostEqual(
            quest._page_k_max[0][phys_page_1, 0, 0].item(), 20.0, places=4
        )
        # Min should still be 5.0
        self.assertAlmostEqual(
            quest._page_k_min[0][phys_page_1, 0, 0].item(), 5.0, places=4
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
