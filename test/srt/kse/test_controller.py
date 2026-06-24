"""Tests for KSEController coordination logic."""

import os
import sys
import unittest
from unittest.mock import MagicMock

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.controller import KSEController
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

from mock_utils import (
    MockForwardMode,
    MockKVCache,
    MockReqToTokenPool,
    MockTritonMetadata,
    build_identity_req_to_token,
    make_decode_batch,
    make_extend_batch,
)


# ---------------------------------------------------------------------------
# Concrete stub policies for testing
# ---------------------------------------------------------------------------

class _StubTokenPolicy(SparsityPolicy):
    """TOKEN granularity, PER_REQUEST frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.TOKEN

    def frequency(self):
        return Frequency.PER_REQUEST

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _StubPagePolicy(SparsityPolicy):
    """PAGE granularity, PER_LAYER frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.PAGE

    def frequency(self):
        return Frequency.PER_LAYER

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _StubPerStepPolicy(SparsityPolicy):
    """TOKEN granularity, PER_STEP frequency."""

    def __init__(self):
        self.select_call_count = 0

    def granularity(self):
        return Granularity.TOKEN

    def frequency(self):
        return Frequency.PER_STEP

    def select(self, query, layer_id, req_pool_indices, seq_lens,
               forward_batch, **kw):
        self.select_call_count += 1
        bs = seq_lens.shape[0]
        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.zeros(bs, 1, dtype=torch.int32),
            valid_lengths=torch.ones(bs, dtype=torch.int32),
            sparse_mask=torch.ones(bs, dtype=torch.bool),
        )


class _NoopAdapter(MetadataAdapter):
    def __init__(self):
        self.save_count = 0
        self.apply_count = 0

    def save_dense_metadata(self, forward_metadata):
        self.save_count += 1

    def apply(self, result, forward_metadata, forward_batch, layer_id):
        self.apply_count += 1
        return forward_metadata

    def restore_dense_metadata(self, forward_metadata):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_controller(policy, adapter=None, backend="triton",
                     page_size=64, end_layer=-1):
    if adapter is None:
        adapter = _NoopAdapter()
    pool = MockReqToTokenPool(4, 512)
    kv = MockKVCache(4, 512, 2, 64)
    cfg = KSEConfig(
        policy_name="test",
        backend_name=backend,
        page_size=page_size,
        end_layer=end_layer,
    )
    return KSEController(
        policy=policy,
        adapter=adapter,
        req_to_token_pool=pool,
        token_to_kv_pool=kv,
        config=cfg,
    )


class TestGranularityCompatibility(unittest.TestCase):
    def test_token_policy_on_page_backend_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubTokenPolicy(), backend="flashattention",
                             page_size=256)
        self.assertIn("incompatible", str(ctx.exception))

    def test_page_policy_on_token_backend_ok(self):
        ctrl = _make_controller(_StubPagePolicy(), backend="triton")
        self.assertIsNotNone(ctrl)

    def test_page_policy_small_page_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubPagePolicy(), backend="flashattention",
                             page_size=128)
        self.assertIn("page_size", str(ctx.exception))

    def test_page_policy_non_multiple_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_controller(_StubPagePolicy(), backend="flashattention",
                             page_size=300)
        self.assertIn("integer multiple", str(ctx.exception))

    def test_page_policy_valid_multiple_ok(self):
        ctrl = _make_controller(_StubPagePolicy(), backend="flashattention",
                                page_size=256)
        self.assertIsNotNone(ctrl)


class TestShouldApply(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller(_StubPagePolicy(), end_layer=3)
        self.pool = self.ctrl.req_to_token_pool
        self.kv = self.ctrl.token_to_kv_pool

    def test_decode_in_range(self):
        batch = make_decode_batch([128], self.pool, self.kv)
        self.assertTrue(self.ctrl._should_apply(0, batch))
        self.assertTrue(self.ctrl._should_apply(2, batch))

    def test_extend_mode_false(self):
        batch = make_extend_batch([128], self.pool, self.kv)
        self.assertFalse(self.ctrl._should_apply(0, batch))

    def test_layer_out_of_range(self):
        batch = make_decode_batch([128], self.pool, self.kv)
        self.assertFalse(self.ctrl._should_apply(3, batch))
        self.assertFalse(self.ctrl._should_apply(10, batch))

    def test_end_layer_minus_one_uses_layer_num(self):
        ctrl = _make_controller(_StubPagePolicy(), end_layer=-1)
        batch = make_decode_batch([128], ctrl.req_to_token_pool,
                                  ctrl.token_to_kv_pool)
        self.assertTrue(ctrl._should_apply(3, batch))
        self.assertFalse(ctrl._should_apply(4, batch))


class TestFrequencyCaching(unittest.TestCase):
    def test_per_request_caches_after_prefill(self):
        policy = _StubTokenPolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        build_identity_req_to_token(pool, 0, 64)
        batch = make_decode_batch([64], pool, kv)

        ctrl.after_prefill(batch)
        self.assertIsNotNone(ctrl._cached_result)
        self.assertEqual(policy.select_call_count, 1)

        r1 = ctrl._get_selection(None, 0, batch)
        r2 = ctrl._get_selection(None, 1, batch)
        self.assertIs(r1, r2)
        self.assertEqual(policy.select_call_count, 1)

    def test_per_step_recomputes_each_step(self):
        policy = _StubPerStepPolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)

        ctrl.before_forward(batch)
        r1 = ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        self.assertEqual(policy.select_call_count, 1)
        r2 = ctrl._get_selection(torch.randn(1, 2, 64), 1, batch)
        self.assertIs(r1, r2)
        self.assertEqual(policy.select_call_count, 1)

        ctrl.before_forward(batch)
        r3 = ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        self.assertEqual(policy.select_call_count, 2)
        self.assertIsNot(r1, r3)

    def test_per_layer_always_recomputes(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)

        ctrl._get_selection(torch.randn(1, 2, 64), 0, batch)
        ctrl._get_selection(torch.randn(1, 2, 64), 1, batch)
        ctrl._get_selection(torch.randn(1, 2, 64), 2, batch)
        self.assertEqual(policy.select_call_count, 3)


class TestBeforeAttention(unittest.TestCase):
    def test_saves_metadata_once(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter, end_layer=4)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_decode_batch([64], pool, kv)
        metadata = MagicMock()

        ctrl.before_forward(batch)
        ctrl.before_attention(torch.randn(1, 2, 64), 0, batch, metadata)
        ctrl.before_attention(torch.randn(1, 2, 64), 1, batch, metadata)
        ctrl.before_attention(torch.randn(1, 2, 64), 2, batch, metadata)

        self.assertEqual(adapter.save_count, 1)
        self.assertEqual(adapter.apply_count, 3)

    def test_skips_non_decode(self):
        policy = _StubPagePolicy()
        adapter = _NoopAdapter()
        ctrl = _make_controller(policy, adapter, end_layer=4)
        pool = ctrl.req_to_token_pool
        kv = ctrl.token_to_kv_pool
        batch = make_extend_batch([64], pool, kv)
        metadata = MagicMock()

        result = ctrl.before_attention(torch.randn(1, 2, 64), 0, batch, metadata)
        self.assertIs(result, metadata)
        self.assertEqual(adapter.apply_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
