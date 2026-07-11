"""Tests for the KSE policy/adapter registry and factory."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.controller import KSEController
from sglang.srt.layers.kse.registry import (
    _ADAPTER_REGISTRY,
    _POLICY_REGISTRY,
    create_kse_controller,
    register_policy,
)
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

# Ensure built-in policies/adapters are registered
import sglang.srt.layers.kse.policies  # noqa: F401
import sglang.srt.layers.kse.adapters  # noqa: F401

from mock_utils import MockKVCache, MockReqToTokenPool


class TestBuiltinRegistration(unittest.TestCase):
    def test_builtin_policies_registered(self):
        self.assertIn("quest", _POLICY_REGISTRY)
        self.assertIn("streaming_llm", _POLICY_REGISTRY)

    def test_builtin_adapters_registered(self):
        self.assertIn("triton", _ADAPTER_REGISTRY)
        self.assertIn("flashattention", _ADAPTER_REGISTRY)
        self.assertIn("flashinfer", _ADAPTER_REGISTRY)


class TestRegisterCustomPolicy(unittest.TestCase):
    def test_register_and_lookup(self):
        @register_policy("_test_dummy_policy")
        class DummyPolicy(SparsityPolicy):
            def __init__(self, config, device):
                pass

            def granularity(self):
                return Granularity.TOKEN

            def frequency(self):
                return Frequency.PER_REQUEST

            def select(self, query, layer_id, req_pool_indices, seq_lens,
                       forward_batch, **kw):
                return SelectionResult(
                    granularity=Granularity.TOKEN,
                    selected_indices=torch.zeros(1, 1, dtype=torch.int32),
                    valid_lengths=torch.ones(1, dtype=torch.int32),
                    sparse_mask=torch.ones(1, dtype=torch.bool),
                )

        self.assertIn("_test_dummy_policy", _POLICY_REGISTRY)
        self.assertIs(_POLICY_REGISTRY["_test_dummy_policy"], DummyPolicy)
        # Cleanup
        del _POLICY_REGISTRY["_test_dummy_policy"]


class TestCreateController(unittest.TestCase):
    def test_unknown_policy_raises(self):
        cfg = KSEConfig(policy_name="nonexistent", backend_name="triton")
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(2, 256, 2, 64)
        with self.assertRaises(ValueError):
            create_kse_controller(cfg, pool, kv, torch.device("cpu"))

    def test_unknown_adapter_raises(self):
        cfg = KSEConfig(policy_name="quest", backend_name="nonexistent")
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(2, 256, 2, 64)
        with self.assertRaises(ValueError):
            create_kse_controller(cfg, pool, kv, torch.device("cpu"))

    def test_create_quest_triton_success(self):
        cfg = KSEConfig(
            policy_name="quest",
            backend_name="triton",
            page_size=4,
            policy_kwargs={"max_pool_pages": 64},
        )
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(2, 256, 2, 64)
        ctrl = create_kse_controller(cfg, pool, kv, torch.device("cpu"))
        self.assertIsInstance(ctrl, KSEController)
        self.assertIsNone(ctrl.eviction)

    def test_create_streaming_llm_triton_success(self):
        cfg = KSEConfig(
            policy_name="streaming_llm",
            backend_name="triton",
        )
        pool = MockReqToTokenPool(4, 256)
        kv = MockKVCache(2, 256, 2, 64)
        ctrl = create_kse_controller(cfg, pool, kv, torch.device("cpu"))
        self.assertIsInstance(ctrl, KSEController)
        self.assertIsNotNone(ctrl.eviction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
