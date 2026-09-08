"""Tests for KSE core data types and configuration."""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)

import torch

from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult


class TestGranularity(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(Granularity.TOKEN.value, "token")
        self.assertEqual(Granularity.PAGE.value, "page")

    def test_members_count(self):
        self.assertEqual(len(Granularity), 2)


class TestFrequency(unittest.TestCase):
    def test_enum_values(self):
        self.assertEqual(Frequency.PER_REQUEST.value, "per_request")
        self.assertEqual(Frequency.PER_STEP.value, "per_step")
        self.assertEqual(Frequency.PER_LAYER.value, "per_layer")

    def test_members_count(self):
        self.assertEqual(len(Frequency), 3)


class TestSelectionResult(unittest.TestCase):
    def test_construction(self):
        result = SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=torch.tensor([[0, 1, -1]], dtype=torch.int32),
            valid_lengths=torch.tensor([2], dtype=torch.int32),
            sparse_mask=torch.tensor([True]),
        )
        self.assertEqual(result.granularity, Granularity.TOKEN)
        self.assertEqual(result.selected_indices.shape, (1, 3))
        self.assertEqual(result.valid_lengths.shape, (1,))
        self.assertEqual(result.sparse_mask.shape, (1,))

    def test_default_layer_ids_none(self):
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.zeros(2, 4, dtype=torch.int32),
            valid_lengths=torch.zeros(2, dtype=torch.int32),
            sparse_mask=torch.ones(2, dtype=torch.bool),
        )
        self.assertIsNone(result.layer_ids)

    def test_layer_ids_set(self):
        result = SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=torch.zeros(1, 1, dtype=torch.int32),
            valid_lengths=torch.zeros(1, dtype=torch.int32),
            sparse_mask=torch.ones(1, dtype=torch.bool),
            layer_ids=[0, 2, 4],
        )
        self.assertEqual(result.layer_ids, [0, 2, 4])


class TestKSEConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KSEConfig(policy_name="quest", backend_name="triton")
        self.assertEqual(cfg.start_layer, 0)
        self.assertEqual(cfg.end_layer, -1)
        self.assertEqual(cfg.min_seq_len, 2048)
        self.assertEqual(cfg.page_size, 64)
        self.assertEqual(cfg.policy_kwargs, {})

    def test_custom_values(self):
        cfg = KSEConfig(
            policy_name="streaming_llm",
            backend_name="flashattention",
            start_layer=2,
            end_layer=10,
            min_seq_len=512,
            page_size=128,
            policy_kwargs={"num_sink_tokens": 8},
        )
        self.assertEqual(cfg.policy_name, "streaming_llm")
        self.assertEqual(cfg.backend_name, "flashattention")
        self.assertEqual(cfg.start_layer, 2)
        self.assertEqual(cfg.end_layer, 10)
        self.assertEqual(cfg.policy_kwargs["num_sink_tokens"], 8)

    def test_policy_kwargs_independent(self):
        cfg1 = KSEConfig(policy_name="a", backend_name="b")
        cfg2 = KSEConfig(policy_name="c", backend_name="d")
        cfg1.policy_kwargs["x"] = 1
        self.assertNotIn("x", cfg2.policy_kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
