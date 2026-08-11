"""CPU contracts for GLM-5-Next eager KPool writes spanning multiple pools.

This intentionally does not cover CP, speculative decoding, or CUDA graphs.
The source modules are loaded directly so the integer contracts remain runnable
in a kernel-only environment without importing the full ``sglang`` package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
NSA_DIR = REPO_ROOT / "python/sglang/srt/layers/attention/nsa"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_kpool_sources():
    fp8_index = _load_module(
        "_glm5_next_kpool_fp8_index_under_test",
        NSA_DIR / "kpool_fp8_index.py",
    )

    stubs = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.nsa",
        "sglang.srt.model_executor",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        stubs[name] = package

    environ = types.ModuleType("sglang.srt.environ")
    environ.envs = types.SimpleNamespace(
        SGLANG_NSA_FUSE_TOPK=types.SimpleNamespace(get=lambda: False)
    )
    stubs[environ.__name__] = environ

    forward_context = types.ModuleType("sglang.srt.model_executor.forward_context")
    forward_context.get_req_to_token_pool = lambda: None
    stubs[forward_context.__name__] = forward_context

    utils = types.ModuleType("sglang.srt.utils")
    utils.is_cuda = lambda: False
    stubs[utils.__name__] = utils
    stubs["sglang.srt.layers.attention.nsa.kpool_fp8_index"] = fp8_index

    with patch.dict(sys.modules, stubs):
        plan = _load_module(
            "_glm5_next_kpool_plan_under_test",
            NSA_DIR / "kpool_plan.py",
        )
    return fp8_index, plan


KPOOL_FP8_INDEX, KPOOL_PLAN = _load_kpool_sources()


class TestDsaKpoolMultiPool(unittest.TestCase):
    POOL_SIZE = 4
    SLOTS_PER_PAGE = 64

    def test_layout_gate_is_glm5_next_specific(self):
        self.assertTrue(KPOOL_PLAN._is_kpool_layout_enabled(4, 64))
        self.assertFalse(KPOOL_PLAN._is_kpool_layout_enabled(2, 64))
        self.assertFalse(KPOOL_PLAN._is_kpool_layout_enabled(4, 1))

    def test_exact_integer_decomposition_across_boundaries(self):
        cases = {
            (0, 1): (0, 0, 0, 1),
            (3, 1): (3, 0, 1, 0),
            (3, 6): (3, 0, 2, 1),
            (4, 4): (0, 1, 1, 0),
            (255, 6): (3, 63, 2, 1),
        }
        for (start, length), expected in cases.items():
            with self.subTest(start=start, length=length):
                actual = KPOOL_PLAN._decompose_compress(start, length, self.POOL_SIZE)
                self.assertEqual(tuple(actual), expected)

    def test_integer_decomposition_exhaustive_near_page_boundaries(self):
        for start in range(2 * self.SLOTS_PER_PAGE * self.POOL_SIZE):
            for length in range(1, self.SLOTS_PER_PAGE + 1):
                actual = KPOOL_PLAN._decompose_compress(start, length, self.POOL_SIZE)
                expected_closed = (
                    start + length
                ) // self.POOL_SIZE - start // self.POOL_SIZE
                consumed = max(
                    0,
                    actual.n_pool * self.POOL_SIZE - actual.first_slot,
                )
                self.assertEqual(actual.first_slot, start % self.POOL_SIZE)
                self.assertEqual(actual.base_pool, start // self.POOL_SIZE)
                self.assertEqual(actual.n_pool, expected_closed)
                self.assertEqual(actual.tail_n_write, length - consumed)
                self.assertLess(actual.tail_n_write, self.POOL_SIZE)

    def test_eager_plan_records_every_closed_pool_and_tail(self):
        plan = KPOOL_PLAN._KPoolCpuPlan()
        KPOOL_PLAN._append_compress_rows(
            plan,
            pool_size=self.POOL_SIZE,
            batch_size=2,
            extend_seq_lens_cpu=[6, 6],
            seq_lens_cpu=[9, 261],
            req_pool_indices_cpu=[7, 11],
        )

        self.assertEqual(plan.pool_batch_idx, [0, 0, 1, 1])
        self.assertEqual(plan.pool_req, [7, 7, 11, 11])
        self.assertEqual(plan.pool_pool_id, [0, 1, 63, 64])
        self.assertEqual(plan.pool_n_from_tail, [3, 0, 3, 0])
        self.assertEqual(plan.pool_chunk_src, [0, 1, 6, 7])
        self.assertEqual(plan.pool_tail_logical_base, [0, 4, 252, 256])
        self.assertEqual(plan.tail_req, [7, 11])
        self.assertEqual(plan.tail_dst_logical_start, [8, 260])
        self.assertEqual(plan.tail_chunk_src, [5, 11])
        self.assertEqual(plan.tail_n_write, [1, 1])

    def test_ragged_page_plan_uses_exact_integer_counts(self):
        plan = KPOOL_PLAN._KPoolCpuPlan()
        KPOOL_PLAN._append_local_rows(
            plan,
            pool_size=self.POOL_SIZE,
            slots_per_page=self.SLOTS_PER_PAGE,
            local_extend_seq_lens_cpu=[6, 6],
            local_seq_lens_cpu=[9, 261],
        )

        self.assertEqual(plan.ragged_q_len, [6, 6])
        self.assertEqual(plan.ragged_pool_pages, [1, 2])
        self.assertEqual(plan.cu_pages_excl, [0, 1])
        self.assertEqual(plan.cu_q_len_excl, [0, 6])
        self.assertEqual(plan.total_pool_pages, 3)

    def test_pooled_page_table_and_physical_locations(self):
        page_table = torch.arange(16, dtype=torch.int32).reshape(2, 8)
        pooled = KPOOL_FP8_INDEX.build_pooled_page_table_64(page_table, self.POOL_SIZE)
        torch.testing.assert_close(
            pooled,
            torch.tensor([[0, 4], [8, 12]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        self.assertTrue(pooled.is_contiguous())
        self.assertEqual(pooled.stride(-1), 1)

        token_pages = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
        locs = KPOOL_FP8_INDEX.compute_pooled_write_locs(
            token_pages,
            torch.tensor([0, 63, 64], dtype=torch.int64),
            self.POOL_SIZE,
        )
        torch.testing.assert_close(
            locs,
            torch.tensor([128, 191, 384], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_pool_groups_expand_to_strict_token_width(self):
        expanded = KPOOL_FP8_INDEX.expand_pooled_groups_to_topk(
            group_ids=torch.tensor([[2, 0]], dtype=torch.int64),
            group_valid=torch.tensor([[True, False]]),
            topk=8,
            pool_size=self.POOL_SIZE,
        )
        torch.testing.assert_close(
            expanded,
            torch.tensor([[8, 9, 10, 11, -1, -1, -1, -1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_exact_overflow_fallback_keeps_late_unique_max_and_tail(self):
        num_groups = KPOOL_FP8_INDEX.KPOOL_RADIX_EXACT_ROW_CAPACITY + 904
        scores = torch.ones((1, num_groups), dtype=torch.float32)
        scores[0, -1] = 1.05
        result = KPOOL_FP8_INDEX._exact_topk_from_pooled_history_logits(
            scores,
            torch.tensor([num_groups], dtype=torch.int32),
            pool_size=self.POOL_SIZE,
            topk=2048,
        )

        unique_max_tokens = torch.arange(
            (num_groups - 1) * self.POOL_SIZE,
            num_groups * self.POOL_SIZE,
            dtype=torch.int32,
        )
        for token in unique_max_tokens:
            self.assertTrue(torch.any(result[0, :2048] == token))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_production_topk_routes_overflow_to_exact_fallback(self):
        for num_groups in (
            KPOOL_FP8_INDEX.KPOOL_RADIX_EXACT_ROW_CAPACITY + 904,
            6000,
            125000,
        ):
            with self.subTest(num_groups=num_groups):
                scores = torch.ones((1, num_groups), dtype=torch.float32, device="cuda")
                scores[0, -1] = 1.05
                seq_len = num_groups * self.POOL_SIZE + 1

                result = KPOOL_FP8_INDEX.topk_from_pooled_history_logits(
                    scores,
                    torch.tensor([num_groups], dtype=torch.int32, device="cuda"),
                    pool_size=self.POOL_SIZE,
                    topk=2048,
                    seq_lens=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
                )

                last_group_start = (num_groups - 1) * self.POOL_SIZE
                expected = torch.arange(
                    last_group_start,
                    last_group_start + self.POOL_SIZE,
                    dtype=torch.int32,
                    device="cuda",
                )
                for token in expected:
                    self.assertTrue(torch.any(result[0] == token))
                self.assertEqual(tuple(result.shape), (1, 2051))
                self.assertEqual(int(result[0, 2048]), seq_len - 1)
                self.assertTrue(torch.all(result[0, 2049:] == -1))


if __name__ == "__main__":
    unittest.main()
