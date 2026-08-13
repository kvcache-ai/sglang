"""Contracts for GLM-5-Next eager KPool writes spanning multiple pools.

This intentionally does not cover CP or speculative decoding.  Source modules
are loaded directly so integer contracts remain runnable in a kernel-only
environment, while SM120-only tests exercise real CUDA scoring and selection.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
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
    indexer_logits = _load_module(
        "_glm5_next_indexer_logits_under_test",
        NSA_DIR / "glm5_next_indexer_logits.py",
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

    utils = types.ModuleType("sglang.srt.utils")
    utils.is_cuda = lambda: False
    stubs[utils.__name__] = utils
    stubs["sglang.srt.layers.attention.nsa.kpool_fp8_index"] = fp8_index

    with patch.dict(sys.modules, stubs):
        plan = _load_module(
            "_glm5_next_kpool_plan_under_test",
            NSA_DIR / "kpool_plan.py",
        )
    return fp8_index, plan, indexer_logits


KPOOL_FP8_INDEX, KPOOL_PLAN, INDEXER_LOGITS = _load_kpool_sources()


def _compile_indexer_method(function_name: str, globals_: dict):
    path = NSA_DIR / "nsa_indexer_kpool.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    indexer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IndexerKPool"
    )
    function = next(
        node
        for node in indexer.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function.decorator_list = []
    function.body = [
        node
        for node in function.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


@dataclass(frozen=True)
class _PooledMetadata:
    real_page_table: torch.Tensor
    pooled_index_kpool: int = 1
    pooled_cache_seqlens_int32: torch.Tensor | None = None
    pooled_real_page_table: torch.Tensor | None = None
    pooled_paged_mqa_schedule_metadata: torch.Tensor | None = None


class TestDsaKpoolMultiPool(unittest.TestCase):
    POOL_SIZE = 4
    SLOTS_PER_PAGE = 64

    def test_layout_gate_is_glm5_next_specific(self):
        self.assertTrue(KPOOL_PLAN._is_kpool_layout_enabled(4, 64))
        self.assertFalse(KPOOL_PLAN._is_kpool_layout_enabled(2, 64))
        self.assertFalse(KPOOL_PLAN._is_kpool_layout_enabled(4, 1))

    def test_metadata_uses_explicit_forward_batch_request_pool(self):
        table = object()
        batch = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_to_token=table)
        )
        self.assertIs(KPOOL_PLAN._req_to_token_table_from_batch(batch), table)

        for incomplete in (
            types.SimpleNamespace(),
            types.SimpleNamespace(req_to_token_pool=None),
            types.SimpleNamespace(
                req_to_token_pool=types.SimpleNamespace(req_to_token=None)
            ),
        ):
            with self.subTest(incomplete=incomplete):
                with self.assertRaisesRegex(RuntimeError, "KPool metadata requires"):
                    KPOOL_PLAN._req_to_token_table_from_batch(incomplete)

    def test_production_topk_source_has_no_host_sync(self):
        tree = ast.parse((NSA_DIR / "kpool_fp8_index.py").read_text(encoding="utf-8"))
        functions = {
            node.name: ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_exact_topk_from_pooled_history_logits",
                "topk_from_pooled_history_logits",
            }
        }
        self.assertEqual(len(functions), 2)
        for source in functions.values():
            for host_sync in (".item()", ".cpu()", ".tolist()"):
                self.assertNotIn(host_sync, source)

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

    def test_hierarchical_exact_topk_keeps_late_unique_max(self):
        num_groups = KPOOL_FP8_INDEX.KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE + 904
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

    def test_hierarchical_topk_short_rows_keep_only_finite_candidates(self):
        scores = torch.full((1, 32), -9.0, dtype=torch.float32)
        scores[0, 7] = 3.0
        result = KPOOL_FP8_INDEX._exact_topk_from_pooled_history_logits(
            scores,
            torch.tensor([1], dtype=torch.int32),
            pool_size=self.POOL_SIZE,
            topk=8,
            row_starts=torch.tensor([7], dtype=torch.int32),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_hierarchical_topk_row_starts_and_page_row_mapping(self):
        scores = torch.zeros((2, 12), dtype=torch.float32)
        scores[0, 5] = 4.0
        scores[1, 8] = 5.0
        page_table = torch.stack(
            [
                torch.arange(100, 112, dtype=torch.int32),
                torch.arange(200, 212, dtype=torch.int32),
            ]
        )
        result = KPOOL_FP8_INDEX._exact_topk_from_pooled_history_logits(
            scores,
            torch.tensor([3, 3], dtype=torch.int32),
            pool_size=self.POOL_SIZE,
            topk=4,
            row_starts=torch.tensor([4, 8], dtype=torch.int32),
            page_table=page_table,
            page_table_row_index=torch.tensor([1, 0], dtype=torch.int32),
        )

        torch.testing.assert_close(
            result,
            torch.tensor(
                [[204, 205, 206, 207], [100, 101, 102, 103]],
                dtype=torch.int32,
            ),
            rtol=0,
            atol=0,
        )

    def test_graph_metadata_replay_updates_in_place(self):
        forward_mode = types.SimpleNamespace(is_decode_or_idle=lambda: True)
        seqlens = torch.tensor([7, 260], dtype=torch.int32)
        metadata = _PooledMetadata(
            real_page_table=torch.arange(16, dtype=torch.int32).reshape(2, 8)
        )

        with patch.object(KPOOL_PLAN, "is_cuda", return_value=True):
            metadata = KPOOL_PLAN.init_pooled_paged_mqa_metadata(
                metadata,
                seqlens,
                forward_mode,
                pool_size=self.POOL_SIZE,
                real_page_size=self.SLOTS_PER_PAGE,
                slots_per_page=self.SLOTS_PER_PAGE,
                build_schedule_metadata=False,
            )

            pooled_lens_ptr = metadata.pooled_cache_seqlens_int32.data_ptr()
            pooled_table_ptr = metadata.pooled_real_page_table.data_ptr()
            metadata.real_page_table.add_(100)
            seqlens.copy_(torch.tensor([15, 256], dtype=torch.int32))
            KPOOL_PLAN.update_pooled_paged_mqa_metadata(
                metadata,
                seqlens,
                forward_mode,
                pool_size=self.POOL_SIZE,
                real_page_size=self.SLOTS_PER_PAGE,
                slots_per_page=self.SLOTS_PER_PAGE,
                build_schedule_metadata=False,
            )

        self.assertEqual(
            metadata.pooled_cache_seqlens_int32.data_ptr(), pooled_lens_ptr
        )
        self.assertEqual(metadata.pooled_real_page_table.data_ptr(), pooled_table_ptr)
        torch.testing.assert_close(
            metadata.pooled_cache_seqlens_int32,
            torch.tensor([3, 64], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            metadata.pooled_real_page_table,
            torch.tensor([[100, 104], [108, 112]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_production_topk_is_exact_across_hierarchical_chunks(self):
        for num_groups in (
            KPOOL_FP8_INDEX.KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE + 904,
            60000,
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_hierarchical_topk_cuda_graph_replay_uses_device_lengths(self):
        num_groups = KPOOL_FP8_INDEX.KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE + 4096
        scores = torch.zeros((1, num_groups), dtype=torch.float32, device="cuda")
        group_lengths = torch.tensor([35000], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([140001], dtype=torch.int32, device="cuda")
        scores[0, 34000] = 7.0

        def select():
            return KPOOL_FP8_INDEX.topk_from_pooled_history_logits(
                scores,
                group_lengths,
                pool_size=self.POOL_SIZE,
                topk=2048,
                seq_lens=seq_lens,
            )

        # Warm up both torch.topk shards before capture.
        select()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = select()

        graph.replay()
        torch.cuda.synchronize()
        initial_tokens = torch.arange(34000 * 4, 34000 * 4 + 4, device="cuda")
        for token in initial_tokens:
            self.assertTrue(torch.any(result[0, :2048] == token))

        scores.zero_()
        scores[0, -1] = 9.0
        group_lengths.fill_(num_groups)
        seq_lens.fill_(num_groups * self.POOL_SIZE + 1)
        graph.replay()
        torch.cuda.synchronize()
        replay_tokens = torch.arange(
            (num_groups - 1) * 4,
            num_groups * 4,
            device="cuda",
        )
        for token in replay_tokens:
            self.assertTrue(torch.any(result[0, :2048] == token))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
        "SM120 CUDA is required",
    )
    def test_sm120_streamed_prefill_matches_monolithic_ragged_and_paged(self):
        torch.manual_seed(20260812)
        device = torch.device("cuda")
        rows, keys, heads, dim = 35, 320, 4, 128
        q = (torch.randn(rows, heads, dim, device=device) * 0.25).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(keys, dim, device=device) * 0.25).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.01, 0.05, keys, device=device)
        weights = torch.randn(rows, heads, device=device)
        starts = (torch.arange(rows, device=device, dtype=torch.int32) % 3) * 5
        pool_lens = torch.full((rows,), keys, device=device, dtype=torch.int32) - starts
        ends = starts + pool_lens
        seq_lens = pool_lens * self.POOL_SIZE + (
            torch.arange(rows, device=device, dtype=torch.int32) % self.POOL_SIZE
        )

        def select(logits, lengths, **kwargs):
            return KPOOL_FP8_INDEX.topk_from_pooled_history_logits(
                logits,
                lengths,
                pool_size=self.POOL_SIZE,
                topk=512,
                **kwargs,
            )

        streamed = _compile_indexer_method(
            "_topk_from_glm5_next_eager_logits_rows",
            {
                "torch": torch,
                "iter_glm5_next_eager_fp8_mqa_logits": (
                    INDEXER_LOGITS.iter_glm5_next_eager_fp8_mqa_logits
                ),
            },
        )
        runner = types.SimpleNamespace(
            index_topk=512,
            index_kpool=self.POOL_SIZE,
            _topk_from_kpool_logits=select,
        )
        logits = INDEXER_LOGITS.glm5_next_eager_fp8_mqa_logits(
            q,
            (k, scales),
            weights,
            starts,
            ends,
            query_chunk_size=rows,
            key_chunk_size=97,
        )

        topk_offsets = torch.arange(rows, device=device, dtype=torch.int32) * 2048
        expected_ragged = select(
            logits,
            pool_lens,
            seq_lens=seq_lens,
            topk_offsets=topk_offsets,
            row_starts=starts,
            out_rows=rows + 2,
        )
        actual_ragged = streamed(
            runner,
            q,
            (k, scales),
            weights,
            pool_lens,
            seq_lens,
            starts,
            ends,
            total_q=rows + 2,
            page_table=None,
            topk_offsets=topk_offsets,
            page_table_row_index=None,
        )
        torch.testing.assert_close(actual_ragged, expected_ragged, rtol=0, atol=0)

        page_table = torch.arange(
            41 * keys * self.POOL_SIZE,
            device=device,
            dtype=torch.int32,
        ).reshape(41, keys * self.POOL_SIZE)
        page_rows = (torch.arange(rows, device=device, dtype=torch.int64) * 7) % 41
        expected_paged = select(
            logits,
            pool_lens,
            seq_lens=seq_lens,
            page_table=page_table,
            row_starts=starts,
            out_rows=rows + 2,
            page_table_row_index=page_rows,
        )
        actual_paged = streamed(
            runner,
            q,
            (k, scales),
            weights,
            pool_lens,
            seq_lens,
            starts,
            ends,
            total_q=rows + 2,
            page_table=page_table,
            topk_offsets=None,
            page_table_row_index=page_rows,
        )
        torch.testing.assert_close(actual_paged, expected_paged, rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
        "SM120 CUDA is required",
    )
    def test_sm120_500k_pooled_width_prefill_temporary_peak_is_bounded(self):
        torch.manual_seed(20260813)
        device = torch.device("cuda")
        rows, keys, heads, dim = 32, 125_056, 32, 128
        q = (torch.randn(rows, heads, dim, device=device) * 0.125).to(
            torch.float8_e4m3fn
        )
        k = (torch.randn(keys, dim, device=device) * 0.125).to(torch.float8_e4m3fn)
        scales = torch.full((keys,), 0.01, device=device)
        weights = torch.randn(rows, heads, device=device)
        starts = torch.zeros(rows, device=device, dtype=torch.int32)
        ends = torch.full((rows,), keys, device=device, dtype=torch.int32)
        pool_lens = ends.clone()
        seq_lens = pool_lens * self.POOL_SIZE

        def select(logits, lengths, **kwargs):
            return KPOOL_FP8_INDEX.topk_from_pooled_history_logits(
                logits,
                lengths,
                pool_size=self.POOL_SIZE,
                topk=2048,
                **kwargs,
            )

        streamed = _compile_indexer_method(
            "_topk_from_glm5_next_eager_logits_rows",
            {
                "torch": torch,
                "iter_glm5_next_eager_fp8_mqa_logits": (
                    INDEXER_LOGITS.iter_glm5_next_eager_fp8_mqa_logits
                ),
            },
        )
        runner = types.SimpleNamespace(
            index_topk=2048,
            index_kpool=self.POOL_SIZE,
            _topk_from_kpool_logits=select,
        )

        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = streamed(
            runner,
            q,
            (k, scales),
            weights,
            pool_lens,
            seq_lens,
            starts,
            ends,
            total_q=rows,
            page_table=None,
            topk_offsets=torch.zeros(rows, device=device, dtype=torch.int32),
            page_table_row_index=None,
        )
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline

        self.assertEqual(tuple(result.shape), (rows, 2051))
        # Exact top-k retains one 128-group candidate pair per 32768-key shard;
        # the bounded row chunk still stays far below the old multi-GiB QxK
        # allocation.  Keep headroom for allocator/workspace changes.
        self.assertLessEqual(peak_delta, 768 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
