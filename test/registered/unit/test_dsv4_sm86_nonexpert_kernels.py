"""Contracts and numerical tests for DeepSeek V4 SM86 decode kernels."""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEXER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/compressed/sm86_indexer.py"
)
SPARSE_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/nsa/v4_triton_kernel.py"
DISPATCH_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/debug_flash_mla_adapter.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    return ast.unparse(function)


def _cuda_bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


class TestDeepSeekV4SM86NonExpertKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = _load_module("_dsv4_sm86_indexer_test", INDEXER_PATH)
        cls.sparse = _load_module("_dsv4_sm86_sparse_test", SPARSE_PATH)

    def test_indexer_is_page_parallel_with_conservative_sm86_tile(self):
        self.assertEqual(self.indexer.DSV4_SM86_INDEXER_BLOCK_K, 32)
        self.assertEqual(self.indexer.DSV4_SM86_INDEXER_NUM_WARPS, 4)
        self.assertEqual(self.indexer.DSV4_SM86_INDEXER_NUM_STAGES, 2)
        source = _function_source(
            INDEXER_PATH, "_bf16_direct_paged_mqa_logits_triton_kernel"
        )
        self.assertIn("key_block = tl.program_id(1)", source)
        self.assertIn("head_scores = tl.dot", source)
        self.assertIn("key_offsets < seq_len", source)
        for host_sync in (".item()", ".cpu()", ".tolist()"):
            self.assertNotIn(host_sync, source)

    def test_sparse_parallelizes_qk_and_value_columns_without_reassociation(self):
        self.assertEqual(self.sparse.DSV4_SM86_SPARSE_BLOCK_H, 8)
        self.assertEqual(self.sparse.DSV4_SM86_SPARSE_BLOCK_N, 16)
        self.assertEqual(self.sparse.DSV4_SM86_SPARSE_BLOCK_D_OUT, 128)
        qk = _function_source(SPARSE_PATH, "_decode_sparse_attention_bf16_qk_kernel")
        value = _function_source(
            SPARSE_PATH, "_decode_sparse_attention_bf16_value_kernel"
        )
        self.assertIn("tile_id = tl.program_id(2)", qk)
        self.assertIn("tl.dot(q, tl.trans", qk)
        self.assertIn("value_block = tl.program_id(2)", value)
        self.assertIn("for start in range(0, loop_end, BLOCK_N)", value)
        self.assertIn("tl.dot(probabilities.to(k.dtype), k)", value)

    def test_dispatch_is_exact_sm86_batch1_and_uses_static_capacity_gate(self):
        source = _function_source(DISPATCH_PATH, "_v4_triton_decode_dispatch")
        self.assertIn("_device_capability() == (8, 6)", source)
        self.assertIn("num_tokens == 1", source)
        self.assertIn("selected_capacity >= 512", source)
        self.assertIn("decode_sparse_attention_bf16_legacy", source)

    @unittest.skipUnless(_cuda_bf16_available(), "an SM80+ CUDA GPU is required")
    def test_page_parallel_indexer_matches_formula_and_replays_graph(self):
        torch.manual_seed(20260901)
        device = torch.device("cuda")
        batch, num_pages, max_seq_len = 2, 6, 192
        query = (torch.randn(batch, 1, 64, 128, device=device) * 0.1).to(torch.bfloat16)
        cache = (torch.randn(num_pages, 64, 128, device=device) * 0.1).to(
            torch.bfloat16
        )
        weights = torch.randn(batch, 64, dtype=torch.float32, device=device)
        page_table = torch.tensor(
            [[2, 0, 4], [5, 1, 3]], dtype=torch.int32, device=device
        )
        seq_lens = torch.tensor([137, 65], dtype=torch.int32, device=device)

        actual = self.indexer.bf16_direct_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
        )
        expected = torch.full_like(actual, -float("inf"))
        for row in range(batch):
            ordered_keys = cache.index_select(
                0, page_table[row].to(torch.int64)
            ).reshape(-1, 128)
            scores = torch.matmul(ordered_keys, query[row, 0].T)
            logits = (torch.relu(scores).float() * weights[row]).sum(dim=1)
            length = int(seq_lens[row])
            expected[row, :length] = logits[:length]

        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=2e-2, atol=2e-2
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self.indexer.bf16_direct_paged_mqa_logits_triton(
                query,
                cache,
                weights,
                seq_lens,
                page_table,
                None,
                max_seq_len,
                False,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)

    @unittest.skipUnless(_cuda_bf16_available(), "an SM80+ CUDA GPU is required")
    def test_parallel_sparse_matches_legacy_oracle_and_replays_graph(self):
        torch.manual_seed(20260902)
        device = torch.device("cuda")
        query = (torch.randn(1, 64, 512, device=device) * 0.05).to(torch.bfloat16)
        swa_cache = (torch.randn(1, 128, 512, device=device) * 0.05).to(torch.bfloat16)
        extra_cache = (torch.randn(8, 64, 512, device=device) * 0.05).to(torch.bfloat16)
        swa_indices = torch.arange(128, dtype=torch.int32, device=device).unsqueeze(0)
        extra_indices = torch.arange(512, dtype=torch.int32, device=device).unsqueeze(0)
        swa_lens = torch.tensor([127], dtype=torch.int32, device=device)
        extra_lens = torch.tensor([511], dtype=torch.int32, device=device)
        sink = torch.randn(64, dtype=torch.float32, device=device) * 0.1
        scale = 1.0 / math.sqrt(512)
        actual = torch.empty_like(query)
        self.sparse.decode_sparse_attention_bf16(
            query,
            swa_cache,
            swa_indices,
            swa_lens,
            scale,
            sink,
            actual,
            extra_cache,
            extra_indices,
            extra_lens,
        )
        legacy = torch.empty_like(query)
        self.sparse.decode_sparse_attention_bf16_legacy(
            query,
            swa_cache,
            swa_indices,
            swa_lens,
            scale,
            sink,
            legacy,
            extra_cache,
            extra_indices,
            extra_lens,
        )
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, legacy, rtol=0, atol=1e-3)

        selected = torch.cat(
            (
                extra_cache.reshape(-1, 512)[:511],
                swa_cache.reshape(-1, 512)[:127],
            )
        )
        scores = torch.einsum("hd,nd->hn", query[0].float(), selected.float()) * scale
        probabilities = torch.softmax(torch.cat((sink[:, None], scores), dim=1), dim=1)[
            :, 1:
        ]
        expected = torch.matmul(probabilities, selected.float()).to(torch.bfloat16)
        torch.testing.assert_close(actual[0], expected, rtol=2e-2, atol=2e-2)

        captured = torch.empty_like(query)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.sparse.decode_sparse_attention_bf16(
                query,
                swa_cache,
                swa_indices,
                swa_lens,
                scale,
                sink,
                captured,
                extra_cache,
                extra_indices,
                extra_lens,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
