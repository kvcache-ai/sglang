"""CPU numerical contracts for the GLM-5-Next SM120 indexer fallback."""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/glm5_next_indexer_logits.py"
)


def _load_module():
    name = "_glm5_next_indexer_logits_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOGITS = _load_module()


def _independent_scores(q, k, scales, weights, ks, ke):
    output = torch.full((q.shape[0], k.shape[0]), float("-inf"))
    for qi in range(q.shape[0]):
        for ki in range(int(ks[qi]), int(ke[qi])):
            per_head = []
            for head in range(q.shape[1]):
                dot = torch.dot(
                    q[qi, head].to(torch.bfloat16),
                    k[ki].to(torch.bfloat16),
                )
                per_head.append(F.relu(dot).float() * weights[qi, head].float())
            output[qi, ki] = torch.stack(per_head).sum() * scales[ki].float()
    return output


def _pack_pages(keys, scales):
    pages = keys.shape[0]
    buf = torch.zeros((pages, LOGITS.GLM5_NEXT_INDEX_PAGE_NBYTES), dtype=torch.uint8)
    key_bytes = keys.contiguous().view(torch.uint8).reshape(pages, -1)
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(pages, -1)
    buf[:, : LOGITS.GLM5_NEXT_INDEX_SCALE_OFFSET] = key_bytes
    buf[:, LOGITS.GLM5_NEXT_INDEX_SCALE_OFFSET :] = scale_bytes
    return buf.view(pages, 64, 1, 132)


class TestGlm5NextIndexerLogits(unittest.TestCase):
    def test_ragged_score_formula_masking_and_chunks(self):
        torch.manual_seed(17)
        num_queries, num_keys, num_heads, dim = 5, 11, 3, 128
        q = (torch.randn(num_queries, num_heads, dim) * 3).to(torch.float8_e4m3fn)
        k = (torch.randn(num_keys, dim) * 2).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.01, 0.11, num_keys)
        weights = torch.tensor(
            [
                [0.5, -0.25, 1.0],
                [-1.0, 0.75, 0.25],
                [0.0, 1.5, -0.5],
                [2.0, -1.0, 0.125],
                [-0.5, -0.25, -0.125],
            ],
            dtype=torch.float32,
        )
        ks = torch.tensor([0, 1, 4, 7, 11], dtype=torch.int32)
        ke = torch.tensor([3, 9, 11, 10, 11], dtype=torch.int32)

        actual = LOGITS.glm5_next_eager_fp8_mqa_logits(
            q,
            (k, scales),
            weights,
            ks,
            ke,
            query_chunk_size=2,
            key_chunk_size=3,
        )
        expected = _independent_scores(q, k, scales, weights, ks, ke)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.isneginf(actual[4]).all())

    def test_ragged_row_iterator_matches_one_shot_with_partial_chunk(self):
        torch.manual_seed(29)
        num_queries, num_keys, num_heads, dim = 7, 17, 3, 128
        q = torch.randn(num_queries, num_heads, dim).to(torch.float8_e4m3fn)
        k = torch.randn(num_keys, dim).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.01, 0.17, num_keys)
        weights = torch.randn(num_queries, num_heads)
        ks = torch.tensor([0, 0, 1, 4, 8, 13, 17], dtype=torch.int32)
        ke = torch.tensor([0, 5, 17, 9, 16, 17, 17], dtype=torch.int32)

        expected = LOGITS.glm5_next_eager_fp8_mqa_logits(
            q,
            (k, scales),
            weights,
            ks,
            ke,
            query_chunk_size=num_queries,
            key_chunk_size=4,
        )
        chunks = list(
            LOGITS.iter_glm5_next_eager_fp8_mqa_logits(
                q,
                (k, scales),
                weights,
                ks,
                ke,
                query_chunk_size=3,
                key_chunk_size=4,
            )
        )

        self.assertEqual(
            [(start, end) for start, end, _ in chunks], [(0, 3), (3, 6), (6, 7)]
        )
        actual = torch.cat([chunk for _, _, chunk in chunks], dim=0)
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_500k_query_row_plan_is_statically_bounded(self):
        pooled_keys = ((500_000 // 4 + 63) // 64) * 64
        ranges = list(
            LOGITS.glm5_next_prefill_query_row_ranges(
                4096,
                query_chunk_size=LOGITS.GLM5_NEXT_PREFILL_QUERY_ROW_CHUNK,
            )
        )

        self.assertEqual(len(ranges), 128)
        self.assertEqual(ranges[0], (0, 32))
        self.assertEqual(ranges[-1], (4064, 4096))
        self.assertEqual(
            [start for start, _ in ranges],
            [0, *[end for _, end in ranges[:-1]]],
        )
        self.assertTrue(all(end - start <= 32 for start, end in ranges))
        self.assertLessEqual(
            max((end - start) * pooled_keys for start, end in ranges), 4_001_792
        )

    def test_paged_layout_page_order_lengths_and_chunks(self):
        torch.manual_seed(23)
        pages, heads, dim = 4, 2, 128
        keys = (torch.randn(pages, 64, dim) * 2).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.005, 0.05, pages * 64).reshape(pages, 64)
        cache = _pack_pages(keys, scales)
        q = (torch.randn(2, 1, heads, dim) * 3).to(torch.float8_e4m3fn)
        weights = torch.tensor([[0.5, -1.25], [-0.75, 2.0]])
        page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
        lengths = torch.tensor([[67], [5]], dtype=torch.int32)

        actual = LOGITS.glm5_next_eager_fp8_paged_mqa_logits(
            q,
            cache,
            weights,
            lengths,
            page_table,
            max_seq_len=128,
            key_chunk_size=64,
        )

        expected = torch.full_like(actual, float("-inf"))
        for row, length in enumerate((67, 5)):
            ordered_keys = torch.cat(
                [keys[int(page)] for page in page_table[row]], dim=0
            )[:length]
            ordered_scales = torch.cat(
                [scales[int(page)] for page in page_table[row]], dim=0
            )[:length]
            local = _independent_scores(
                q[row, 0].unsqueeze(0),
                ordered_keys,
                ordered_scales,
                weights[row].unsqueeze(0),
                torch.tensor([0]),
                torch.tensor([length]),
            )
            expected[row, :length] = local[0]

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(torch.isneginf(actual[0, 67:]).all())
        self.assertTrue(torch.isneginf(actual[1, 5:]).all())

    def test_contract_validation_and_cpu_route_is_disabled(self):
        self.assertFalse(
            LOGITS.use_glm5_next_eager_logits_on_device(torch.device("cpu"))
        )
        q = torch.zeros((1, 1, 64), dtype=torch.float8_e4m3fn)
        k = torch.zeros((1, 64), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "head_dim=128"):
            LOGITS.glm5_next_eager_fp8_mqa_logits(
                q,
                (k, torch.ones(1)),
                torch.ones((1, 1)),
                torch.zeros(1, dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
            )

    def test_paged_decode_source_has_no_host_sync_or_dynamic_python_loop(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        paged = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "glm5_next_eager_fp8_paged_mqa_logits"
        )
        source = ast.unparse(paged)

        for host_sync in (".item()", ".cpu()", ".tolist()"):
            self.assertNotIn(host_sync, source)
        self.assertFalse(
            any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(paged))
        )


if __name__ == "__main__":
    unittest.main()
