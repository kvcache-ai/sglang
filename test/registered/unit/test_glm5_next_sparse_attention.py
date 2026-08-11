"""CPU numerical tests for the eager GLM-5-Next sparse MLA fallback."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/glm5_next_sparse_attention.py"
)
BACKEND_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/nsa_backend.py"


def _load_fallback_module():
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_sparse_attention_test", FALLBACK_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGlm5NextSparseAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_fallback_module()

    def test_chunked_h512_attention_matches_float_reference(self):
        torch.manual_seed(20260811)
        query = torch.randn(5, 3, 512, dtype=torch.bfloat16) * 0.1
        kv = torch.randn(2, 4, 512, dtype=torch.bfloat16) * 0.1
        indices = torch.tensor(
            [
                [0, 2, -1, -1],
                [1, 3, 5, -1],
                [7, -1, -1, -1],
                [0, 4, 6, 7],
                [2, 5, -1, -1],
            ],
            dtype=torch.int32,
        )
        scale = 512**-0.5

        actual = self.module.glm5_next_sparse_mla_reference(
            query,
            kv,
            indices,
            sm_scale=scale,
            chunk_size=2,
        )

        flat_kv = kv.reshape(-1, 512).float()
        expected_rows = []
        for row, query_row in zip(indices, query, strict=True):
            selected = flat_kv[row[row >= 0].long()]
            scores = query_row.float() @ selected.T * scale
            expected_rows.append(torch.softmax(scores, dim=-1) @ selected)
        expected = torch.stack(expected_rows).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_padding_values_and_empty_rows_are_ignored(self):
        query = torch.ones(2, 1, 512, dtype=torch.bfloat16)
        kv = (
            torch.arange(6 * 512, dtype=torch.float32)
            .reshape(6, 512)
            .to(torch.bfloat16)
        )
        indices = torch.tensor([[1, 4, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)

        output = self.module.glm5_next_sparse_mla_reference(
            query,
            kv,
            indices,
            sm_scale=0.0,
        )

        expected = (kv[1].float() + kv[4].float()).mul(0.5).to(torch.bfloat16)
        torch.testing.assert_close(output[0, 0], expected)
        self.assertTrue(torch.equal(output[1], torch.zeros_like(output[1])))

    def test_rejects_invalid_layout_and_out_of_range_indices(self):
        query = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
        kv = torch.zeros(2, 512, dtype=torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "int32"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv,
                torch.zeros(1, 1, dtype=torch.int64),
                sm_scale=1.0,
            )
        with self.assertRaisesRegex(IndexError, "exceeds"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv,
                torch.tensor([[2]], dtype=torch.int32),
                sm_scale=1.0,
            )

    def test_shared_backend_dispatch_is_exactly_glm_gated(self):
        tree = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeSparseAttnBackend"
        )
        forward = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_trtllm"
        )
        source = ast.unparse(forward)
        self.assertIn("if self.is_glm5_next", source)
        self.assertIn("glm5_next_sparse_mla_reference", source)
        self.assertIn("self.nsa_index_topk + self.nsa_index_kpool - 1", source)
        self.assertIn("sparse_mla_top_k != expected_top_k", source)


if __name__ == "__main__":
    unittest.main()
