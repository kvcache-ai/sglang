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
SCALED_FP8_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/glm5_next_scaled_fp8.py"
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


def _load_scaled_fp8_module():
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_scaled_fp8_test", SCALED_FP8_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestGlm5NextSparseAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_fallback_module()
        cls.scaled_fp8 = _load_scaled_fp8_module()

    def test_block_scaled_fp8_reduces_latent_quantization_error(self):
        torch.manual_seed(20260813)
        blocks = [torch.randn(6, 128) * 1.0e-4 for _ in range(4)]
        latent = torch.cat(blocks, dim=-1).to(torch.bfloat16)

        raw = latent.to(torch.float8_e4m3fn).to(torch.bfloat16)
        quantized, scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(latent)
        restored = self.scaled_fp8.glm5_next_dequantize_latent_fp8(quantized, scale)

        raw_error = (raw.float() - latent.float()).square().mean()
        scaled_error = (restored.float() - latent.float()).square().mean()
        self.assertEqual(scale.shape, (6, 4))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertLess(scaled_error.item(), raw_error.item() * 0.2)

    def test_exact_glm_preparation_keeps_ephemeral_query_in_bf16(self):
        torch.manual_seed(20260816)
        query = torch.randn(2, 3, 512, dtype=torch.bfloat16)
        key = torch.randn(2, 512, dtype=torch.bfloat16)
        query_rope = torch.empty(2, 3, 0, dtype=torch.bfloat16)
        key_rope = torch.empty(2, 0, dtype=torch.bfloat16)

        prepared_q, key_fp8, key_rope_fp8, key_scale = (
            self.scaled_fp8.glm5_next_mla_prepare_scaled_fp8_no_rope(
                query, query_rope, key, key_rope
            )
        )

        self.assertEqual(prepared_q.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(prepared_q, query))
        self.assertEqual(key_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(key_rope_fp8.shape, (2, 0))
        self.assertEqual(key_scale.shape, (2, 4))

    def test_scaled_fp8_sparse_attention_tracks_bf16_oracle(self):
        torch.manual_seed(20260814)
        query_bf16 = (torch.randn(3, 2, 512) * 1.0e-4).to(torch.bfloat16)
        kv_bf16 = (torch.randn(12, 512) * 1.0e-4).to(torch.bfloat16)
        indices = torch.tensor(
            [[0, 2, 7, -1], [1, 4, 8, 11], [3, 5, 9, -1]],
            dtype=torch.int32,
        )
        sm_scale = 512**-0.5

        kv_fp8, kv_scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(kv_bf16)
        actual = self.module.glm5_next_sparse_mla_reference(
            query_bf16,
            kv_fp8,
            indices,
            sm_scale=sm_scale,
            kv_scale=kv_scale,
        )

        expected_rows = []
        for row, query_row in zip(indices, query_bf16, strict=True):
            selected = kv_bf16[row[row >= 0].long()].float()
            scores = query_row.float() @ selected.T * sm_scale
            expected_rows.append(torch.softmax(scores, dim=-1) @ selected)
        expected = torch.stack(expected_rows).to(torch.bfloat16)
        raw = self.module.glm5_next_sparse_mla_reference(
            query_bf16.to(torch.float8_e4m3fn),
            kv_bf16.to(torch.float8_e4m3fn),
            indices,
            sm_scale=sm_scale,
        )
        scaled_error = (actual.float() - expected.float()).abs().mean()
        raw_error = (raw.float() - expected.float()).abs().mean()
        self.assertLess(scaled_error.item(), raw_error.item() * 0.1)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-5)

    def test_bf16_query_reduces_typical_attention_error_vs_raw_fp8(self):
        torch.manual_seed(44)
        query = torch.randn(3, 2, 512, dtype=torch.bfloat16)
        kv = torch.randn(12, 512, dtype=torch.bfloat16)
        indices = torch.tensor([[0, 2, 7], [1, 4, 8], [3, 5, 9]], dtype=torch.int32)
        sm_scale = 256**-0.5
        kv_fp8, kv_scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(kv)

        expected = self.module.glm5_next_sparse_mla_reference(
            query, kv, indices, sm_scale=sm_scale
        )
        scaled_kv = self.module.glm5_next_sparse_mla_reference(
            query,
            kv_fp8,
            indices,
            sm_scale=sm_scale,
            kv_scale=kv_scale,
        )
        raw_qkv = self.module.glm5_next_sparse_mla_reference(
            query.to(torch.float8_e4m3fn),
            kv.to(torch.float8_e4m3fn),
            indices,
            sm_scale=sm_scale,
        )

        scaled_error = (scaled_kv.float() - expected.float()).norm()
        raw_error = (raw_qkv.float() - expected.float()).norm()
        self.assertLess(scaled_error.item(), raw_error.item() * 0.95)

    def test_prefill_uses_bf16_current_rows_and_scaled_fp8_history(self):
        torch.manual_seed(20260817)
        query = torch.randn(3, 2, 512, dtype=torch.bfloat16)
        kv_bf16 = torch.randn(10, 512, dtype=torch.bfloat16)
        kv_fp8, kv_scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(kv_bf16)
        # Deliberately unsorted physical locations exercise allocator reuse.
        current_locs = torch.tensor([7, 2, 9], dtype=torch.int64)
        current_kv = kv_bf16[current_locs].clone()
        indices = torch.tensor(
            [[0, 2, 7, -1], [1, 4, 9, -1], [2, 5, 7, 9]], dtype=torch.int32
        )
        scale = 256**-0.5

        actual = self.module.glm5_next_sparse_mla_reference(
            query,
            kv_fp8,
            indices,
            sm_scale=scale,
            use_cuda_decode_kernel=False,
            kv_scale=kv_scale,
            current_chunk_kv=current_kv,
            current_chunk_locs=current_locs,
        )

        reconstructed = self.scaled_fp8.glm5_next_dequantize_latent_fp8(
            kv_fp8, kv_scale
        )
        mixed = reconstructed.clone()
        mixed[current_locs] = current_kv
        expected = self.module.glm5_next_sparse_mla_reference(
            query,
            mixed,
            indices,
            sm_scale=scale,
            use_cuda_decode_kernel=False,
        )
        cache_only = self.module.glm5_next_sparse_mla_reference(
            query,
            kv_fp8,
            indices,
            sm_scale=scale,
            use_cuda_decode_kernel=False,
            kv_scale=kv_scale,
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, cache_only))

    def test_current_chunk_overlay_is_prefill_only_and_fail_closed(self):
        query = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
        kv_bf16 = torch.zeros(2, 512, dtype=torch.bfloat16)
        kv_fp8, kv_scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(kv_bf16)
        indices = torch.tensor([[0]], dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "provided together"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv_fp8,
                indices,
                sm_scale=1.0,
                use_cuda_decode_kernel=False,
                kv_scale=kv_scale,
                current_chunk_kv=kv_bf16[:1],
            )
        with self.assertRaisesRegex(ValueError, "valid only for EXTEND"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv_fp8,
                indices,
                sm_scale=1.0,
                use_cuda_decode_kernel=True,
                kv_scale=kv_scale,
                current_chunk_kv=kv_bf16[:1],
                current_chunk_locs=torch.tensor([0]),
            )
        with self.assertRaisesRegex(ValueError, "scaled-FP8 cache"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv_bf16,
                indices,
                sm_scale=1.0,
                use_cuda_decode_kernel=False,
                current_chunk_kv=kv_bf16[:1],
                current_chunk_locs=torch.tensor([0]),
            )

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
        with self.assertRaisesRegex(ValueError, "KV scale"):
            self.module.glm5_next_sparse_mla_reference(
                query,
                kv,
                torch.tensor([[0]], dtype=torch.int32),
                sm_scale=1.0,
                kv_scale=torch.ones(1, 3),
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
        self.assertIn(
            "self.is_glm5_next and self.device_capability in _GLM5_NEXT_NATIVE_CUDA_CAPS",
            source,
        )
        self.assertIn(
            "self.device_capability in _GLM5_NEXT_SCALED_FP8_CUDA_CAPS", source
        )
        self.assertIn("glm5_next_sparse_mla_reference", source)
        self.assertIn("use_cuda_decode_kernel=not is_prefill", source)
        self.assertIn("forward_batch.forward_mode.is_extend_without_speculative()", source)
        self.assertIn("current_chunk_kv=glm5_current_chunk_kv", source)
        self.assertIn("current_chunk_locs=glm5_current_chunk_locs", source)
        self.assertIn("self.nsa_index_topk + self.nsa_index_kpool - 1", source)
        self.assertIn("sparse_mla_top_k != expected_top_k", source)
        self.assertLess(
            source.index("glm5_next_sparse_mla_reference"),
            source.index("padded_sparse_mla_top_k"),
        )

    def test_cuda_decode_kernel_is_an_explicit_dispatch_choice(self):
        tree = ast.parse(FALLBACK_PATH.read_text(encoding="utf-8"))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "glm5_next_sparse_mla_reference"
        )
        source = ast.unparse(helper)
        self.assertIn("query.is_cuda and use_cuda_decode_kernel", source)
        self.assertIn("use_cuda_decode_kernel: bool=True", source)

    def test_cuda_launcher_source_has_no_host_sync(self):
        tree = ast.parse(FALLBACK_PATH.read_text(encoding="utf-8"))
        launcher = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_glm5_next_sparse_mla_cuda"
        )
        source = ast.unparse(launcher)
        for host_sync in (".item()", ".cpu()", ".tolist()"):
            self.assertNotIn(host_sync, source)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_raw_fp8_cuda_kernel_matches_cpu_oracle_and_replays_graph(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("this legacy route requires native FP8")
        if torch.cuda.get_device_capability() == (8, 9):
            self.skipTest("consumer profiles use BF16 or scaled FP8 cache")
        torch.manual_seed(20260812)
        query_cpu = (torch.randn(1, 2, 512) * 0.1).to(torch.float8_e4m3fn)
        kv_cpu = (torch.randn(16, 512) * 0.1).to(torch.float8_e4m3fn)
        first_indices_cpu = torch.tensor(
            [[0, 3, 5, 9, -1, -1, -1, -1]], dtype=torch.int32
        )
        second_indices_cpu = torch.tensor(
            [[1, 2, 7, 12, 15, -1, -1, -1]], dtype=torch.int32
        )
        scale = 512**-0.5

        first_expected = self.module.glm5_next_sparse_mla_reference(
            query_cpu,
            kv_cpu,
            first_indices_cpu,
            sm_scale=scale,
        )
        second_expected = self.module.glm5_next_sparse_mla_reference(
            query_cpu,
            kv_cpu,
            second_indices_cpu,
            sm_scale=scale,
        )

        query = query_cpu.cuda()
        kv = kv_cpu.cuda()
        indices = first_indices_cpu.cuda()
        # Warm up Triton compilation before capture.
        self.module.glm5_next_sparse_mla_reference(query, kv, indices, sm_scale=scale)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.module.glm5_next_sparse_mla_reference(
                query, kv, indices, sm_scale=scale
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), first_expected, rtol=3e-2, atol=3e-3)

        indices.copy_(second_indices_cpu)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), second_expected, rtol=3e-2, atol=3e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_scaled_fp8_cuda_kernel_matches_oracle_and_replays_graph(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("pre-SM89 GPUs use an unscaled BF16 latent cache")
        torch.manual_seed(20260815)
        query_bf16 = (torch.randn(1, 2, 512) * 0.02).to(torch.bfloat16)
        kv_bf16 = (torch.randn(16, 512) * 0.02).to(torch.bfloat16)
        kv_fp8, kv_scale = self.scaled_fp8.glm5_next_quantize_latent_fp8(kv_bf16)
        indices_cpu = torch.tensor([[0, 3, 5, 9, 14, -1, -1, -1]], dtype=torch.int32)
        scale = 512**-0.5
        expected = self.module.glm5_next_sparse_mla_reference(
            query_bf16,
            kv_fp8,
            indices_cpu,
            sm_scale=scale,
            kv_scale=kv_scale,
        )

        query = query_bf16.cuda()
        kv = kv_fp8.cuda()
        indices = indices_cpu.cuda()
        kv_scale_cuda = kv_scale.cuda()
        self.module.glm5_next_sparse_mla_reference(
            query,
            kv,
            indices,
            sm_scale=scale,
            kv_scale=kv_scale_cuda,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.module.glm5_next_sparse_mla_reference(
                query,
                kv,
                indices,
                sm_scale=scale,
                kv_scale=kv_scale_cuda,
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=3e-2, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
