"""Static contracts for the GLM-5-Next SM86/SM89 kernel boundary."""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEXER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/glm5_next_indexer_triton.py"
)
INDEXER_CALLER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/nsa_indexer_kpool.py"
)
KPOOL_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/nsa/kpool_fp8_index.py"
SPARSE_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/nsa/glm5_next_sparse_attention.py"
)
BACKEND_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/nsa_backend.py"
CONFIG_PATH = REPO_ROOT / "python/sglang/srt/configs/glm5_next.py"


def _load_indexer_module():
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_consumer_indexer_test", INDEXER_PATH
    )
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


def _consumer_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    capability = torch.cuda.get_device_capability()
    return capability if capability in ((8, 6), (8, 9)) else None


def _expected_logits(query, keys, weights, starts, ends, scales=None):
    scores = torch.bmm(
        keys.to(torch.bfloat16).unsqueeze(0).expand(query.shape[0], -1, -1),
        query.to(torch.bfloat16).transpose(1, 2),
    )
    logits = (torch.relu(scores).float() * weights.float().unsqueeze(1)).sum(2)
    if scales is not None:
        logits *= scales.float().unsqueeze(0)
    positions = torch.arange(keys.shape[0], device=query.device).unsqueeze(0)
    valid = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


def _hadamard128_reference(values: torch.Tensor) -> torch.Tensor:
    output = values.float()
    stride = 1
    while stride < 128:
        view = output.reshape(*output.shape[:-1], -1, 2, stride)
        left = view[..., 0, :].clone()
        right = view[..., 1, :].clone()
        output = torch.stack((left + right, left - right), dim=-2).reshape_as(output)
        stride *= 2
    return output * (1.0 / math.sqrt(128.0))


def _compressed_bf16_reference(slot_k, slot_score, ape):
    probabilities = torch.softmax(slot_score.float() + ape.unsqueeze(0), dim=1)
    pooled = (probabilities * slot_k.float()).sum(1).to(torch.bfloat16).float()
    return _hadamard128_reference(pooled).to(torch.bfloat16)


class TestGlm5NextConsumerGPUKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = _load_indexer_module()

    def test_indexer_has_exact_static_configs(self):
        self.assertEqual(
            self.indexer.glm5_next_indexer_launch_config((8, 6)), (32, 4, 2)
        )
        self.assertEqual(
            self.indexer.glm5_next_indexer_launch_config((8, 9)), (64, 4, 3)
        )
        for capability in ((8, 0), (9, 0), (12, 0)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    self.indexer.glm5_next_indexer_launch_config(capability)

    def test_indexer_cpu_route_is_disabled(self):
        self.assertFalse(self.indexer.use_glm5_next_triton_indexer(torch.device("cpu")))

    def test_indexer_kernels_keep_exact_score_contract(self):
        flat = _function_source(INDEXER_PATH, "_glm5_next_flat_mqa_logits_kernel")
        paged = _function_source(INDEXER_PATH, "_glm5_next_paged_mqa_logits_kernel")
        for source in (flat, paged):
            self.assertIn("tl.dot", source)
            self.assertIn("tl.maximum(head_scores, 0.0)", source)
            self.assertIn("tl.sum(head_scores * weights[None, :], axis=1)", source)
            self.assertIn("if USE_K_SCALE", source)
            for host_sync in (".item()", ".cpu()", ".tolist()"):
                self.assertNotIn(host_sync, source)
        self.assertIn("page_table_ptr", paged)
        self.assertIn("key_offsets < seq_len", paged)

    def test_dispatch_preserves_sm120_and_generic_nsa_paths(self):
        paged = _function_source(INDEXER_CALLER_PATH, "_get_topk_paged")
        ragged = _function_source(INDEXER_CALLER_PATH, "_get_topk_ragged_kpool_plan")
        forward = _function_source(INDEXER_CALLER_PATH, "forward_cuda")

        self.assertLess(
            paged.index("if use_triton_logits"), paged.index("elif use_eager_logits")
        )
        self.assertIn("deep_gemm.fp8_paged_mqa_logits", paged)
        self.assertIn("gather_index_k_bf16_prefix_into", ragged)
        self.assertIn("deep_gemm.fp8_mqa_logits", ragged)
        self.assertIn("if getattr(pool, 'index_cache_is_bf16', False)", forward)
        self.assertIn("q_fp8, q_scale = act_quant", forward)

    def test_sm86_kpool_has_isolated_bf16_transport_kernels(self):
        source = KPOOL_PATH.read_text(encoding="utf-8")
        for name in (
            "_gather_index_k_bf16_prefix_into_kernel",
            "_kpool_softmax_rotate_write_cache_bf16_kernel",
            "_kpool_decode_update_and_maybe_write_cache_bf16_kernel",
            "_kpool_assemble_softmax_rotate_write_cache_bf16_kernel",
        ):
            self.assertIn(f"def {name}(", source)
        self.assertIn("if buf.dtype == torch.bfloat16", source)

    def test_capability_matrix_and_flashinfer_boundary_fail_closed(self):
        config = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("if capability == (8, 6)", config)
        self.assertIn("if capability == (8, 9)", config)
        self.assertIn("if capability[0] >= 10", config)

        forward = _function_source(BACKEND_PATH, "_forward_trtllm")
        native_dispatch = forward.index("glm5_next_sparse_mla_reference")
        flashinfer_import = forward.index("import flashinfer.decode")
        self.assertLess(native_dispatch, flashinfer_import)

    @unittest.skipUnless(
        _consumer_capability() is not None,
        "an SM86 or SM89 GPU is required",
    )
    def test_flat_indexer_matches_bf16_formula(self):
        capability = _consumer_capability()
        assert capability is not None
        torch.manual_seed(20260817)
        device = torch.device("cuda")
        query_bf16 = (torch.randn(5, 32, 128, device=device) * 0.1).to(torch.bfloat16)
        keys_bf16 = (torch.randn(137, 128, device=device) * 0.1).to(torch.bfloat16)
        weights = torch.randn(5, 32, dtype=torch.float32, device=device)
        starts = torch.tensor([0, 1, 17, 64, 137], dtype=torch.int32, device=device)
        ends = torch.tensor([137, 80, 101, 137, 137], dtype=torch.int32, device=device)

        if capability == (8, 6):
            query, keys, scales = query_bf16, keys_bf16, None
        else:
            query = query_bf16.to(torch.float8_e4m3fn)
            keys = keys_bf16.to(torch.float8_e4m3fn)
            scales = torch.linspace(0.5, 1.5, keys.shape[0], device=device)

        chunks = list(
            self.indexer.iter_glm5_next_triton_mqa_logits(
                query,
                keys,
                weights,
                starts,
                ends,
                k_scale=scales,
                query_chunk_size=2,
            )
        )
        actual = torch.cat([chunk for _, _, chunk in chunks])
        expected = _expected_logits(query, keys, weights, starts, ends, scales)
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=2e-2, atol=2e-2
        )

    @unittest.skipUnless(
        _consumer_capability() is not None,
        "an SM86 or SM89 GPU is required",
    )
    def test_paged_indexer_matches_formula_and_replays_cuda_graph(self):
        capability = _consumer_capability()
        assert capability is not None
        torch.manual_seed(20260818)
        device = torch.device("cuda")
        pages = 4
        keys_bf16 = (torch.randn(pages, 64, 128, device=device) * 0.1).to(
            torch.bfloat16
        )
        query_bf16 = (torch.randn(2, 32, 128, device=device) * 0.1).to(torch.bfloat16)
        weights = torch.randn(2, 32, dtype=torch.float32, device=device)
        page_table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([70, 9], dtype=torch.int32, device=device)

        if capability == (8, 6):
            query = query_bf16
            cache = keys_bf16.reshape(pages, -1).contiguous()
            page_scales = None
            use_scale = False
        else:
            query = query_bf16.to(torch.float8_e4m3fn)
            keys = keys_bf16.to(torch.float8_e4m3fn)
            page_scales = torch.linspace(
                0.5, 1.5, pages * 64, dtype=torch.float32, device=device
            ).reshape(pages, 64)
            cache = torch.empty(
                (pages, 64 * 128 + 64 * 4), dtype=torch.uint8, device=device
            )
            cache[:, : 64 * 128] = keys.reshape(pages, -1).view(torch.uint8)
            cache[:, 64 * 128 :] = page_scales.view(torch.uint8).reshape(pages, -1)
            use_scale = True

        def expected_for_lengths(lengths):
            rows = []
            for row in range(2):
                ordered_keys = keys_bf16.index_select(
                    0, page_table[row].to(torch.int64)
                ).reshape(-1, 128)
                if capability == (8, 9):
                    ordered_keys = ordered_keys.to(torch.float8_e4m3fn)
                    ordered_scales = page_scales.index_select(
                        0, page_table[row].to(torch.int64)
                    ).reshape(-1)
                else:
                    ordered_scales = None
                rows.append(
                    _expected_logits(
                        query[row : row + 1],
                        ordered_keys,
                        weights[row : row + 1],
                        torch.zeros(1, dtype=torch.int32, device=device),
                        lengths[row : row + 1],
                        ordered_scales,
                    )[0]
                )
            return torch.stack(rows)

        self.indexer.glm5_next_triton_paged_mqa_logits(
            query, cache, weights, seq_lens, page_table, 128, use_k_scale=use_scale
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.indexer.glm5_next_triton_paged_mqa_logits(
                query,
                cache,
                weights,
                seq_lens,
                page_table,
                128,
                use_k_scale=use_scale,
            )
        graph.replay()
        torch.cuda.synchronize()
        expected = expected_for_lengths(seq_lens)
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=2e-2, atol=2e-2
        )

        seq_lens.copy_(torch.tensor([3, 127], dtype=torch.int32, device=device))
        graph.replay()
        torch.cuda.synchronize()
        expected = expected_for_lengths(seq_lens)
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=2e-2, atol=2e-2
        )

    @unittest.skipUnless(
        _consumer_capability() is not None,
        "an SM86 or SM89 GPU is required",
    )
    def test_consumer_sparse_mla_matches_oracle_and_replays_cuda_graph(self):
        capability = _consumer_capability()
        assert capability is not None
        spec = importlib.util.spec_from_file_location(
            "_glm5_next_sparse_consumer_test", SPARSE_PATH
        )
        assert spec is not None and spec.loader is not None
        sparse = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sparse)

        torch.manual_seed(20260822)
        query_cpu = (torch.randn(1, 2, 512) * 0.1).to(torch.bfloat16)
        kv_bf16 = (torch.randn(16, 512) * 0.1).to(torch.bfloat16)
        first_indices_cpu = torch.tensor(
            [[0, 3, 5, 9, -1, -1, -1, -1]], dtype=torch.int32
        )
        second_indices_cpu = torch.tensor(
            [[1, 2, 7, 12, 15, -1, -1, -1]], dtype=torch.int32
        )
        scale = 512**-0.5
        if capability == (8, 6):
            kv_cpu = kv_bf16
            kv_scale_cpu = None
        else:
            blocks = kv_bf16.float().view(-1, 4, 128)
            kv_scale_cpu = blocks.abs().amax(-1).clamp_min(1e-4) / 448.0
            kv_cpu = (
                (blocks / kv_scale_cpu.unsqueeze(-1))
                .to(torch.float8_e4m3fn)
                .view_as(kv_bf16)
            )

        expected = sparse.glm5_next_sparse_mla_reference(
            query_cpu,
            kv_cpu,
            first_indices_cpu,
            sm_scale=scale,
            kv_scale=kv_scale_cpu,
        )
        query = query_cpu.cuda()
        kv = kv_cpu.cuda()
        indices = first_indices_cpu.cuda()
        kv_scale = None if kv_scale_cpu is None else kv_scale_cpu.cuda()
        actual = sparse.glm5_next_sparse_mla_reference(
            query,
            kv,
            indices,
            sm_scale=scale,
            kv_scale=kv_scale,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse.glm5_next_sparse_mla_reference(
                query,
                kv,
                indices,
                sm_scale=scale,
                kv_scale=kv_scale,
            )
        indices.copy_(second_indices_cpu)
        graph.replay()
        torch.cuda.synchronize()
        expected = sparse.glm5_next_sparse_mla_reference(
            query_cpu,
            kv_cpu,
            second_indices_cpu,
            sm_scale=scale,
            kv_scale=kv_scale_cpu,
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        _consumer_capability() == (8, 6),
        "an SM86 GPU is required",
    )
    def test_sm86_bf16_kpool_writer_matches_reference(self):
        spec = importlib.util.spec_from_file_location(
            "_glm5_next_kpool_sm86_test", KPOOL_PATH
        )
        assert spec is not None and spec.loader is not None
        kpool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kpool)

        torch.manual_seed(20260819)
        device = torch.device("cuda")
        slot_k = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        slot_score = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        ape = torch.randn(4, 128, dtype=torch.float32, device=device)
        loc = torch.tensor([0, 65], dtype=torch.int64, device=device)
        cache = torch.zeros((2, 64 * 128), dtype=torch.bfloat16, device=device)
        pool = SimpleNamespace(page_size=64, index_head_dim=128)

        compressed, scales = kpool.kpool_softmax_rotate_write_cache(
            pool,
            cache,
            slot_k,
            slot_score,
            ape,
            loc,
            return_compressed=True,
        )
        expected = _compressed_bf16_reference(slot_k, slot_score, ape)

        torch.testing.assert_close(compressed, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(scales, torch.ones_like(scales), rtol=0, atol=0)
        torch.testing.assert_close(cache[0, :128], expected[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(cache[1, 128:256], expected[1], rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        _consumer_capability() == (8, 6),
        "an SM86 GPU is required",
    )
    def test_sm86_bf16_gather_and_extend_assemble_match_reference(self):
        spec = importlib.util.spec_from_file_location(
            "_glm5_next_kpool_sm86_extend_test", KPOOL_PATH
        )
        assert spec is not None and spec.loader is not None
        kpool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kpool)

        torch.manual_seed(20260820)
        device = torch.device("cuda")
        pool = SimpleNamespace(
            page_size=64,
            index_head_dim=128,
            index_kpool=4,
            tail_extra_slots=0,
            slots_per_page=64,
        )
        cache = torch.randn(3, 64 * 128, dtype=torch.bfloat16, device=device)
        page_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
        gathered = torch.empty(70, 128, dtype=torch.bfloat16, device=device)
        kpool.gather_index_k_bf16_prefix_into(pool, cache, page_indices, 70, gathered)
        expected_gather = cache.index_select(0, page_indices.long()).view(-1, 128)[:70]
        self.assertTrue(torch.equal(gathered, expected_gather))

        cache.zero_()
        chunk_k = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        chunk_score = torch.randn(4, 128, dtype=torch.bfloat16, device=device)
        tail_k = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        tail_score = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        ape = torch.randn(4, 128, dtype=torch.float32, device=device)
        request = torch.tensor([1], dtype=torch.int32, device=device)
        n_from_tail = torch.tensor([2], dtype=torch.int32, device=device)
        chunk_start = torch.tensor([1], dtype=torch.int64, device=device)
        tail_base = torch.tensor([3], dtype=torch.int64, device=device)
        loc = torch.tensor([65], dtype=torch.int64, device=device)
        kpool.kpool_assemble_softmax_rotate_write_cache(
            pool,
            cache,
            chunk_k,
            chunk_score,
            tail_k,
            tail_score,
            request,
            n_from_tail,
            chunk_start,
            tail_base,
            ape,
            loc,
        )
        selected_k = torch.stack(
            (tail_k[1, 3], tail_k[1, 0], chunk_k[1], chunk_k[2])
        ).unsqueeze(0)
        selected_score = torch.stack(
            (tail_score[1, 3], tail_score[1, 0], chunk_score[1], chunk_score[2])
        ).unsqueeze(0)
        expected = _compressed_bf16_reference(selected_k, selected_score, ape)[0]
        torch.testing.assert_close(cache[1, 128:256], expected, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        _consumer_capability() == (8, 6),
        "an SM86 GPU is required",
    )
    def test_sm86_bf16_decode_writer_replays_cuda_graph(self):
        spec = importlib.util.spec_from_file_location(
            "_glm5_next_kpool_sm86_decode_test", KPOOL_PATH
        )
        assert spec is not None and spec.loader is not None
        kpool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kpool)

        torch.manual_seed(20260821)
        device = torch.device("cuda")
        pool = SimpleNamespace(
            page_size=64,
            index_head_dim=128,
            index_kpool=4,
            tail_extra_slots=0,
            slots_per_page=64,
        )
        cache = torch.zeros(2, 64 * 128, dtype=torch.bfloat16, device=device)
        tail_seed = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        score_seed = torch.randn(2, 4, 128, dtype=torch.bfloat16, device=device)
        tail_k = tail_seed.clone()
        tail_score = score_seed.clone()
        key = torch.randn(1, 128, dtype=torch.bfloat16, device=device)
        score = torch.randn(1, 128, dtype=torch.bfloat16, device=device)
        ape = torch.randn(4, 128, dtype=torch.float32, device=device)
        block_tables = torch.zeros(1, 8, dtype=torch.int32, device=device)
        block_tables[0, 0] = 1
        request = torch.tensor([1], dtype=torch.int32, device=device)
        positions = torch.tensor([3], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=device)
        cache_locs = torch.tensor([1], dtype=torch.int64, device=device)

        def write_decode():
            kpool.kpool_decode_update_and_maybe_write_cache(
                pool,
                cache,
                tail_k,
                tail_score,
                key,
                score,
                ape,
                block_tables,
                request,
                positions,
                seq_lens,
                cache_locs,
            )

        write_decode()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        tail_k.copy_(tail_seed)
        tail_score.copy_(score_seed)
        cache.zero_()
        with torch.cuda.graph(graph):
            write_decode()
        tail_k.copy_(tail_seed)
        tail_score.copy_(score_seed)
        cache.zero_()
        graph.replay()
        torch.cuda.synchronize()

        selected_k = torch.stack(
            (tail_seed[1, 0], tail_seed[1, 1], tail_seed[1, 2], key[0])
        ).unsqueeze(0)
        selected_score = torch.stack(
            (score_seed[1, 0], score_seed[1, 1], score_seed[1, 2], score[0])
        ).unsqueeze(0)
        expected = _compressed_bf16_reference(selected_k, selected_score, ape)[0]
        torch.testing.assert_close(cache[1, :128], expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(tail_k[1, 3], key[0]))
        self.assertTrue(torch.equal(tail_score[1, 3], score[0]))


if __name__ == "__main__":
    unittest.main()
