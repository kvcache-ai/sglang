"""CPU contracts for the isolated GLM-5-Next hybrid KPool state."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_SOURCE = REPO_ROOT / "python/sglang/srt/mem_cache/glm5_next_memory_pool.py"
COORDINATOR_SOURCE = (
    REPO_ROOT / "python/sglang/srt/managers/glm5_next_kpool_coordinator.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeNSATokenToKVPool:
    quant_block_size = 128

    def __init__(self, **kwargs):
        self.size = kwargs["size"]
        self.page_size = kwargs["page_size"]
        self.dtype = kwargs["dtype"]
        self.layer_num = kwargs["layer_num"]
        self.device = kwargs["device"]
        self.start_layer = kwargs.get("start_layer") or 0
        self.custom_mem_pool = None
        self.memory_saver_adapter = SimpleNamespace(region=lambda tag: _NullContext())
        self.nsa_kv_cache_store_fp8 = (
            self.dtype == torch.float8_e4m3fn
            and kwargs["kv_cache_dim"]
            != kwargs["kv_lora_rank"] + kwargs["qk_rope_head_dim"]
        )
        self.kv_cache_dim = kwargs["kv_cache_dim"]
        self.index_head_dim = kwargs["index_head_dim"]
        self.index_cache_dtype = kwargs.get(
            "index_cache_dtype", torch.float8_e4m3fn
        )
        self.index_cache_is_bf16 = self.index_cache_dtype == torch.bfloat16
        self.mem_usage = 0
        self.index_k_with_scale_buffer = [
            torch.zeros(
                (1, 128 if self.index_cache_is_bf16 else 132),
                dtype=torch.bfloat16 if self.index_cache_is_bf16 else torch.uint8,
            )
            for _ in range(self.layer_num)
        ]

    def get_kv_size_bytes(self):
        return 1024

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(self, layer_id, seq_len, page_indices):
        return ("k", layer_id, seq_len, page_indices)

    def get_index_k_scale_continuous(self, layer_id, seq_len, page_indices):
        return ("scale", layer_id, seq_len, page_indices)

    def get_index_k_scale_buffer(self, layer_id, seq_len, page_indices):
        return ("both", layer_id, seq_len, page_indices)

    def set_index_k_scale_buffer(self, layer_id, loc, index_k, index_k_scale) -> None:
        self.last_index_write = (layer_id, loc, index_k, index_k_scale)


class _FakeHybridLinearKVPool:
    def _transfer_full_attention_id(self, layer_id):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(f"not a DSA layer: {layer_id}")
        return self.full_attention_layer_id_mapping[layer_id]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _load_sources():
    registered_pools = {}
    registered_coordinators = {}
    registered_hooks = {}

    packages = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.mem_cache",
        "sglang.srt.managers",
        "sglang.srt.model_executor",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    config = types.ModuleType("sglang.srt.configs.glm5_next")
    config.is_glm5_next = lambda value: bool(getattr(value, "exact_glm", False))
    config.uses_kpool4_compress = lambda value: bool(
        getattr(value, "exact_kpool", False)
    )

    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"

    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.HybridLinearKVPool = _FakeHybridLinearKVPool
    memory_pool.NSATokenToKVPool = _FakeNSATokenToKVPool

    pool_registry = types.ModuleType("sglang.srt.mem_cache.pool_registry")

    def register_pool(name, predicate, factory):
        registered_pools[name] = (predicate, factory)

    pool_registry.register_kv_pool_factory = register_pool

    coordinator_registry = types.ModuleType("sglang.srt.managers.coordinator_registry")
    coordinator_registry.register_request_coordinator = lambda name, factory: (
        registered_coordinators.setdefault(name, factory)
    )

    hook_registry = types.ModuleType("sglang.srt.managers.forward_hooks_registry")
    hook_registry.register_forward_hook = lambda name, hook: (
        registered_hooks.setdefault(name, hook)
    )

    stubs = {
        **packages,
        config.__name__: config,
        constants.__name__: constants,
        memory_pool.__name__: memory_pool,
        pool_registry.__name__: pool_registry,
        coordinator_registry.__name__: coordinator_registry,
        hook_registry.__name__: hook_registry,
    }
    with patch.dict(sys.modules, stubs):
        coordinator = _load_module(
            "_glm5_next_kpool_coordinator_under_test", COORDINATOR_SOURCE
        )
        pool = _load_module("_glm5_next_memory_pool_under_test", POOL_SOURCE)

    return (
        pool,
        coordinator,
        registered_pools,
        registered_coordinators,
        registered_hooks,
    )


(
    POOL,
    COORDINATOR,
    REGISTERED_POOLS,
    REGISTERED_COORDINATORS,
    REGISTERED_HOOKS,
) = _load_sources()


def _make_hybrid_pool(req_pool_size=6, dtype=torch.float8_e4m3fn):
    return POOL.Glm5NextHybridKVPool(
        size=128,
        dtype=dtype,
        page_size=64,
        head_num=1,
        head_dim=256,
        full_attention_layer_ids=[3, 7, 11],
        device="cpu",
        mamba_pool=object(),
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        kv_cache_dim=512,
        req_pool_size=req_pool_size,
    )


class TestGlm5NextKPoolMemoryPool(unittest.TestCase):
    def test_registration_predicate_is_exact_glm_and_kpool4(self):
        self.assertIn("glm5_next_kpool4", REGISTERED_POOLS)
        predicate, factory = REGISTERED_POOLS["glm5_next_kpool4"]
        self.assertIs(factory, POOL.build_glm5_next_kv_pool)
        self.assertTrue(
            predicate(SimpleNamespace(exact_glm=True, exact_kpool=True), None)
        )
        self.assertFalse(
            predicate(SimpleNamespace(exact_glm=True, exact_kpool=False), None)
        )
        self.assertFalse(
            predicate(SimpleNamespace(exact_glm=False, exact_kpool=True), None)
        )

    def test_tail_shape_dtype_and_memory_accounting_follow_kernel_abi(self):
        sparse = POOL.Glm5NextNSATokenToKVPool(
            size=128,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=0,
            layer_num=2,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=512,
            req_pool_size=6,
        )
        tail_k, tail_score = sparse.get_compress_tail_buffers(0)
        self.assertEqual(tail_k.shape, (6, 4, 128))
        self.assertEqual(tail_score.shape, tail_k.shape)
        self.assertEqual(tail_k.dtype, torch.bfloat16)
        self.assertEqual(tail_score.dtype, torch.bfloat16)
        latent_scale_bytes = 2 * (128 + 64) * 4 * 4
        expected_bytes = 1024 + latent_scale_bytes + 2 * 2 * 6 * 4 * 128 * 2
        self.assertEqual(sparse.get_kv_size_bytes(), expected_bytes)

    def test_scaled_fp8_latent_sidecar_is_compact_and_writable(self):
        pool = _make_hybrid_pool()
        for layer_id in (3, 7, 11):
            scale = pool.get_latent_scale_buffer(layer_id)
            self.assertEqual(scale.shape, (128 + 64, 4))
            self.assertEqual(scale.dtype, torch.float32)

        loc = torch.tensor([2, 9], dtype=torch.long)
        values = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4]], [[1.1, 1.2, 1.3, 1.4]]],
            dtype=torch.float32,
        )
        pool.set_latent_scale_buffer(7, loc, values)
        actual = pool.get_latent_scale_buffer(7)[loc]
        self.assertTrue(torch.equal(actual, values.squeeze(1)))
        self.assertEqual(torch.count_nonzero(pool.get_latent_scale_buffer(3)), 0)

        with self.assertRaises(ValueError):
            pool.get_latent_scale_buffer(0)

    def test_sm86_bf16_cache_omits_all_fp8_scale_sidecars(self):
        pool = _make_hybrid_pool(dtype=torch.bfloat16)
        self.assertTrue(pool.index_cache_is_bf16)
        self.assertEqual(pool.index_cache_dtype, torch.bfloat16)
        self.assertEqual(
            pool.get_index_k_with_scale_buffer(3).dtype, torch.bfloat16
        )
        self.assertIsNone(pool.get_latent_scale_buffer(3))
        with self.assertRaisesRegex(RuntimeError, "only for GLM FP8 KV cache"):
            pool.set_latent_scale_buffer(
                3,
                torch.tensor([1], dtype=torch.long),
                torch.ones((1, 4), dtype=torch.float32),
            )

    def test_speculative_cache_move_fails_closed(self):
        pool = _make_hybrid_pool()
        with self.assertRaisesRegex(
            NotImplementedError, "MTP/speculative decoding is not supported"
        ):
            pool.move_kv_cache(
                torch.tensor([5], dtype=torch.long),
                torch.tensor([3], dtype=torch.long),
            )

    def test_zero_rope_fp8_uses_raw_trtllm_cache_layout(self):
        pool = _make_hybrid_pool()
        self.assertEqual(pool.kv_cache_dim, 512)
        self.assertFalse(pool.nsa_kv_cache_store_fp8)

        kwargs = dict(
            size=128,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            req_pool_size=2,
        )
        with self.assertRaisesRegex(ValueError, "TRTLLM raw KV layout"):
            POOL.Glm5NextNSATokenToKVPool(**kwargs, kv_cache_dim=528)
        with self.assertRaisesRegex(ValueError, "zero-RoPE"):
            POOL.Glm5NextNSATokenToKVPool(
                **{**kwargs, "qk_rope_head_dim": 64}, kv_cache_dim=576
            )

    def test_tail_rows_clear_for_every_compact_dsa_layer(self):
        pool = _make_hybrid_pool()
        for layer_id in (3, 7, 11):
            tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
            tail_k[1:3].fill_(3)
            tail_score[1:3].fill_(5)

        pool.clear_compress_tail_rows(torch.tensor([2, 1, 2]))
        for layer_id in (3, 7, 11):
            tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
            self.assertEqual(torch.count_nonzero(tail_k[1:3]).item(), 0)
            self.assertEqual(torch.count_nonzero(tail_score[1:3]).item(), 0)

        with self.assertRaises(IndexError):
            pool.clear_compress_tail_rows(6)

    def test_global_dsa_ids_map_to_compact_index_and_kpool_calls(self):
        pool = _make_hybrid_pool()
        self.assertIs(
            pool.get_index_k_with_scale_buffer(7),
            pool.full_kv_pool.index_k_with_scale_buffer[1],
        )
        result = pool.get_index_k_continuous(11, 9, "pages")
        self.assertEqual(result, ("k", 2, 9, "pages"))

        pool.full_kv_pool.kpool_decode_update_index_cache = Mock()
        pool.kpool_decode_update_index_cache(layer_id=7, marker="decode")
        pool.full_kv_pool.kpool_decode_update_index_cache.assert_called_once_with(
            layer_id=1, marker="decode"
        )

        with self.assertRaises(ValueError):
            pool.get_index_k_with_scale_buffer(0)

    def test_wrong_layouts_fail_before_allocating(self):
        kwargs = dict(
            size=128,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=512,
            req_pool_size=2,
        )
        with self.assertRaisesRegex(ValueError, "page_size=64"):
            POOL.Glm5NextNSATokenToKVPool(**{**kwargs, "page_size": 1})
        with self.assertRaisesRegex(ValueError, "index_kpool=4"):
            POOL.Glm5NextNSATokenToKVPool(**kwargs, index_kpool=2)
        with self.assertRaisesRegex(ValueError, "always-selected tail"):
            POOL.Glm5NextNSATokenToKVPool(
                **kwargs, index_kpool_always_select_tail=False
            )

    def test_model_runner_factory_filters_pp_layers_and_attaches_lifecycle(self):
        text_config = SimpleNamespace(
            full_attention_layer_ids=[3, 7, 11],
            index_head_dim=128,
            index_kpool=4,
            index_kpool_compress=True,
            index_kpool_always_select_tail=True,
        )
        model_config = SimpleNamespace(
            exact_glm=True,
            exact_kpool=True,
            hf_text_config=text_config,
            head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            get_num_kv_heads=lambda tp: 1,
        )
        runner = SimpleNamespace(
            model_config=model_config,
            server_args=SimpleNamespace(enable_memory_saver=False),
            is_draft_worker=False,
            use_mla_backend=True,
            req_to_token_pool=SimpleNamespace(size=6, mamba_pool=object()),
            start_layer=4,
            end_layer=12,
            max_total_num_tokens=128,
            kv_cache_dtype=torch.float8_e4m3fn,
            page_size=64,
            device="cpu",
            calculate_mla_kv_cache_dim=lambda: 512,
        )

        layers_package = types.ModuleType("sglang.srt.layers")
        layers_package.__path__ = []
        dp_attention = types.ModuleType("sglang.srt.layers.dp_attention")
        dp_attention.get_attention_tp_size = lambda: 1
        coordinator = types.ModuleType(
            "sglang.srt.managers.glm5_next_kpool_coordinator"
        )
        lifecycle = object()
        coordinator.attach_glm5_next_kpool_lifecycle = lambda pool: lifecycle
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers": layers_package,
                dp_attention.__name__: dp_attention,
                coordinator.__name__: coordinator,
            },
        ):
            pool = POOL.build_glm5_next_kv_pool(model_runner=runner)

        self.assertEqual(pool.full_attention_layer_id_mapping, {7: 0, 11: 1})
        self.assertIs(pool.mamba_pool, runner.req_to_token_pool.mamba_pool)
        self.assertIs(pool.kpool_lifecycle_coordinator, lifecycle)


class TestGlm5NextKPoolLifecycle(unittest.TestCase):
    def tearDown(self):
        COORDINATOR.detach_glm5_next_kpool_lifecycle()

    def test_registration_is_inert_until_exact_pool_attaches(self):
        self.assertIs(
            REGISTERED_COORDINATORS["glm5_next_kpool"],
            COORDINATOR.Glm5NextKPoolCoordinator,
        )
        self.assertIn("glm5_next_kpool", REGISTERED_HOOKS)
        COORDINATOR.detach_glm5_next_kpool_lifecycle()
        hook = REGISTERED_HOOKS["glm5_next_kpool"]
        hook.on_request_finished(SimpleNamespace(req_pool_idx=1))
        self.assertIsNone(COORDINATOR.get_glm5_next_kpool_coordinator())

        with self.assertRaises(TypeError):
            COORDINATOR.Glm5NextKPoolCoordinator(SimpleNamespace())

    def test_prepare_is_pre_forward_but_post_prefill_admit_is_noop(self):
        pool = _make_hybrid_pool()
        coordinator = COORDINATOR.attach_glm5_next_kpool_lifecycle(pool)
        tail_k, tail_score = pool.get_compress_tail_buffers(3)

        tail_k[2].fill_(1)
        tail_score[2].fill_(2)
        coordinator.prepare_kpool_request(torch.tensor([2]))
        self.assertEqual(torch.count_nonzero(tail_k[2]).item(), 0)
        self.assertTrue(coordinator.is_prepared(2))

        # SGLang's current admit event is after prefill.  Preserve the tail
        # that prefill produced instead of accidentally clearing it.
        tail_k[2].fill_(7)
        tail_score[2].fill_(9)
        coordinator.on_request_admit(SimpleNamespace(req_pool_idx=2))
        self.assertEqual(torch.unique(tail_k[2]).tolist(), [7.0])
        self.assertEqual(torch.unique(tail_score[2]).tolist(), [9.0])

    def test_finish_and_retract_clear_before_request_row_release(self):
        pool = _make_hybrid_pool()
        coordinator = COORDINATOR.attach_glm5_next_kpool_lifecycle(pool)

        for row, event in (
            (1, coordinator.on_request_finished),
            (4, coordinator.on_request_retract),
        ):
            coordinator.prepare_kpool_request(row)
            for layer_id in (3, 7, 11):
                tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
                tail_k[row].fill_(11)
                tail_score[row].fill_(13)
            event(SimpleNamespace(req_pool_idx=row))
            self.assertFalse(coordinator.is_prepared(row))
            for layer_id in (3, 7, 11):
                tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
                self.assertEqual(torch.count_nonzero(tail_k[row]).item(), 0)
                self.assertEqual(torch.count_nonzero(tail_score[row]).item(), 0)

    def test_cache_flush_resets_prepared_rows_and_tail_state(self):
        pool = _make_hybrid_pool()
        coordinator = COORDINATOR.attach_glm5_next_kpool_lifecycle(pool)
        hook = REGISTERED_HOOKS["glm5_next_kpool"]

        coordinator.prepare_kpool_request([1, 4])
        for layer_id in (3, 7, 11):
            tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
            tail_k[1].fill_(17)
            tail_k[4].fill_(19)
            tail_score[1].fill_(23)
            tail_score[4].fill_(29)

        hook.on_cache_flush()

        self.assertFalse(coordinator.is_prepared(1))
        self.assertFalse(coordinator.is_prepared(4))
        for layer_id in (3, 7, 11):
            tail_k, tail_score = pool.get_compress_tail_buffers(layer_id)
            self.assertEqual(torch.count_nonzero(tail_k[[1, 4]]).item(), 0)
            self.assertEqual(torch.count_nonzero(tail_score[[1, 4]]).item(), 0)


if __name__ == "__main__":
    unittest.main()
