import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _DummyHostKVCache(HostKVCache):
    def get_size_per_token(self):
        return 10**9

    def init_kv_buffer(self):
        return torch.empty((1,), dtype=self.dtype)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        return None

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        return None

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        return torch.empty((0,), dtype=self.dtype)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.empty((0,), dtype=self.dtype)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        return None


class _FakeAllocator:
    def __init__(self, kv_cache):
        self._kv_cache = kv_cache

    def get_kvcache(self):
        return self._kv_cache


class _FakeKVCache:
    def __init__(self, size: int):
        self.size = size


class TestHostKVCacheSizing(CustomTestCase):
    def _device_pool(self, size: int = 1024):
        return SimpleNamespace(
            store_dtype=torch.float16,
            size=size,
            start_layer=0,
            end_layer=1,
        )

    def _virtual_memory(self):
        return SimpleNamespace(available=10**15)

    def test_cache_mode_rejects_host_pool_smaller_than_device_pool(self):
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.psutil.virtual_memory",
            return_value=self._virtual_memory(),
        ):
            with self.assertRaisesRegex(AssertionError, "host memory should be larger"):
                _DummyHostKVCache(
                    device_pool=self._device_pool(),
                    host_to_device_ratio=0.5,
                    host_size=0,
                    page_size=64,
                    layout="layer_first",
                    pin_memory=False,
                    device="cpu",
                    host_memory_mode="cache",
                )

    def test_buffer_only_mode_allows_host_pool_smaller_than_device_pool(self):
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.psutil.virtual_memory",
            return_value=self._virtual_memory(),
        ):
            cache = _DummyHostKVCache(
                device_pool=self._device_pool(),
                host_to_device_ratio=0.5,
                host_size=0,
                page_size=64,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                host_memory_mode="buffer_only",
            )

        self.assertEqual(cache.size, 512)
        self.assertEqual(cache.page_num, 8)
        self.assertEqual(cache.available_size(), 512)

    def test_explicit_hicache_size_allows_missing_ratio(self):
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.psutil.virtual_memory",
            return_value=self._virtual_memory(),
        ):
            cache = _DummyHostKVCache(
                device_pool=self._device_pool(size=32),
                host_to_device_ratio=None,
                host_size=1,
                page_size=64,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                host_memory_mode="buffer_only",
            )

        self.assertEqual(cache.size, 64)

    def test_host_pool_requires_ratio_or_size(self):
        with patch(
            "sglang.srt.mem_cache.memory_pool_host.psutil.virtual_memory",
            return_value=self._virtual_memory(),
        ):
            with self.assertRaisesRegex(ValueError, "Either --hicache-size or --hicache-ratio"):
                _DummyHostKVCache(
                    device_pool=self._device_pool(),
                    host_to_device_ratio=None,
                    host_size=0,
                    page_size=64,
                    layout="layer_first",
                    pin_memory=False,
                    device="cpu",
                    host_memory_mode="buffer_only",
                )


class TestHiRadixCacheBufferSizing(CustomTestCase):
    def _make_server_args(self, **overrides) -> ServerArgs:
        base = dict(
            model_path="dummy",
            device="cuda",
            enable_hierarchical_cache=True,
            hicache_host_memory_mode="buffer_only",
            hicache_write_policy="write_through",
            hicache_mem_layout="page_first",
            hicache_ratio=None,
            hicache_size=0,
            chunked_prefill_size=512,
            max_prefill_tokens=2048,
        )
        base.update(overrides)
        return ServerArgs(**base)

    def _make_params(self, kv_size: int):
        kv_cache = _FakeKVCache(size=kv_size)
        return SimpleNamespace(
            page_size=64,
            token_to_kv_pool_allocator=_FakeAllocator(kv_cache),
            tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            enable_metrics=False,
        )

    def _construct_cache(self, server_args: ServerArgs, kv_size: int):
        params = self._make_params(kv_size)
        with patch("sglang.srt.mem_cache.hiradix_cache.MHATokenToKVPool", _FakeKVCache), patch(
            "sglang.srt.mem_cache.hiradix_cache.MHATokenToKVPoolHost"
        ) as mock_host_pool, patch(
            "sglang.srt.mem_cache.hiradix_cache.HiCacheController",
            return_value=MagicMock(),
        ), patch(
            "sglang.srt.mem_cache.hiradix_cache.RadixCache.__init__",
            return_value=None,
        ), patch(
            "sglang.srt.mem_cache.hiradix_cache.torch.distributed.get_world_size",
            return_value=1,
        ), patch("sglang.srt.mem_cache.hiradix_cache.atexit.register"):
            HiRadixCache(params, server_args)
        return mock_host_pool.call_args

    def test_buffer_only_auto_ratio_uses_chunked_prefill_size(self):
        call_args = self._construct_cache(self._make_server_args(), kv_size=8192)

        self.assertAlmostEqual(call_args.args[1], 0.25)
        self.assertEqual(call_args.args[2], 0)
        self.assertEqual(call_args.kwargs["host_memory_mode"], "buffer_only")

    def test_buffer_only_auto_ratio_falls_back_to_max_prefill_tokens(self):
        call_args = self._construct_cache(
            self._make_server_args(chunked_prefill_size=0, max_prefill_tokens=256),
            kv_size=8192,
        )

        self.assertAlmostEqual(call_args.args[1], 0.125)

    def test_explicit_hicache_size_disables_auto_ratio(self):
        call_args = self._construct_cache(self._make_server_args(hicache_size=2), kv_size=8192)

        self.assertIsNone(call_args.args[1])
        self.assertEqual(call_args.args[2], 2)


class TestHiRadixCacheBufferOnlyPrefetchAnchor(CustomTestCase):
    def _make_cache(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.disable = False
        cache.page_size = 1
        cache.device = "cpu"
        cache.host_memory_mode = "buffer_only"
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.prefetch_threshold = 1
        cache.is_eagle = False
        cache.tp_world_size = 1
        cache.tp_group = None
        cache.evictable_size_ = 1
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.ongoing_prefetch = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.enable_kv_cache_events = False
        cache.kv_event_queue = []
        cache.key_match_fn = lambda node_key, key: 0
        cache.get_child_key_fn = lambda key: key[0]
        cache._get_extra_pools = lambda: {}
        cache.can_terminate_prefetch = lambda operation: True
        cache._load_to_device = lambda host_indices, node_id: torch.tensor(
            [5, 6], dtype=torch.int64
        )

        root = TreeNode(priority=-sys.maxsize)
        root.key = RadixKey(token_ids=[], extra_key=None)
        root.value = []
        root.host_value = []
        root.lock_ref = 1
        root.hash_value = []

        anchor = TreeNode(priority=0)
        anchor.parent = root
        anchor.key = RadixKey(token_ids=[11], extra_key=None)
        anchor.value = torch.tensor([1], dtype=torch.int64)
        anchor.hash_value = ["anchor"]
        root.children[11] = anchor

        cache.root_node = root
        cache._update_leaf_status(anchor)

        mem_pool_host = SimpleNamespace(
            alloc=lambda size: torch.arange(size, dtype=torch.int64),
            free=lambda indices: len(indices),
        )
        cache.cache_controller = SimpleNamespace(
            mem_pool_host=mem_pool_host,
            prefetch_rate_limited=lambda: False,
            prefetch=lambda req_id, host_indices, new_input_tokens, last_hash, prefix_keys, **kwargs: SimpleNamespace(host_indices=host_indices),
            terminate_prefetch=lambda operation: (2, ["h1", "h2"]),
            prefetch_tokens_occupied=0,
            evict_device=lambda indices: len(indices),
        )
        cache._record_store_event = lambda node: None
        return cache, anchor

    def test_buffer_only_prefetch_protects_anchor_from_eviction(self):
        cache, anchor = self._make_cache()

        cache.prefetch_from_storage(
            req_id="req-1",
            last_host_node=anchor,
            new_input_tokens=[21, 22],
            last_hash="anchor",
            prefix_keys=None,
        )

        self.assertEqual(anchor.host_ref_counter, 1)
        self.assertEqual(anchor.lock_ref, 1)
        self.assertNotIn(anchor, cache.evictable_leaves)

        done = cache.check_prefetch_progress("req-1")

        self.assertTrue(done)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(anchor.lock_ref, 0)


if __name__ == "__main__":
    unittest.main()