import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.io_struct import (
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.tokenizer_communicator_mixin import (
    TokenizerCommunicatorMixin,
    get_static_kt_lora_ref,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _Registry:
    def __init__(self, adapter):
        self.adapter = adapter
        self.acquired = None

    async def acquire(self, names):
        self.acquired = names
        if isinstance(names, list):
            return [self.adapter.lora_id for _ in names]
        return self.adapter.lora_id


def _static_manager():
    adapter = LoRARef(
        lora_id="static-id",
        lora_name="production-adapter",
        lora_path="/converted/nonexpert",
        pinned=True,
    )
    args = SimpleNamespace(
        kt_expert_lora_path="/converted/expert",
        kt_composite_lora_name=adapter.lora_name,
        kt_composite_lora_id=adapter.lora_id,
        lora_paths=[adapter],
        enable_lora=True,
    )
    manager = SimpleNamespace(
        server_args=args,
        lora_registry=_Registry(adapter),
        auto_create_handle_loop=lambda: None,
    )

    async def resolve_lora_path(obj):
        obj.lora_id = await manager.lora_registry.acquire(obj.lora_path)

    manager._resolve_lora_path = resolve_lora_path
    return manager, adapter


def test_static_kt_lora_ref_requires_exactly_one_matching_dynamic_adapter():
    manager, adapter = _static_manager()
    assert get_static_kt_lora_ref(manager.server_args) is adapter

    manager.server_args.lora_paths = []
    with pytest.raises(ValueError, match="exactly one"):
        get_static_kt_lora_ref(manager.server_args)
    manager.server_args.lora_paths = [adapter, adapter]
    with pytest.raises(ValueError, match="exactly one"):
        get_static_kt_lora_ref(manager.server_args)


def test_static_kt_lora_ref_requires_complete_matching_identity():
    manager, adapter = _static_manager()

    manager.server_args.kt_composite_lora_id = None
    with pytest.raises(ValueError, match="identity is incomplete"):
        get_static_kt_lora_ref(manager.server_args)

    manager.server_args.kt_composite_lora_id = "other-id"
    with pytest.raises(ValueError, match="identity does not match"):
        get_static_kt_lora_ref(manager.server_args)

    manager.server_args.kt_composite_lora_id = adapter.lora_id
    manager.server_args.kt_composite_lora_name = "other-name"
    with pytest.raises(ValueError, match="identity does not match"):
        get_static_kt_lora_ref(manager.server_args)


def test_static_kt_lora_ref_does_not_infer_identity_from_expert_path():
    manager, _ = _static_manager()
    manager.server_args.kt_composite_lora_name = None
    manager.server_args.kt_composite_lora_id = None

    assert get_static_kt_lora_ref(manager.server_args) is None


@pytest.mark.parametrize(
    "requested",
    [None, "other", ["production-adapter", None], ["production-adapter", "other"]],
)
def test_static_kt_lora_rejects_base_other_and_mixed_requests(requested):
    manager, _ = _static_manager()
    request = SimpleNamespace(rid="user-request", lora_path=requested, lora_id=None)

    with pytest.raises(ValueError, match="Every request in the batch"):
        asyncio.run(TokenizerManager._validate_and_resolve_lora(manager, request))


@pytest.mark.parametrize(
    "requested",
    ["production-adapter", ["production-adapter", "production-adapter"]],
)
def test_static_kt_lora_accepts_only_the_paired_adapter(requested):
    manager, adapter = _static_manager()
    request = SimpleNamespace(rid="user-request", lora_path=requested, lora_id=None)

    asyncio.run(TokenizerManager._validate_and_resolve_lora(manager, request))

    assert manager.lora_registry.acquired == requested
    expected_id = (
        adapter.lora_id
        if isinstance(requested, str)
        else [adapter.lora_id, adapter.lora_id]
    )
    assert request.lora_id == expected_id


def test_static_kt_lora_health_check_is_bound_to_complete_adapter():
    manager, adapter = _static_manager()
    request = SimpleNamespace(
        rid="HEALTH_CHECK_123",
        lora_path=None,
        lora_id=None,
    )

    asyncio.run(TokenizerManager._validate_and_resolve_lora(manager, request))

    assert request.lora_path == adapter.lora_name
    assert request.lora_id == adapter.lora_id


def test_static_kt_lora_disables_dynamic_load_and_unload():
    manager, adapter = _static_manager()

    load_result = asyncio.run(
        TokenizerCommunicatorMixin.load_lora_adapter(
            manager,
            LoadLoRAAdapterReqInput(lora_name="other", lora_path="/other"),
        )
    )
    tensor_result = asyncio.run(
        TokenizerCommunicatorMixin.load_lora_adapter_from_tensors(
            manager,
            LoadLoRAAdapterFromTensorsReqInput(
                lora_name="other",
                config_dict={},
                serialized_tensors="",
            ),
        )
    )
    unload_result = asyncio.run(
        TokenizerCommunicatorMixin.unload_lora_adapter(
            manager,
            UnloadLoRAAdapterReqInput(lora_name=adapter.lora_name),
        )
    )

    assert not load_result.success
    assert "statically paired" in load_result.error_message
    assert not tensor_result.success
    assert "statically paired" in tensor_result.error_message
    assert not unload_result.success
    assert "statically paired" in unload_result.error_message


def test_execution_layer_rejects_incomplete_static_kt_adapter_batches():
    manager = LoRAManager.__new__(LoRAManager)
    manager.static_kt_lora_id = "static-id"

    manager._validate_static_kt_lora_batch(["static-id", "static-id"])
    for invalid_ids in (None, [], [None], ["static-id", None], ["other"]):
        with pytest.raises(RuntimeError, match="every sequence"):
            manager._validate_static_kt_lora_batch(invalid_ids)


def test_internal_batches_select_the_complete_static_kt_adapter():
    manager = LoRAManager.__new__(LoRAManager)
    manager.static_kt_lora_id = "paired-adapter-id"

    assert manager.get_internal_lora_ids(3) == ["paired-adapter-id"] * 3
    manager._validate_static_kt_lora_batch(manager.get_internal_lora_ids(3))

    manager.static_kt_lora_id = None
    assert manager.get_internal_lora_ids(2) == [None, None]


def test_memory_pool_preloads_static_composite_adapter(monkeypatch):
    manager = LoRAManager.__new__(LoRAManager)
    manager.base_hf_config = SimpleNamespace()
    manager.max_loras_per_batch = 2
    manager.dtype = torch.bfloat16
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.max_lora_rank = 8
    manager.base_model = SimpleNamespace()
    manager.lora_modules = []
    manager.eviction_policy = "lru"
    manager.lora_added_tokens_size = 0
    manager.static_kt_lora_id = "paired-adapter-id"
    manager.get_runtime_target_modules = lambda: set()
    captured = []
    manager.fetch_new_loras = lambda ids: captured.append(ids)
    monkeypatch.setattr(
        "sglang.srt.lora.lora_manager.LoRAMemoryPool",
        lambda **kwargs: SimpleNamespace(),
    )

    manager.init_memory_pool()

    assert captured == [{"paired-adapter-id"}]
