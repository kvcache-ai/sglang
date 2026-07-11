"""Policy / adapter registry and controller factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.controller import KSEController

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool

_POLICY_REGISTRY: Dict[str, Type[SparsityPolicy]] = {}
_ADAPTER_REGISTRY: Dict[str, Type[MetadataAdapter]] = {}


def register_policy(name: str):
    """Decorator: register a ``SparsityPolicy`` implementation."""

    def wrapper(cls: Type[SparsityPolicy]):
        _POLICY_REGISTRY[name] = cls
        return cls

    return wrapper


def register_adapter(name: str):
    """Decorator: register a ``MetadataAdapter`` implementation."""

    def wrapper(cls: Type[MetadataAdapter]):
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return wrapper


def create_kse_controller(
    config: KSEConfig,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool: KVCache,
    device: torch.device,
) -> KSEController:
    """Instantiate a fully wired ``KSEController``."""
    if config.policy_name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown KSE policy '{config.policy_name}'. "
            f"Available: {list(_POLICY_REGISTRY)}"
        )
    if config.backend_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown KSE adapter '{config.backend_name}'. "
            f"Available: {list(_ADAPTER_REGISTRY)}"
        )

    policy_cls = _POLICY_REGISTRY[config.policy_name]
    adapter_cls = _ADAPTER_REGISTRY[config.backend_name]

    policy = policy_cls(config, device)
    adapter = adapter_cls(config, device)

    return KSEController(
        policy=policy,
        adapter=adapter,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        config=config,
    )
