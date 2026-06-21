"""Registry for KV memory-pool / allocator factories.

Plugin slot used by ModelRunnerKvCacheMixin to pick a non-default KV pool
(e.g. compressed-attention pool for DSv4, hisparse pool for V4-Flash) without
importing the pool classes from base code.

Plugins register themselves at import time via the DSV4 model entry-point.
Lookups are tried in registration order; first non-None match wins.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


_PoolFactory = Callable[..., Any]
_PoolPredicate = Callable[["ModelConfig", "ServerArgs"], bool]

_KV_POOL_FACTORIES: List[Tuple[str, _PoolPredicate, _PoolFactory]] = []
_HOST_POOL_FACTORIES: List[Tuple[str, _PoolPredicate, _PoolFactory]] = []


def register_kv_pool_factory(
    name: str, predicate: _PoolPredicate, factory: _PoolFactory
) -> None:
    """Register a KV pool / allocator factory.

    Args:
      name: stable id (for logging / debugging).
      predicate: (model_config, server_args) -> bool. Return True to claim
        ownership of pool construction.
      factory: callable invoked with whatever kwargs ModelRunnerKvCacheMixin
        passes; must return a (token_to_kv_pool, allocator) pair or whatever
        the caller expects.
    """
    for existing_name, _, _ in _KV_POOL_FACTORIES:
        if existing_name == name:
            return
    _KV_POOL_FACTORIES.append((name, predicate, factory))


def resolve_kv_pool_factory(
    model_config: "ModelConfig", server_args: "ServerArgs"
) -> Optional[Tuple[str, _PoolFactory]]:
    """Return (name, factory) for the first matching plugin, else None."""
    for name, predicate, factory in _KV_POOL_FACTORIES:
        if predicate(model_config, server_args):
            return name, factory
    return None


def register_host_pool_factory(
    name: str, predicate: _PoolPredicate, factory: _PoolFactory
) -> None:
    """Same shape as register_kv_pool_factory, but for host (CPU) backup pools
    used by hisparse / compressed-state coordinators."""
    for existing_name, _, _ in _HOST_POOL_FACTORIES:
        if existing_name == name:
            return
    _HOST_POOL_FACTORIES.append((name, predicate, factory))


def resolve_host_pool_factory(
    model_config: "ModelConfig", server_args: "ServerArgs"
) -> Optional[Tuple[str, _PoolFactory]]:
    for name, predicate, factory in _HOST_POOL_FACTORIES:
        if predicate(model_config, server_args):
            return name, factory
    return None


# ---------------------------------------------------------------------------
# MiniMax M3 sparse-attention pool registration
# ---------------------------------------------------------------------------

from sglang.srt.configs.model_config import is_minimax_sparse
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool


def _minimax_sparse_pool_predicate(model_config, server_args):
    return is_minimax_sparse(model_config.hf_config)


register_kv_pool_factory(
    "minimax_m3_sparse", _minimax_sparse_pool_predicate, MiniMaxSparseKVPool
)
