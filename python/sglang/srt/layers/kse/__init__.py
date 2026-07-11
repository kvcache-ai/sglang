"""KVCache Sparsity Engine (KSE) — a sidecar module for sglang.

KSE provides pluggable KV Cache sparsity for models that do **not** have
native sparse attention.  It intercepts attention metadata (page tables,
kv_indices, seq_lens) and rewrites them so that only a selected subset of
KV entries participates in the attention computation — zero-copy, backend-
agnostic, and CUDA-graph compatible.

Public surface:
    create_kse_controller   — factory that wires policy + adapter
    KSEController           — central coordinator (hooks into forward path)
    KSEConfig               — dataclass configuration
    register_policy         — decorator for SparsityPolicy implementations
    register_adapter        — decorator for MetadataAdapter implementations
"""

from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.controller import KSEController
from sglang.srt.layers.kse.registry import (
    create_kse_controller,
    register_adapter,
    register_policy,
)

# Register built-in policies and adapters on import.
import sglang.srt.layers.kse.policies  # noqa: F401
import sglang.srt.layers.kse.adapters  # noqa: F401

__all__ = [
    "KSEConfig",
    "KSEController",
    "create_kse_controller",
    "register_adapter",
    "register_policy",
]
