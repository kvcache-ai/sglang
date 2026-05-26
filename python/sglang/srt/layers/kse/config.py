"""Configuration dataclass for the KVCache Sparsity Engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KSEConfig:
    """All tunables for a KSE instance."""

    policy_name: str  # e.g. "quest", "streaming_llm"
    backend_name: str  # e.g. "flashinfer", "triton", "flashattention"
    start_layer: int = 0
    end_layer: int = -1  # exclusive; -1 → all layers
    min_seq_len: int = 2048
    page_size: int = 64
    policy_kwargs: dict = field(default_factory=dict)
