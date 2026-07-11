"""Abstract base classes for sparsity policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class SparsityPolicy(ABC):
    """Algorithm-agnostic interface for all KV-cache sparsity strategies.

    KSE operates purely via **metadata rewriting** — ``select()`` produces
    a mask that determines which KV entries the attention kernel sees.
    No physical KV data is moved, freed, or evicted.

    Lifecycle:
        1. ``__init__(config, device)``     — parse config, allocate buffers
        2. ``on_request_begin(req)``        — register per-request state
        3. ``on_prefill_complete(...)``      — build initial representation
        4. ``select(...)``                   — produce ``SelectionResult``
        5. ``on_attention_complete(...)``    — incremental update (optional)
        6. ``on_request_end(req)``           — clean up
    """

    @abstractmethod
    def granularity(self) -> Granularity:
        """Selection granularity (TOKEN or PAGE)."""
        ...

    @abstractmethod
    def frequency(self) -> Frequency:
        """How often ``select()`` needs to be called."""
        ...

    @abstractmethod
    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        """Produce a selection over KV entries.

        Args:
            query: ``[batch, num_heads, head_dim]``; may be ``None`` for
                query-unaware / fixed strategies.
            layer_id: Current layer index.
            req_pool_indices: ``[batch]`` indices into ``req_to_token_pool``.
            seq_lens: ``[batch]`` current sequence lengths.
            forward_batch: Full batch context.
        """
        ...

    # -- optional hooks --------------------------------------------------

    def on_request_begin(self, req) -> None:
        """Called when a new request enters the system."""

    def on_request_end(self, req) -> None:
        """Called when a request finishes or is aborted."""

    def on_prefill_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Called after each layer's prefill attention to build representations."""

    def on_attention_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Called after each decode attention to incrementally update state."""
