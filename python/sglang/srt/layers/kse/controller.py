"""Central coordinator for the KVCache Sparsity Engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.config import KSEConfig
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_BACKEND_GRANULARITY = {
    "flashinfer": Granularity.TOKEN,
    "triton": Granularity.TOKEN,
    "flashattention": Granularity.PAGE,
}

_BACKEND_PAGE_SIZE = {
    "flashattention": 256,
}


def _get_backend_granularity(backend_name: str) -> Granularity:
    return _BACKEND_GRANULARITY.get(backend_name, Granularity.TOKEN)


def _get_backend_page_size(backend_name: str) -> int:
    return _BACKEND_PAGE_SIZE.get(backend_name, 1)


class KSEController:
    """Orchestrates sparsity policy, metadata adapter, and selection.

    KSE operates purely via **metadata rewriting** — it never physically
    modifies the KV cache or the allocator.  The ``select()`` mask produced
    each step determines which KV entries the attention kernel sees.

    Physical eviction (compacting ``req_to_token`` and calling
    ``allocator.free()``) is intentionally **not** performed here because
    the scheduler's ``Req.kv_committed_len`` bookkeeping is not accessible
    from the model-runner side.  Freeing slots without updating
    ``kv_committed_len`` causes double-frees when the request finishes.

    Integration points (4 call-sites in sglang):
        1. After extend forward in ``ModelRunner._forward_raw``
           → ``controller.after_prefill()``
        2. Start of ``ModelRunner.forward_decode()``
           → ``controller.before_forward()``
        3. ``AttentionBackend.forward()`` before decode kernel
           → ``controller.before_attention()``
        4. ``AttentionBackend.forward()`` after decode kernel
           → ``controller.after_attention()``
    """

    def __init__(
        self,
        policy: SparsityPolicy,
        adapter: MetadataAdapter,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: KVCache,
        config: KSEConfig,
    ):
        self.policy = policy
        self.adapter = adapter
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.config = config

        self._validate_granularity_compatibility(
            policy, config.backend_name, config
        )

        self._cached_result: Optional[SelectionResult] = None
        self._cached_step: int = -1
        self._metadata_saved: bool = False

    # ---- Hook: request lifecycle ----------------------------------------

    def on_request_begin(self, req) -> None:
        self.policy.on_request_begin(req)

    def on_request_end(self, req) -> None:
        self.policy.on_request_end(req)

    # ---- Hook: after prefill --------------------------------------------

    def after_prefill(self, forward_batch: ForwardBatch) -> None:
        """Build representations after prefill (e.g. bounding boxes for Quest)."""
        end_layer = (
            self.config.end_layer
            if self.config.end_layer > 0
            else self.token_to_kv_pool.layer_num
        )
        for layer_id in range(self.config.start_layer, end_layer):
            k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
            v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
            self.policy.on_prefill_complete(
                layer_id,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                k_buf,
                v_buf,
                forward_batch,
            )

        if self.policy.frequency() == Frequency.PER_REQUEST:
            self._cached_result = self.policy.select(
                query=None,
                layer_id=-1,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_batch=forward_batch,
            )

    # ---- Hook: before decode forward ------------------------------------

    def before_forward(self, forward_batch: ForwardBatch) -> None:
        """Reset per-step caches at the start of each decode step."""
        if self.policy.frequency() == Frequency.PER_STEP:
            self._cached_result = None
        self._metadata_saved = False

    # ---- Hook: before each layer's attention ----------------------------

    def before_attention(
        self,
        query: torch.Tensor,
        layer_id: int,
        forward_batch: ForwardBatch,
        forward_metadata: Any,
    ) -> Any:
        """Rewrite metadata for sparse attention; return (possibly modified) metadata."""
        if not self._should_apply(layer_id, forward_batch):
            return forward_metadata

        if not self._metadata_saved:
            self.adapter.save_dense_metadata(forward_metadata)
            self._metadata_saved = True

        result = self._get_selection(query, layer_id, forward_batch)
        return self.adapter.apply(result, forward_metadata, forward_batch, layer_id)

    # ---- Hook: after each layer's attention -----------------------------

    def after_attention(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> None:
        """Incrementally update policy state after attention."""
        if not forward_batch.forward_mode.is_decode():
            return
        k_buf = self.token_to_kv_pool.get_key_buffer(layer_id)
        v_buf = self.token_to_kv_pool.get_value_buffer(layer_id)
        self.policy.on_attention_complete(
            layer_id,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            k_buf,
            v_buf,
            forward_batch,
        )

    # ---- internal -------------------------------------------------------

    def _get_selection(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> SelectionResult:
        freq = self.policy.frequency()
        if freq == Frequency.PER_REQUEST:
            assert self._cached_result is not None
            return self._cached_result
        if freq == Frequency.PER_STEP:
            if self._cached_result is None:
                self._cached_result = self.policy.select(
                    query=query,
                    layer_id=layer_id,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    forward_batch=forward_batch,
                )
            return self._cached_result
        # PER_LAYER — always recompute
        return self.policy.select(
            query=query,
            layer_id=layer_id,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            forward_batch=forward_batch,
        )

    def _should_apply(self, layer_id: int, forward_batch: ForwardBatch) -> bool:
        if not forward_batch.forward_mode.is_decode():
            return False
        end_layer = (
            self.config.end_layer
            if self.config.end_layer > 0
            else self.token_to_kv_pool.layer_num
        )
        if layer_id < self.config.start_layer or layer_id >= end_layer:
            return False
        return True

    # ---- validation -----------------------------------------------------

    def _validate_granularity_compatibility(
        self,
        policy: SparsityPolicy,
        backend_name: str,
        config: KSEConfig,
    ) -> None:
        backend_gran = _get_backend_granularity(backend_name)
        policy_gran = policy.granularity()

        if (
            policy_gran == Granularity.TOKEN
            and backend_gran == Granularity.PAGE
        ):
            raise ValueError(
                f"Token-granularity sparse policy '{config.policy_name}' is "
                f"incompatible with page-granularity backend '{backend_name}'. "
                f"Use a token-granularity backend (FlashInfer, Triton) instead."
            )

        if (
            policy_gran == Granularity.PAGE
            and backend_gran == Granularity.PAGE
        ):
            backend_ps = _get_backend_page_size(backend_name)
            if config.page_size < backend_ps:
                raise ValueError(
                    f"Sparse page_size ({config.page_size}) < backend "
                    f"page_size ({backend_ps}). Must be >= and an integer multiple."
                )
            if config.page_size % backend_ps != 0:
                raise ValueError(
                    f"Sparse page_size ({config.page_size}) is not an integer "
                    f"multiple of backend page_size ({backend_ps})."
                )
