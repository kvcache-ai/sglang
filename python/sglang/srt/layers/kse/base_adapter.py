"""Abstract base class for metadata adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sglang.srt.layers.kse.types import SelectionResult

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MetadataAdapter(ABC):
    """Translates a ``SelectionResult`` into backend-specific metadata rewrites.

    Each attention backend has its own metadata format (FlashInfer uses
    ``kv_indptr`` / ``kv_indices``; FlashAttention uses ``page_table`` /
    ``cache_seqlens``; Triton uses ``kv_indptr`` / ``kv_indices``).  The
    adapter knows how to rewrite these structures to reflect the sparse
    selection.
    """

    @abstractmethod
    def save_dense_metadata(self, forward_metadata: Any) -> None:
        """Snapshot the original (dense) metadata before any sparse rewrite.

        Called once at the beginning of each forward pass (first sparse layer).
        """
        ...

    @abstractmethod
    def apply(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Any:
        """Rewrite *forward_metadata* in-place to reflect the sparse selection.

        Non-sparse requests (``result.sparse_mask == False``) keep their
        original (dense) metadata.
        """
        ...

    @abstractmethod
    def restore_dense_metadata(self, forward_metadata: Any) -> None:
        """Restore original dense metadata after sparse layers are done."""
        ...
