"""MetadataAdapter for the FlashAttention (FA3) backend.

FlashAttention uses ``page_table`` (2-D tensor mapping logical pages to
physical page indices) and ``cache_seqlens_int32`` (per-request KV length).
This adapter rewrites those tensors so that only selected pages are visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.registry import register_adapter
from sglang.srt.layers.kse.types import Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_adapter("flashattention")
class FlashAttentionAdapter(MetadataAdapter):
    """Rewrite ``page_table`` / ``cache_seqlens_int32`` for FA3."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self._dense_page_table: Optional[torch.Tensor] = None
        self._dense_cache_seqlens: Optional[torch.Tensor] = None
        self._dense_max_seq_len_k: Optional[int] = None

    def save_dense_metadata(self, forward_metadata: Any) -> None:
        self._dense_page_table = forward_metadata.page_table.clone()
        self._dense_cache_seqlens = forward_metadata.cache_seqlens_int32.clone()
        self._dense_max_seq_len_k = forward_metadata.max_seq_len_k

    def apply(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Any:
        if result.granularity == Granularity.PAGE:
            return self._apply_page(result, forward_metadata, forward_batch)
        else:
            return self._apply_token(result, forward_metadata, forward_batch)

    def _apply_page(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any:
        """Rewrite page_table for page-granularity selection."""
        bs = result.selected_indices.shape[0]
        page_size = forward_batch.token_to_kv_pool.page_size
        sparse_page_size = self.config.page_size
        pages_per_sparse = sparse_page_size // page_size

        # Restore dense state first so we read from the original
        forward_metadata.page_table[:] = self._dense_page_table
        forward_metadata.cache_seqlens_int32[:] = self._dense_cache_seqlens

        max_new_pages = 0

        for i in range(bs):
            if not result.sparse_mask[i]:
                continue

            n_sel = result.valid_lengths[i].item()
            if n_sel == 0:
                forward_metadata.cache_seqlens_int32[i] = 0
                continue

            sel = result.selected_indices[i, :n_sel].long()

            # Map logical sparse-page indices to backend page indices
            new_page_indices = []
            for sp_idx in sel:
                base_backend_page = sp_idx.item() * pages_per_sparse
                for offset in range(pages_per_sparse):
                    bp = base_backend_page + offset
                    if bp < self._dense_page_table.shape[1]:
                        new_page_indices.append(
                            self._dense_page_table[i, bp].item()
                        )

            n_backend_pages = len(new_page_indices)
            if n_backend_pages > 0:
                page_tensor = torch.tensor(
                    new_page_indices,
                    dtype=forward_metadata.page_table.dtype,
                    device=self.device,
                )
                forward_metadata.page_table[i, :n_backend_pages] = page_tensor
            forward_metadata.cache_seqlens_int32[i] = n_backend_pages * page_size
            max_new_pages = max(max_new_pages, n_backend_pages)

        # Update max_seq_len_k to reflect the reduced KV length
        if max_new_pages > 0:
            forward_metadata.max_seq_len_k = int(
                forward_metadata.cache_seqlens_int32.max().item()
            )

        return forward_metadata

    def _apply_token(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
    ) -> Any:
        """Token-granularity on a page backend is disallowed at init time.

        This method exists only as a safety net; the controller's
        compatibility check prevents this combination.
        """
        raise RuntimeError(
            "Token-granularity selection is not supported on the "
            "FlashAttention (page-granularity) backend."
        )

    def restore_dense_metadata(self, forward_metadata: Any) -> None:
        if self._dense_page_table is not None:
            forward_metadata.page_table[:] = self._dense_page_table
            forward_metadata.cache_seqlens_int32[:] = self._dense_cache_seqlens
            forward_metadata.max_seq_len_k = self._dense_max_seq_len_k
