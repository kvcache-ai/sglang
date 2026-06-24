"""MetadataAdapter for the Triton attention backend.

The Triton backend uses CSR-format ``kv_indptr`` / ``kv_indices`` arrays.
This adapter rewrites those arrays so that only the selected KV entries
(token-level or page-level) appear in the index list.
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


@register_adapter("triton")
class TritonAdapter(MetadataAdapter):
    """Rewrite ``kv_indptr`` / ``kv_indices`` for the Triton backend."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self._dense_kv_indptr: Optional[torch.Tensor] = None
        self._dense_kv_indices: Optional[torch.Tensor] = None

    def save_dense_metadata(self, forward_metadata: Any) -> None:
        self._dense_kv_indptr = forward_metadata.kv_indptr.clone()
        self._dense_kv_indices = forward_metadata.kv_indices.clone()

    def apply(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Any:
        bs = result.selected_indices.shape[0]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        page_size = self.config.page_size

        # Build new kv_indices from the selection
        new_indices_list = []
        new_indptr = torch.zeros(
            bs + 1, dtype=self._dense_kv_indptr.dtype, device=self.device
        )

        for i in range(bs):
            if not result.sparse_mask[i]:
                # Dense: use original indices
                start = self._dense_kv_indptr[i].item()
                end = self._dense_kv_indptr[i + 1].item()
                new_indices_list.append(self._dense_kv_indices[start:end])
                new_indptr[i + 1] = new_indptr[i] + (end - start)
                continue

            n_sel = result.valid_lengths[i].item()
            if n_sel == 0:
                new_indptr[i + 1] = new_indptr[i]
                continue

            sel = result.selected_indices[i, :n_sel].long()
            req_idx = forward_batch.req_pool_indices[i]

            if result.granularity == Granularity.TOKEN:
                phys = req_to_token[req_idx, sel]
                new_indices_list.append(phys.to(self._dense_kv_indices.dtype))
                new_indptr[i + 1] = new_indptr[i] + n_sel
            else:
                # PAGE: expand each logical page to page_size token indices
                token_positions = []
                for page_idx in sel:
                    page_start = page_idx.item() * page_size
                    page_end = page_start + page_size
                    # Clamp to actual sequence length
                    seq_len = forward_batch.seq_lens[i].item()
                    page_end = min(page_end, seq_len)
                    if page_start >= seq_len:
                        continue
                    positions = torch.arange(
                        page_start, page_end, device=self.device, dtype=torch.long
                    )
                    token_positions.append(positions)

                if token_positions:
                    all_positions = torch.cat(token_positions)
                    phys = req_to_token[req_idx, all_positions]
                    new_indices_list.append(phys.to(self._dense_kv_indices.dtype))
                    new_indptr[i + 1] = new_indptr[i] + all_positions.shape[0]
                else:
                    new_indptr[i + 1] = new_indptr[i]

        if new_indices_list:
            new_kv_indices = torch.cat(new_indices_list)
        else:
            new_kv_indices = torch.empty(
                0, dtype=self._dense_kv_indices.dtype, device=self.device
            )

        forward_metadata.kv_indptr = new_indptr
        forward_metadata.kv_indices = new_kv_indices
        return forward_metadata

    def restore_dense_metadata(self, forward_metadata: Any) -> None:
        if self._dense_kv_indptr is not None:
            forward_metadata.kv_indptr = self._dense_kv_indptr
            forward_metadata.kv_indices = self._dense_kv_indices
