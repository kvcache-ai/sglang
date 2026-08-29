"""MetadataAdapter for the FlashInfer attention backend.

FlashInfer uses wrapper objects (``BatchDecodeWithPagedKVCacheWrapper``)
that internally manage CSR-format ``kv_indptr`` / ``kv_indices``.  The
wrappers are pre-planned during ``init_forward_metadata``.

Because re-planning a FlashInfer wrapper per-layer would be expensive,
this adapter takes a lightweight approach: it stores the original
``kv_indptr`` buffer that the wrapper references and overwrites it
in-place with the sparse selection.  Since FlashInfer's decode wrapper
reads ``kv_indptr`` / ``kv_indices`` at kernel-launch time (not at
plan time), in-place mutation is sufficient.

Note: This adapter works with the standard (non-MLA) FlashInfer backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.kse.base_adapter import MetadataAdapter
from sglang.srt.layers.kse.registry import register_adapter
from sglang.srt.layers.kse.types import Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@register_adapter("flashinfer")
class FlashInferAdapter(MetadataAdapter):
    """In-place rewrite of FlashInfer's kv_indptr / kv_indices buffers.

    FlashInfer's ``DecodeMetadata`` holds a list of
    ``BatchDecodeWithPagedKVCacheWrapper`` objects.  Each wrapper
    references shared ``kv_indptr`` and ``kv_indices`` buffers that were
    populated during ``init_forward_metadata``.  We snapshot those
    buffers and overwrite them with the sparse selection.

    Limitations:
        * Requires the FlashInfer backend to expose its ``kv_indptr``
          and ``kv_indices`` buffers (available on ``FlashInferAttnBackend``).
        * Re-planning the wrapper is not performed; this works because
          the decode kernel reads the CSR arrays at launch time.
    """

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self._dense_kv_indptr: Optional[torch.Tensor] = None
        self._dense_kv_indices: Optional[torch.Tensor] = None
        self._attn_backend_ref: Any = None

    def save_dense_metadata(self, forward_metadata: Any) -> None:
        """Snapshot the kv_indptr/kv_indices from the attention backend.

        ``forward_metadata`` for FlashInfer is a ``DecodeMetadata`` that
        holds wrapper objects.  We access the underlying buffers through
        the attention backend reference stored on the ForwardBatch.
        """
        # forward_metadata is DecodeMetadata — we need the raw buffers
        # from the FlashInferAttnBackend.  The controller passes the
        # metadata object; we'll look for the indptr/indices there.
        # FlashInfer stores kv_indptr as a list (one per wrapper).
        if hasattr(forward_metadata, '_kse_kv_indptr'):
            self._dense_kv_indptr = forward_metadata._kse_kv_indptr.clone()
            self._dense_kv_indices = forward_metadata._kse_kv_indices.clone()

    def apply(
        self,
        result: SelectionResult,
        forward_metadata: Any,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Any:
        if not hasattr(forward_metadata, '_kse_kv_indptr'):
            logger.warning(
                "FlashInfer adapter: forward_metadata lacks _kse_kv_indptr; "
                "skipping sparse rewrite for layer %d", layer_id
            )
            return forward_metadata

        bs = result.selected_indices.shape[0]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        page_size = self.config.page_size

        kv_indptr = forward_metadata._kse_kv_indptr
        kv_indices = forward_metadata._kse_kv_indices

        # Restore dense first
        kv_indptr[:] = self._dense_kv_indptr
        kv_indices[: self._dense_kv_indices.shape[0]] = self._dense_kv_indices

        new_indices_list = []
        new_indptr = torch.zeros(
            bs + 1, dtype=kv_indptr.dtype, device=self.device
        )

        for i in range(bs):
            if not result.sparse_mask[i]:
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
                new_indices_list.append(phys.to(kv_indices.dtype))
                new_indptr[i + 1] = new_indptr[i] + n_sel
            else:
                token_positions = []
                for page_idx in sel:
                    page_start = page_idx.item() * page_size
                    page_end = page_start + page_size
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
                    new_indices_list.append(phys.to(kv_indices.dtype))
                    new_indptr[i + 1] = new_indptr[i] + all_positions.shape[0]
                else:
                    new_indptr[i + 1] = new_indptr[i]

        if new_indices_list:
            new_kv_indices = torch.cat(new_indices_list)
        else:
            new_kv_indices = torch.empty(0, dtype=kv_indices.dtype, device=self.device)

        # In-place overwrite
        kv_indptr[: bs + 1] = new_indptr
        total = new_kv_indices.shape[0]
        kv_indices[:total] = new_kv_indices

        return forward_metadata

    def restore_dense_metadata(self, forward_metadata: Any) -> None:
        if self._dense_kv_indptr is None:
            return
        if not hasattr(forward_metadata, '_kse_kv_indptr'):
            return
        kv_indptr = forward_metadata._kse_kv_indptr
        kv_indices = forward_metadata._kse_kv_indices
        kv_indptr[:] = self._dense_kv_indptr
        n = self._dense_kv_indices.shape[0]
        kv_indices[:n] = self._dense_kv_indices
