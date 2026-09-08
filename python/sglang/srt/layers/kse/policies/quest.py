"""Quest — Query-Aware, Page-granularity, Per-Layer sparsity policy.

Quest maintains per-page bounding boxes (min/max) of keys and uses the
current query to compute an upper-bound attention score for each page.
Only the top-scoring pages (plus a fixed number of recent pages) are
selected for attention.

Reference: Tang et al., "Quest: Query-Aware Sparsity for Efficient
Long-Context LLM Inference", ICML 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.registry import register_policy
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_policy("quest")
class QuestPolicy(SparsityPolicy):
    """Page-level bounding-box scoring with per-layer frequency."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self.page_size = config.page_size
        self.token_budget_ratio = config.policy_kwargs.get("token_budget_ratio", 0.3)
        self.num_recent_pages = config.policy_kwargs.get("num_recent_pages", 4)
        self.min_seq_len = config.min_seq_len

        # Per-layer bounding-box buffers, lazily allocated.
        # Keyed by layer_id → tensor of shape [pool_size, num_kv_heads, head_dim]
        self._page_k_min: Dict[int, torch.Tensor] = {}
        self._page_k_max: Dict[int, torch.Tensor] = {}

    # -- SparsityPolicy interface -----------------------------------------

    def granularity(self) -> Granularity:
        return Granularity.PAGE

    def frequency(self) -> Frequency:
        return Frequency.PER_LAYER

    def on_prefill_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Compute per-page key bounding boxes from the prefill KV cache."""
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        batch_size = req_pool_indices.shape[0]

        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            num_pages = seq_len // self.page_size
            if num_pages == 0:
                continue

            req_idx = req_pool_indices[i].item()
            usable_len = num_pages * self.page_size
            token_indices = req_to_token[req_idx, :usable_len].long()

            # k_buffer: [pool_size, num_kv_heads, head_dim]
            keys = k_buffer[token_indices]  # [usable_len, num_kv_heads, head_dim]
            num_kv_heads = keys.shape[1]
            head_dim = keys.shape[2]

            self._ensure_buffers(layer_id, num_kv_heads, head_dim)

            paged_keys = keys.view(num_pages, self.page_size, num_kv_heads, head_dim)

            # Map logical page → physical page index (first token of page / page_size)
            page_starts = torch.arange(
                0, usable_len, self.page_size, device=self.device
            )
            phys_slots = token_indices[page_starts]
            phys_page_ids = phys_slots // self.page_size

            self._page_k_min[layer_id][phys_page_ids] = paged_keys.amin(dim=1).to(
                self._page_k_min[layer_id].dtype
            )
            self._page_k_max[layer_id][phys_page_ids] = paged_keys.amax(dim=1).to(
                self._page_k_max[layer_id].dtype
            )

    def on_attention_complete(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Incrementally update the bounding box of the last page."""
        if layer_id not in self._page_k_min:
            return

        req_to_token = forward_batch.req_to_token_pool.req_to_token
        batch_size = req_pool_indices.shape[0]

        for i in range(batch_size):
            seq_len = seq_lens[i].item()
            if seq_len < self.page_size:
                continue

            req_idx = req_pool_indices[i].item()
            last_page_start = (seq_len - 1) // self.page_size * self.page_size
            last_page_end = seq_len
            token_indices = req_to_token[
                req_idx, last_page_start:last_page_end
            ].long()

            keys = k_buffer[token_indices]
            phys_slot = token_indices[0]
            phys_page_id = phys_slot // self.page_size

            k_min_cur = keys.amin(dim=0)
            k_max_cur = keys.amax(dim=0)

            dtype = self._page_k_min[layer_id].dtype
            self._page_k_min[layer_id][phys_page_id] = torch.minimum(
                self._page_k_min[layer_id][phys_page_id], k_min_cur.to(dtype)
            )
            self._page_k_max[layer_id][phys_page_id] = torch.maximum(
                self._page_k_max[layer_id][phys_page_id], k_max_cur.to(dtype)
            )

    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        assert query is not None, "Quest requires the query tensor."

        bs = query.shape[0]
        req_to_token = forward_batch.req_to_token_pool.req_to_token

        all_indices = []
        all_lengths = []
        sparse_flags = []

        for i in range(bs):
            seq_len = seq_lens[i].item()
            num_pages = seq_len // self.page_size

            if seq_len < self.min_seq_len or num_pages == 0:
                sparse_flags.append(False)
                all_indices.append(torch.zeros(1, dtype=torch.int32, device=self.device))
                all_lengths.append(0)
                continue

            sparse_flags.append(True)
            req_idx = req_pool_indices[i].item()

            # Gather physical page ids for this request
            page_starts = torch.arange(
                0, num_pages * self.page_size, self.page_size, device=self.device
            )
            phys_slots = req_to_token[req_idx, page_starts.long()].long()
            phys_page_ids = phys_slots // self.page_size

            k_min = self._page_k_min[layer_id][phys_page_ids]  # [num_pages, H, D]
            k_max = self._page_k_max[layer_id][phys_page_ids]  # [num_pages, H, D]

            # query shape: [bs, num_q_heads, head_dim]
            # For GQA, average across query heads that share the same KV head.
            q_i = query[i]  # [num_q_heads, head_dim]
            num_q_heads = q_i.shape[0]
            num_kv_heads = k_min.shape[1]
            if num_q_heads != num_kv_heads:
                heads_per_group = num_q_heads // num_kv_heads
                q_i = q_i.view(num_kv_heads, heads_per_group, -1).mean(dim=1)

            # Bounding-box upper-bound: sum over heads and dims of
            # max(q * k_min, q * k_max)
            q_i = q_i.to(k_min.dtype)
            scores = torch.where(
                q_i.unsqueeze(0) >= 0,
                q_i.unsqueeze(0) * k_max,
                q_i.unsqueeze(0) * k_min,
            ).sum(dim=(-2, -1))  # [num_pages]

            # Number of pages to select (budget)
            budget = max(1, int(num_pages * self.token_budget_ratio))
            budget = min(budget, num_pages)

            topk_k = min(budget, num_pages)
            topk_indices = scores.topk(topk_k).indices

            # Always include the most recent pages
            recent_start = max(0, num_pages - self.num_recent_pages)
            recent_pages = torch.arange(
                recent_start, num_pages, device=self.device, dtype=torch.long
            )

            selected = torch.cat([topk_indices, recent_pages]).unique().sort().values
            # Return logical page indices (0-based within the request)
            all_indices.append(selected.to(torch.int32))
            all_lengths.append(selected.shape[0])

        # Pad and stack
        if all_lengths:
            max_sel = max(max(all_lengths), 1)
        else:
            max_sel = 1
        indices_tensor = torch.full(
            (bs, max_sel), -1, dtype=torch.int32, device=self.device
        )
        lengths_tensor = torch.zeros(bs, dtype=torch.int32, device=self.device)

        for i, sel in enumerate(all_indices):
            n = all_lengths[i]
            if n > 0:
                indices_tensor[i, :n] = sel[:n]
            lengths_tensor[i] = n

        return SelectionResult(
            granularity=Granularity.PAGE,
            selected_indices=indices_tensor,
            valid_lengths=lengths_tensor,
            sparse_mask=torch.tensor(sparse_flags, dtype=torch.bool, device=self.device),
        )

    # -- helpers ----------------------------------------------------------

    def _ensure_buffers(
        self, layer_id: int, num_kv_heads: int, head_dim: int
    ) -> None:
        if layer_id in self._page_k_min:
            return
        pool_pages = (
            self.config.policy_kwargs.get("max_pool_pages", 2048)
        )
        self._page_k_min[layer_id] = torch.zeros(
            pool_pages, num_kv_heads, head_dim,
            dtype=torch.float32, device=self.device,
        )
        self._page_k_max[layer_id] = torch.zeros(
            pool_pages, num_kv_heads, head_dim,
            dtype=torch.float32, device=self.device,
        )
