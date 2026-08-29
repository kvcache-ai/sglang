"""StreamingLLM — Sink + sliding-window sparsity via per-step masking.

StreamingLLM retains a fixed number of "sink" tokens (initial tokens that
absorb attention mass) plus a sliding window of the most recent tokens.
As new tokens arrive during decode, the window slides forward.

This implementation uses **PER_STEP** frequency: ``select()`` is called
every decode step to produce a mask covering only the sink tokens and
the most recent ``window_size`` tokens.  Tokens outside this range are
masked out (not visible to the attention kernel) but remain physically
allocated in the KV cache until the request finishes.

Physical eviction is intentionally not performed because the scheduler's
``Req.kv_committed_len`` bookkeeping is not accessible from the model
runner.  Freeing slots without updating ``kv_committed_len`` would cause
double-frees when the request finishes.

Reference: Xiao et al., "Efficient Streaming Language Models with
Attention Sinks", ICLR 2024.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.kse.base_policy import SparsityPolicy
from sglang.srt.layers.kse.registry import register_policy
from sglang.srt.layers.kse.types import Frequency, Granularity, SelectionResult

if TYPE_CHECKING:
    from sglang.srt.layers.kse.config import KSEConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_policy("streaming_llm")
class StreamingLLMPolicy(SparsityPolicy):
    """Sink + sliding-window with PER_STEP selection (masking only, no eviction)."""

    def __init__(self, config: KSEConfig, device: torch.device):
        self.config = config
        self.device = device
        self.num_sink_tokens = config.policy_kwargs.get("num_sink_tokens", 4)
        self.window_size = config.policy_kwargs.get("window_size", 1024)

    # -- SparsityPolicy interface -----------------------------------------

    def granularity(self) -> Granularity:
        return Granularity.TOKEN

    def frequency(self) -> Frequency:
        return Frequency.PER_STEP

    def select(
        self,
        query: Optional[torch.Tensor],
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> SelectionResult:
        """Select sink tokens + the most recent window_size tokens."""
        bs = seq_lens.shape[0]
        keep_count = self.num_sink_tokens + self.window_size

        all_indices = []
        all_lengths = []
        sparse_flags = []

        for i in range(bs):
            n = seq_lens[i].item()
            if n <= keep_count:
                idx = torch.arange(n, dtype=torch.int32, device=self.device)
                all_indices.append(idx)
                all_lengths.append(n)
                sparse_flags.append(False)
            else:
                sink = torch.arange(
                    self.num_sink_tokens, dtype=torch.int32, device=self.device
                )
                window = torch.arange(
                    n - self.window_size, n, dtype=torch.int32, device=self.device
                )
                selected = torch.cat([sink, window])
                all_indices.append(selected)
                all_lengths.append(selected.shape[0])
                sparse_flags.append(True)

        max_sel = max(all_lengths) if all_lengths else 1
        indices_tensor = torch.full(
            (bs, max_sel), -1, dtype=torch.int32, device=self.device
        )
        lengths_tensor = torch.zeros(bs, dtype=torch.int32, device=self.device)

        for i, sel in enumerate(all_indices):
            n = all_lengths[i]
            indices_tensor[i, :n] = sel[:n]
            lengths_tensor[i] = n

        return SelectionResult(
            granularity=Granularity.TOKEN,
            selected_indices=indices_tensor,
            valid_lengths=lengths_tensor,
            sparse_mask=torch.tensor(sparse_flags, dtype=torch.bool, device=self.device),
        )
