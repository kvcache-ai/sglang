"""GLM-5-Next KDA kernel adapter.

Unlike Kimi, GLM passes raw gate and raw beta logits for both prefill and
decode.  This adapter activates those inputs with GLM's bounded gate and then
reuses the unchanged Triton chunk-KDA implementation for prefill.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.linear.kernels.glm5_next_kda_ops import (
    glm5_next_safe_decode,
    glm5_next_safe_gate,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


class Glm5NextTritonKDAKernel(TritonKDAKernel):
    """Bounded-gate KDA kernel used only by GLM-5-Next."""

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: float,
        **kwargs,
    ) -> torch.Tensor:
        head_k_dim = q.shape[-1]
        if a.ndim == 2:
            a = a.unsqueeze(0)
        if a.ndim == 3:
            a = a.unflatten(-1, (-1, head_k_dim))
        if b.ndim == 2:
            b = b.unsqueeze(0)
        return glm5_next_safe_decode(
            A_log=A_log,
            raw_gate=a,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            q=q,
            k=k,
            v=v,
            raw_beta=b,
            state_source=ssm_states,
            state_indices=cache_indices,
            query_start_loc=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # GLM's model seam must pass both tensors before activation.  Normalize
        # the raw gate to [..., H*D]; glm5_next_safe_gate returns [..., H, D].
        head_k_dim = q.shape[-1]
        if g.ndim == 2:
            g = g.unsqueeze(0)
        if g.ndim >= 4 and g.shape[-1] == head_k_dim:
            g = g.flatten(-2)
        if beta.ndim == 2:
            beta = beta.unsqueeze(0)
        activated_gate = glm5_next_safe_gate(
            g,
            A_log,
            head_k_dim,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
        )
        activated_beta = beta.float().sigmoid()

        # Padding requests use -1.  KT's MambaPool reserves slot 0 as the
        # padding sentinel and allocates real requests from [1, size].
        safe_cache_indices = torch.where(cache_indices >= 0, cache_indices, 0).to(
            torch.int32
        )
        return super().extend(
            q,
            k,
            v,
            activated_gate,
            activated_beta,
            ssm_states=ssm_states,
            cache_indices=safe_cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )
