"""GLM-5-Next KDA kernel adapter.

Unlike Kimi, GLM passes raw gate and raw beta logits for both prefill and
decode.  This adapter activates those inputs with GLM's bounded gate and then
reuses the unchanged Triton chunk-KDA implementation for prefill.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.fla.kda import chunk_kda
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
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
        # The released GLM reference materializes sigmoid in b_proj's dtype
        # before chunk KDA widens beta to FP32.  In production b_proj is BF16,
        # so widening the raw logits first skips a real BF16 rounding boundary.
        activated_beta = beta.sigmoid().float()

        # Padding requests use -1.  KT's MambaPool reserves slot 0 as the
        # padding sentinel and allocates real requests from [1, size].
        safe_cache_indices = torch.where(cache_indices >= 0, cache_indices, 0).to(
            torch.int32
        )
        input_dtype = q.dtype
        # Match the released GLM recurrence: widen q/k/v together, normalize
        # q/k in FP32, and keep every tl.dot operand FP32.  Widening q/k alone
        # is not a valid Triton contract because v participates in the same
        # chunk-core dot products.
        normalized_q = l2norm_fwd(q.contiguous(), output_dtype=torch.float32)
        normalized_k = l2norm_fwd(k.contiguous(), output_dtype=torch.float32)
        output = chunk_kda(
            q=normalized_q,
            k=normalized_k,
            v=v.float(),
            g=activated_gate,
            beta=activated_beta,
            initial_state=ssm_states,
            initial_state_indices=safe_cache_indices,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=query_start_loc,
        )
        return output.to(input_dtype)
