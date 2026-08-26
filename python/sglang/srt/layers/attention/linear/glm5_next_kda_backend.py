"""GLM-5-Next-only KDA attention backend.

Contract: ``a`` and ``b`` are raw gate and raw beta logits in both prefill and
decode.  This differs from Kimi prefill, whose model layer activates both
tensors before entering :class:`KDAAttnBackend`.  Keeping a dedicated backend
prevents the bounded GLM gate from changing Kimi's kernel selection, TF32,
autotune, or padding behavior.

This backend intentionally contains no CP, MTP, or CUDA-graph specialization.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from einops import rearrange

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.glm5_next_kda import (
    Glm5NextTritonKDAKernel,
)
from sglang.srt.layers.attention.linear.kernels.glm5_next_kda_ops import (
    restore_glm5_next_kda_padding,
    trim_glm5_next_kda_padding,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.utils import is_cpu, is_npu

if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Glm5NextKDAAttnBackend(MambaAttnBackendBase):
    """KDA backend consuming raw GLM gate/beta with an explicit lower bound."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.kernel = Glm5NextTritonKDAKernel()

    @staticmethod
    def _lower_bound(layer: RadixLinearAttention) -> float:
        lower_bound = getattr(layer, "lower_bound", None)
        if lower_bound is None:
            raise ValueError(
                "GLM-5-Next KDA requires an explicit negative gate lower_bound"
            )
        lower_bound = float(lower_bound)
        if lower_bound >= 0:
            raise ValueError(
                f"GLM-5-Next KDA lower_bound must be negative, got {lower_bound}"
            )
        return lower_bound

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        q_proj_states, k_proj_states, v_proj_states = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state.transpose(-1, -2),
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state.transpose(-1, -2),
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state.transpose(-1, -2),
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q = rearrange(q, "n (h d) -> 1 n h d", d=layer.head_q_dim)
        k = rearrange(k, "n (h d) -> 1 n h d", d=layer.head_k_dim)
        v = rearrange(v, "n (h d) -> 1 n h d", d=layer.head_v_dim)
        return self.kernel.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=self._lower_bound(layer),
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        mixed_qkv, a, b, physical_num_tokens = trim_glm5_next_kda_padding(
            mixed_qkv, a, b, query_start_loc
        )
        if mixed_qkv.shape[0] == 0:
            return mixed_qkv.new_zeros(
                (1, physical_num_tokens, layer.num_v_heads, layer.head_v_dim)
            )

        q_proj_states, k_proj_states, v_proj_states = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        has_initial_state = forward_batch.extend_prefix_lens > 0

        q = causal_conv1d_fn(
            q_proj_states.transpose(0, 1),
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=q_conv_state.transpose(-1, -2),
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        k = causal_conv1d_fn(
            k_proj_states.transpose(0, 1),
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=k_conv_state.transpose(-1, -2),
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)
        v = causal_conv1d_fn(
            v_proj_states.transpose(0, 1),
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=v_conv_state.transpose(-1, -2),
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q = rearrange(q, "n (h d) -> 1 n h d", d=layer.head_q_dim)
        k = rearrange(k, "n (h d) -> 1 n h d", d=layer.head_k_dim)
        v = rearrange(v, "n (h d) -> 1 n h d", d=layer.head_v_dim)
        output = self.kernel.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=self._lower_bound(layer),
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )
        return restore_glm5_next_kda_padding(output, physical_num_tokens)
