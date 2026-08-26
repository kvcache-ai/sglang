"""GLM-5-Next MLP and MoE specializations.

GLM-5-Next uses the DeepSeek-style packed MLP/checkpoint layout, but its
SwiGLU activation clamps the gate and up projections *before* applying SiLU.
Keeping that small semantic difference here lets the model reuse the mature
DeepSeek TP/EP, FP8, shared-expert, and weight-loading paths without changing
their behavior for existing models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE


def glm5_next_swiglu(
    gate_up: torch.Tensor, swiglu_limit: Optional[float]
) -> torch.Tensor:
    """Reference GLM-5-Next SwiGLU, usable on every torch device.

    The checkpoint contract is ``silu(min(gate, limit)) * clamp(up, -limit,
    limit)``.  In particular, the gate has no lower clamp and the clamp is
    applied before SiLU.  ``None`` retains ordinary SwiGLU semantics.
    """

    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"GLM-5-Next SwiGLU expects an even gate/up width, got {gate_up.shape[-1]}"
        )

    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None:
        limit = float(swiglu_limit)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    # Keep the activation as a named tensor.  For BF16 checkpoints this is a
    # real model semantic: Transformers materializes the BF16 SiLU result
    # before the BF16 multiply.  The generic fused CUDA kernel computes both
    # operations in FP32 and rounds only once, which measurably changes GLM's
    # hidden states even when given identical inputs.
    activated_gate = F.silu(gate)
    return activated_gate * up


class Glm5NextSiluAndMul(nn.Module):
    """Clamp-aware SwiGLU preserving GLM's BF16 intermediate rounding."""

    def __init__(self, swiglu_limit: Optional[float]) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if not gate_up.is_cuda:
            return glm5_next_swiglu(gate_up, self.swiglu_limit)

        from sglang.srt.layers.moe.glm5_next_swiglu import (
            glm5_next_hf_two_round_swiglu,
        )

        # Dense and non-fused shared experts use the same private primitive as
        # routed experts.  It materializes HF's BF16 SiLU boundary while
        # leaving every generic activation kernel unchanged.
        return glm5_next_hf_two_round_swiglu(gate_up, self.swiglu_limit)


class Glm5NextMLP(DeepseekV2MLP):
    """DeepSeek packed/parallel MLP with GLM's clamp-aware activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        swiglu_limit: Optional[float] = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.swiglu_limit = swiglu_limit
        self.act_fn = Glm5NextSiluAndMul(swiglu_limit)


class Glm5NextMoE(DeepseekV2MoE):
    """GLM routing/activation semantics on the existing DeepSeek MoE stack.

    ``layer_idx`` is accepted as a compatibility alias for the current GLM
    decoder factory; upstream-style callers can use ``layer_id``.
    """

    def __init__(
        self,
        config,
        layer_id: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
        *,
        layer_idx: Optional[int] = None,
    ) -> None:
        if layer_id is None:
            layer_id = layer_idx
        elif layer_idx is not None and layer_idx != layer_id:
            raise ValueError(
                f"Conflicting GLM MoE layer ids: layer_id={layer_id}, "
                f"layer_idx={layer_idx}"
            )
        if layer_id is None:
            raise TypeError("Glm5NextMoE requires layer_id (or layer_idx)")

        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            glm5_next_hf_two_round_swiglu=True,
        )
        self.layer_idx = layer_id
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self._validate_kt_cpu_expert_activation()

        # Official GLM uses grouped noaux_tc routing even when n_group and
        # topk_group are both one.  The target DeepSeek base otherwise changes
        # this to the ungrouped path for 1/1, so restore the exact contract in
        # the model-local subclass.
        if not self.is_hash:
            self.use_grouped_topk = True
            self.topk.topk_config.use_grouped_topk = True

        # Routed (and fused shared) experts already receive swiglu_limit from
        # DeepseekV2MoE.  Only the non-fused shared-expert MLP needs its
        # activation specialized; its projections, prefixes, FP8 state, and TP
        # settings remain untouched.
        if hasattr(self, "shared_experts"):
            self.shared_experts.swiglu_limit = self.swiglu_limit
            self.shared_experts.act_fn = Glm5NextSiluAndMul(self.swiglu_limit)

    def _validate_kt_cpu_expert_activation(self) -> None:
        """Reject KT CPU formats whose kernel ABI discards swiglu_limit."""

        if self.swiglu_limit is None:
            return
        quant_method = getattr(self.experts, "quant_method", None)
        kt_config = getattr(quant_method, "kt_config", None)
        if kt_config is None:
            return

        method = (getattr(kt_config, "method", None) or "").upper()
        gpu_experts_mask = getattr(quant_method, "gpu_experts_mask", None)
        if gpu_experts_mask is not None:
            if isinstance(gpu_experts_mask, torch.Tensor):
                all_experts_on_gpu = bool(gpu_experts_mask.all().item())
            else:
                all_experts_on_gpu = all(bool(value) for value in gpu_experts_mask)
            if all_experts_on_gpu:
                return

        if method == "FP8":
            return

        raise NotImplementedError(
            "GLM-5-Next KT CPU experts require --kt-method FP8 for the "
            "checkpoint's block-E4M3 weights and FP32 [128, 128] scales; "
            f"the current {method or 'unspecified'} KT kernel discards "
            "swiglu_limit and would produce incorrect activations"
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if hidden_states.device.type == "cpu" and self.swiglu_limit is not None:
            raise NotImplementedError(
                "GLM-5-Next CPU MoE requires clamp-before-SiLU support in both "
                "sgl_kernel.fused_experts_cpu and sgl_kernel.shared_expert_cpu; "
                "their current AMX ABI has no swiglu_limit argument"
            )

        # DeepseekV2MoE.forward does not pass ForwardBatch through its normal
        # (non-A2A) dispatcher ABI.  Expose only the mode and the exact GLM
        # image marker, temporarily and only to the KT wrapper.  Text EXTEND
        # uses the required layerwise pipeline, while image EXTEND deliberately
        # uses the existing CPU/hybrid path and CUDA-graph DECODE/IDLE bypasses
        # both prefill routes.
        quant_method = getattr(self.experts, "quant_method", None)
        kt_config = getattr(quant_method, "kt_config", None)
        layerwise_required = (
            kt_config is not None
            and bool(getattr(kt_config, "is_glm5_next", False))
            and (getattr(kt_config, "method", "") or "").upper() == "FP8"
            and (getattr(kt_config, "gpu_prefill_token_threshold", 0) or 0) > 0
        )
        if not layerwise_required:
            return super().forward(hidden_states, *args, **kwargs)

        forward_batch = kwargs.get("forward_batch")
        if forward_batch is None and args:
            forward_batch = args[0]
        if forward_batch is None or not hasattr(forward_batch, "forward_mode"):
            raise RuntimeError(
                "GLM-5-Next required FP8 layerwise routing needs "
                "ForwardBatch.forward_mode"
            )

        missing = object()
        previous_mode = getattr(
            quant_method, "_glm5_next_forward_mode", missing
        )
        previous_image_marker = getattr(
            quant_method, "_glm5_next_has_image_inputs", missing
        )
        previous_hybrid_marker = getattr(
            quant_method, "_glm5_next_force_hybrid_prefill", missing
        )
        quant_method._glm5_next_forward_mode = forward_batch.forward_mode
        quant_method._glm5_next_has_image_inputs = bool(
            getattr(forward_batch, "glm5_next_has_image_inputs", False)
        )
        quant_method._glm5_next_force_hybrid_prefill = bool(
            getattr(forward_batch, "glm5_next_force_hybrid_prefill", False)
        )
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            if previous_mode is missing:
                delattr(quant_method, "_glm5_next_forward_mode")
            else:
                quant_method._glm5_next_forward_mode = previous_mode
            if previous_image_marker is missing:
                delattr(quant_method, "_glm5_next_has_image_inputs")
            else:
                quant_method._glm5_next_has_image_inputs = previous_image_marker
            if previous_hybrid_marker is missing:
                delattr(quant_method, "_glm5_next_force_hybrid_prefill")
            else:
                quant_method._glm5_next_force_hybrid_prefill = previous_hybrid_marker


__all__ = [
    "Glm5NextMLP",
    "Glm5NextMoE",
    "Glm5NextSiluAndMul",
    "glm5_next_swiglu",
]
