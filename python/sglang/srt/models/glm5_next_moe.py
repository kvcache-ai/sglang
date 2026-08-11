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
    return F.silu(gate) * up


class Glm5NextSiluAndMul(nn.Module):
    """Clamp-aware SwiGLU with the existing KT CUDA kernel as its fast path."""

    def __init__(self, swiglu_limit: Optional[float]) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        if gate_up.shape[-1] % 2 != 0:
            raise ValueError(
                "GLM-5-Next SwiGLU expects an even gate/up width, got "
                f"{gate_up.shape[-1]}"
            )

        # The in-tree JIT kernel is NVIDIA-only and currently instantiates
        # kernels for fp16/bf16.  Keep CPU, ROCm, and unusual dtypes on the
        # device-agnostic reference path instead of importing CUDA code at
        # module import time.
        use_cuda_kernel = (
            self.swiglu_limit is not None
            and gate_up.is_cuda
            and torch.version.hip is None
            and gate_up.dtype in (torch.float16, torch.bfloat16)
        )
        if not use_cuda_kernel:
            return glm5_next_swiglu(gate_up, self.swiglu_limit)

        from sglang.jit_kernel.deepseek_v4 import silu_and_mul_clamp

        original_shape = gate_up.shape
        gate_up_2d = gate_up.reshape(-1, original_shape[-1])
        output_2d = gate_up.new_empty((gate_up_2d.shape[0], gate_up_2d.shape[1] // 2))
        silu_and_mul_clamp(
            gate_up_2d,
            output_2d,
            float(self.swiglu_limit),
        )
        return output_2d.view(*original_shape[:-1], original_shape[-1] // 2)


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
        return super().forward(hidden_states, *args, **kwargs)


__all__ = [
    "Glm5NextMLP",
    "Glm5NextMoE",
    "Glm5NextSiluAndMul",
    "glm5_next_swiglu",
]
