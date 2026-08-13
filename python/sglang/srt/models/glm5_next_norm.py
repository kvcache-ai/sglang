"""GLM-5-Next checkpoint-native RMSNorm arithmetic."""

from __future__ import annotations

import torch
from torch import nn


class Glm5NextRMSNorm(nn.Module):
    """Apply the exact BF16 rounding boundary used by the released model.

    The checkpoint implementation casts the FP32-normalized activation back
    to the model dtype before multiplying by the learned weight.  SGLang's
    shared optimized RMSNorm multiplies first and casts the product, which is
    a real BF16 numerical difference.  This implementation stays GLM-local so
    existing KT model paths retain their established numerics.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


__all__ = ["Glm5NextRMSNorm"]
