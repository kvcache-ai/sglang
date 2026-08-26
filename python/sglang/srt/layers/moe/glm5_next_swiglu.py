"""GLM-5-Next-private SwiGLU with the HF BF16 rounding contract."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _glm5_next_silu_to_bf16_kernel(
    gate_up_ptr,
    output_ptr,
    numel,
    HALF_WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    rows = offsets // HALF_WIDTH
    columns = offsets - rows * HALF_WIDTH
    gate_offsets = rows * (2 * HALF_WIDTH) + columns

    gate = tl.load(gate_up_ptr + gate_offsets, mask=mask).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
    activated_gate = gate * tl.sigmoid(gate)

    # output_ptr is BF16 for the exact GLM runtime.  This store is the first
    # required rounding boundary and is intentionally consumed by a separate
    # kernel below.
    tl.store(output_ptr + offsets, activated_gate, mask=mask)


@triton.jit
def _glm5_next_bf16_mul_kernel(
    gate_up_ptr,
    output_ptr,
    numel,
    HALF_WIDTH: tl.constexpr,
    LIMIT: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    rows = offsets // HALF_WIDTH
    columns = offsets - rows * HALF_WIDTH
    up_offsets = rows * (2 * HALF_WIDTH) + HALF_WIDTH + columns

    # Reloading output_ptr observes the BF16 value materialized by stage one.
    # The in-place store below is safe because every program owns its elements.
    activated_gate = tl.load(output_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(gate_up_ptr + up_offsets, mask=mask).to(tl.float32)
    if HAS_LIMIT:
        up = tl.maximum(up, -LIMIT)
        up = tl.minimum(up, LIMIT)
    tl.store(output_ptr + offsets, activated_gate * up, mask=mask)


def _reference_two_round_swiglu(
    gate_up: torch.Tensor,
    swiglu_limit: Optional[float],
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    if swiglu_limit is not None:
        limit = float(swiglu_limit)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    activated_gate = F.silu(gate)
    return activated_gate * up


def glm5_next_hf_two_round_swiglu(
    gate_up: torch.Tensor,
    swiglu_limit: Optional[float],
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply GLM SwiGLU with a materialized BF16 SiLU intermediate.

    The released Transformers model evaluates BF16 ``silu(gate)`` and then a
    BF16 multiply.  A single fused pointwise kernel instead keeps SiLU in FP32
    and rounds only after the multiply.  On CUDA BF16, these two fixed Triton
    launches use ``output`` first as the SiLU materialization and then as the
    final result, so no additional activation workspace is needed.  Other
    devices and dtypes use the literal PyTorch expression.
    """

    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"GLM-5-Next SwiGLU expects an even gate/up width, got {gate_up.shape[-1]}"
        )

    output_shape = (*gate_up.shape[:-1], gate_up.shape[-1] // 2)
    if output is not None:
        if output.shape != output_shape:
            raise ValueError(
                "GLM-5-Next SwiGLU output shape mismatch: expected "
                f"{output_shape}, got {tuple(output.shape)}"
            )
        if output.device != gate_up.device or output.dtype != gate_up.dtype:
            raise ValueError(
                "GLM-5-Next SwiGLU output must match input device and dtype"
            )

    use_cuda_kernel = (
        gate_up.is_cuda
        and torch.version.hip is None
        and gate_up.dtype == torch.bfloat16
        and gate_up.is_contiguous()
        and (output is None or output.is_contiguous())
    )
    if not use_cuda_kernel:
        result = _reference_two_round_swiglu(gate_up, swiglu_limit)
        if output is None:
            return result
        output.copy_(result)
        return output

    if output is None:
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
    if output.numel() == 0:
        return output

    gate_up_2d = gate_up.view(-1, gate_up.shape[-1])
    output_2d = output.view(-1, output.shape[-1])
    half_width = output_2d.shape[-1]
    numel = output_2d.numel()
    limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
    grid = (triton.cdiv(numel, 256),)

    _glm5_next_silu_to_bf16_kernel[grid](
        gate_up_2d,
        output_2d,
        numel,
        HALF_WIDTH=half_width,
        LIMIT=limit,
        HAS_LIMIT=swiglu_limit is not None,
        BLOCK_SIZE=256,
    )
    _glm5_next_bf16_mul_kernel[grid](
        gate_up_2d,
        output_2d,
        numel,
        HALF_WIDTH=half_width,
        LIMIT=limit,
        HAS_LIMIT=swiglu_limit is not None,
        BLOCK_SIZE=256,
    )
    return output


__all__ = ["glm5_next_hf_two_round_swiglu"]
