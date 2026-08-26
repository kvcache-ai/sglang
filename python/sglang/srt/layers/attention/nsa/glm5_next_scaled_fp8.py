"""Block-scaled FP8 helpers for GLM-5-Next's zero-RoPE latent cache.

This module is intentionally model-local.  The released GLM-5-Next checkpoint
uses a 512-wide latent as both sparse-attention key and value.  A raw E4M3 cast
wastes most of the format's range, so exact GLM stores four FP32 descales (one
per 128 latent channels) next to the unchanged 512-byte FP8 cache row.
"""

from __future__ import annotations

import torch


GLM5_NEXT_LATENT_DIM = 512
GLM5_NEXT_LATENT_SCALE_GROUP_SIZE = 128
GLM5_NEXT_LATENT_SCALE_GROUPS = (
    GLM5_NEXT_LATENT_DIM // GLM5_NEXT_LATENT_SCALE_GROUP_SIZE
)
GLM5_NEXT_FP8_MAX = 448.0


def _validate_zero_rope_inputs(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> None:
    if q_nope.shape[-1] != GLM5_NEXT_LATENT_DIM:
        raise ValueError(
            f"GLM-5-Next scaled FP8 query must have width 512, got {q_nope.shape[-1]}"
        )
    if k_nope.shape[-1] != GLM5_NEXT_LATENT_DIM:
        raise ValueError(
            f"GLM-5-Next scaled FP8 key must have width 512, got {k_nope.shape[-1]}"
        )
    if q_rope.shape[-1] != 0 or k_rope.shape[-1] != 0:
        raise ValueError(
            "GLM-5-Next scaled FP8 latent helper accepts only zero-RoPE inputs"
        )
    if q_nope.device != q_rope.device:
        raise ValueError("GLM-5-Next query components must share a device")
    if k_nope.device != k_rope.device:
        raise ValueError("GLM-5-Next key components must share a device")


def glm5_next_quantize_latent_fp8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension into four independently scaled FP8 blocks."""

    if tensor.shape[-1] != GLM5_NEXT_LATENT_DIM:
        raise ValueError(
            f"GLM-5-Next latent quantization requires width 512, got {tensor.shape[-1]}"
        )
    if tensor.is_cuda:
        # Reuse NSA's graph-safe Triton quantizer and its established
        # dequant-scale convention: dequantized = fp8 * scale.
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        return act_quant(
            tensor.contiguous(),
            block_size=GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
            # A power-of-two descale is an exact exponent shift for E4M3.  It
            # preserves raw-cast rounding in the normal range while rescuing
            # blocks that would otherwise underflow or overflow.
            scale_fmt="ue8m0",
        )

    grouped = tensor.float().reshape(
        *tensor.shape[:-1],
        GLM5_NEXT_LATENT_SCALE_GROUPS,
        GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
    )
    # Match act_quant's lower bound.  It keeps all-zero rows finite and avoids
    # storing a zero descale in cache metadata.
    amax = grouped.abs().amax(dim=-1).clamp_min(1.0e-4)
    unrounded_scale = amax / GLM5_NEXT_FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(unrounded_scale)))
    quantized = (
        (grouped / scale.unsqueeze(-1))
        .clamp(-GLM5_NEXT_FP8_MAX, GLM5_NEXT_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .reshape_as(tensor)
    )
    return quantized, scale.to(torch.float32)


def glm5_next_dequantize_latent_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference dequantizer used by bounded prefill and CPU tests."""

    expected_scale_shape = (
        *tensor.shape[:-1],
        GLM5_NEXT_LATENT_SCALE_GROUPS,
    )
    if tensor.shape[-1] != GLM5_NEXT_LATENT_DIM:
        raise ValueError(
            "GLM-5-Next latent dequantization requires width 512, "
            f"got {tensor.shape[-1]}"
        )
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            "GLM-5-Next latent scale shape mismatch: expected "
            f"{expected_scale_shape}, got {tuple(scale.shape)}"
        )
    grouped = tensor.float().reshape(
        *tensor.shape[:-1],
        GLM5_NEXT_LATENT_SCALE_GROUPS,
        GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
    )
    return (grouped * scale.float().unsqueeze(-1)).reshape_as(tensor).to(dtype)


def glm5_next_mla_prepare_scaled_fp8_no_rope(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare BF16 Q plus scaled-FP8 K for exact GLM sparse attention.

    Query is ephemeral and the model-local kernel accepts BF16 directly, so
    quantizing it would lose precision without saving persistent memory.  Only
    the cached key/value latent is quantized and accompanied by four descales.
    """

    _validate_zero_rope_inputs(q_nope, q_rope, k_nope, k_rope)
    k_fp8, k_scale = glm5_next_quantize_latent_fp8(k_nope)
    # Preserve the existing zero-width tensor contract used by the MLA writer.
    k_rope_fp8 = k_rope.to(torch.float8_e4m3fn)
    return q_nope.contiguous(), k_fp8, k_rope_fp8, k_scale


__all__ = [
    "GLM5_NEXT_LATENT_DIM",
    "GLM5_NEXT_LATENT_SCALE_GROUP_SIZE",
    "GLM5_NEXT_LATENT_SCALE_GROUPS",
    "glm5_next_dequantize_latent_fp8",
    "glm5_next_mla_prepare_scaled_fp8_no_rope",
    "glm5_next_quantize_latent_fp8",
]
