"""Flat 2-D mHC functional front-ends used by GLM-5-Next.

The existing KT TileLang kernels operate on ``[tokens, hc_mult, hidden]``.
This module keeps the GLM runtime state flat as ``[tokens, hc_mult * hidden]``
and supplies a pure-PyTorch reference.  The optimized backend is imported only
when its existing environment flag is enabled, so importing this module does
not load TileLang or change any pre-existing model path.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def _use_tilelang(env_name: str) -> bool:
    return os.environ.get(env_name, "0").strip().lower() in _TRUE_ENV_VALUES


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """Replicate ``[tokens, hidden]`` into flat ``[tokens, n * hidden]``."""
    assert x.dim() == 2, f"x must be 2-D, got {tuple(x.shape)}"
    assert n > 0, f"n must be positive, got {n}"
    return x.repeat(1, n)


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """Average flat ``[tokens, n * hidden]`` into ``[tokens, hidden]``."""
    assert x.dim() == 2, f"x must be 2-D, got {tuple(x.shape)}"
    assert n > 0, f"n must be positive, got {n}"
    assert x.shape[-1] % n == 0, (
        f"flat hidden width {x.shape[-1]} must be divisible by n={n}"
    )
    return x.unflatten(-1, (n, -1)).mean(dim=-2)


def _validate_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    sinkhorn_repeat: int,
) -> tuple[int, int, int]:
    assert residual.dim() == 3, (
        f"residual must be [tokens, hc_mult, hidden], got {tuple(residual.shape)}"
    )
    tokens, hc_mult, hidden_size = residual.shape
    mix_width = hc_mult * (2 + hc_mult)
    assert fn.shape == (mix_width, hc_mult * hidden_size), (
        f"fn must be {(mix_width, hc_mult * hidden_size)}, got {tuple(fn.shape)}"
    )
    assert hc_scale.shape == (3,), f"hc_scale must be (3,), got {tuple(hc_scale.shape)}"
    assert hc_base.shape == (mix_width,), (
        f"hc_base must be ({mix_width},), got {tuple(hc_base.shape)}"
    )
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32
    assert sinkhorn_repeat >= 1
    return tokens, hc_mult, hidden_size


def _mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Torch mHC pre reference.

    Returns ``post [T,N,1]``, ``comb [T,N,N]`` and ``layer_input [T,H]``.
    Numerically sensitive normalization and Sinkhorn updates run in FP32.
    """
    tokens, hc_mult, hidden_size = _validate_pre_inputs(
        residual, fn, hc_scale, hc_base, sinkhorn_repeat
    )
    output_dtype = residual.dtype

    residual_fp32 = residual.reshape(tokens, hc_mult * hidden_size).float()
    reciprocal_rms = torch.rsqrt(
        residual_fp32.square().mean(-1, keepdim=True) + rms_eps
    )
    mixes = F.linear(residual_fp32, fn) * reciprocal_rms

    pre_raw = mixes[:, :hc_mult]
    post_raw = mixes[:, hc_mult : 2 * hc_mult]
    comb_raw = mixes[:, 2 * hc_mult :].reshape(tokens, hc_mult, hc_mult)
    pre_base = hc_base[:hc_mult]
    post_base = hc_base[hc_mult : 2 * hc_mult]
    comb_base = hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)

    pre = torch.sigmoid(pre_raw * hc_scale[0] + pre_base) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(post_raw * hc_scale[1] + post_base)
    comb = comb_raw * hc_scale[2] + comb_base

    comb = comb.softmax(-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(output_dtype)
    return post.unsqueeze(-1), comb, layer_input


def _mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Pure-Torch mHC post reference.

    ``x`` is ``[T,H]`` and ``residual`` is ``[T,N,H]``.  The result is
    ``[T,N,H]`` and follows the dtype of ``x``.
    """
    assert x.dim() == 2 and residual.dim() == 3
    tokens, hc_mult, hidden_size = residual.shape
    assert x.shape == (tokens, hidden_size)
    assert post_layer_mix.shape == (tokens, hc_mult, 1)
    assert comb_res_mix.shape == (tokens, hc_mult, hc_mult)

    out = post_layer_mix * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    return out.type_as(x)


@torch._dynamo.disable
def _mhc_pre_dispatch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Dispatch to KT's existing TileLang kernel only when explicitly enabled."""
    del norm_eps  # KT's current mHC kernel does not fuse the following RMSNorm.
    if not _use_tilelang("SGLANG_OPT_USE_TILELANG_MHC_PRE"):
        post_mix, comb_mix, layer_input = _mhc_pre_torch(
            residual=residual,
            fn=fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        return post_mix, comb_mix, layer_input, False

    from sglang.srt.layers.mhc import mhc_pre as tilelang_mhc_pre

    post_mix, comb_mix, layer_input = tilelang_mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    # ``norm_weight`` is intentionally not consumed: callers apply it when
    # norm_fused is False.  This preserves the current KT kernel contract.
    del norm_weight
    return post_mix, comb_mix, layer_input, False


@torch._dynamo.disable
def _mhc_post_dispatch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if not _use_tilelang("SGLANG_OPT_USE_TILELANG_MHC_POST"):
        return _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)

    from sglang.srt.layers.mhc import mhc_post as tilelang_mhc_post

    return tilelang_mhc_post(x, residual, post_layer_mix, comb_res_mix)


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult_value: float = 2.0,
    hc_norm_weight: torch.Tensor | None = None,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Run mHC pre on flat ``x [T, N*H]``.

    Returns ``layer_input [T,H]``, flattened ``h_res [T,N*N]``, flattened
    ``h_post [T,N]``, and whether the following normalization was fused.
    """
    assert x.dim() == 2, f"x must be 2-D, got {tuple(x.shape)}"
    assert hc_mult > 0 and x.shape[1] % hc_mult == 0
    tokens, total_width = x.shape
    hidden_size = total_width // hc_mult
    if x.numel() == 0:
        return (
            x.new_zeros((tokens, hidden_size)),
            torch.zeros(
                (tokens, hc_mult * hc_mult),
                device=x.device,
                dtype=torch.float32,
            ),
            torch.zeros((tokens, hc_mult), device=x.device, dtype=torch.float32),
            False,
        )

    fn = hc_fn if hc_norm_weight is None else hc_fn * hc_norm_weight
    residual = x.reshape(tokens, hc_mult, hidden_size)
    post_mix, comb_mix, layer_input, norm_fused = _mhc_pre_dispatch(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        hc_post_mult_value=post_mult_value,
        sinkhorn_repeat=sinkhorn_iters,
        norm_weight=out_norm_weight,
        norm_eps=out_norm_eps,
    )
    return (
        layer_input,
        comb_mix.reshape(tokens, hc_mult * hc_mult),
        post_mix.reshape(tokens, hc_mult),
        norm_fused,
    )


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    hc_mult: int,
) -> torch.Tensor:
    """Run mHC post and return flat ``[T, N*H]`` state."""
    assert x.dim() == 2 and residual.dim() == 2
    tokens, hidden_size = x.shape
    assert residual.shape == (tokens, hc_mult * hidden_size)
    assert h_post.shape == (tokens, hc_mult)
    assert h_res.shape == (tokens, hc_mult * hc_mult)
    if tokens == 0:
        return x.new_zeros((tokens, hc_mult * hidden_size))

    out = _mhc_post_dispatch(
        x,
        residual.reshape(tokens, hc_mult, hidden_size),
        h_post.reshape(tokens, hc_mult, 1),
        h_res.reshape(tokens, hc_mult, hc_mult),
    )
    return out.reshape(tokens, hc_mult * hidden_size)
