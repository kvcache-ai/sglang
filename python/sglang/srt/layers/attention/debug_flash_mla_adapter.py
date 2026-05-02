# DeepSeek V4 Flash attention dispatcher.
#
# Two backends:
#   "kernel"  -> upstream `flash_mla` PyPI package. The installed wheel's
#                cuda*.so contains only `.target sm_90a` (Hopper); any other
#                capability raises "Unsupported architecture" the first time
#                the CUDA op runs. Capabilities outside the whitelist are
#                transparently routed to the Triton fallback.
#   "triton"  -> portable Triton kernel ported from vLLM PR #40929. Works on
#                any arch Triton supports (>= SM_80).
#
# This is a sglang-side change (sglang 本身), not kt-sglang coupling: the V4
# quantizer that produces the FP8/RoPE/ue8m0-packed KV layout already lives
# in sglang main (PR #23600), and no upstream FlashMLA build understands that
# layout on non-Hopper arches, so the fallback has to live next to the
# entrypoint.
import functools
from typing import Any, Tuple

import torch


# Capabilities for which the upstream flash_mla wheel ships a CUDA binary.
# Verified by `strings cuda*.so | grep '\.target'` on the installed package:
# only `sm_90a` is present. Any other capability must take the Triton path.
_FLASHMLA_CUDA_CAPS = {(9, 0)}


@functools.lru_cache(maxsize=1)
def _device_capability() -> Tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


def should_use_triton_fallback() -> bool:
    # Non-whitelist capability -> route to Triton. Used by the V4 metadata
    # factories to skip flash_mla.get_mla_metadata() on arches the wheel
    # cannot run on, and by flash_mla_with_kvcache_entrypoint below to
    # transparently swap the kernel implementation.
    return _device_capability() not in _FLASHMLA_CUDA_CAPS


# Backward-compat alias (was private until consumed by metadata factories).
_should_use_triton_fallback = should_use_triton_fallback


def _v4_triton_decode_dispatch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    is_fp8_kvcache: bool,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    attn_sink: torch.Tensor,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    **_unused: Any,
) -> Tuple[torch.Tensor, None]:
    """Translate V4 backend's input_dict into the Triton kernel signature.

    sglang call site (deepseek_v4_backend_radix.py:1107) does:
        o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]
        o = o.squeeze(1)
    so we must return a tuple whose first element has shape
    [batch, 1, num_heads, head_dim_v].
    """
    from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
        decode_sparse_attention_triton,
    )

    assert is_fp8_kvcache, "Triton V4 fallback only handles FP8 KV cache"
    assert head_dim_v == 512, f"V4 head_dim_v must be 512, got {head_dim_v}"

    # q : sglang passes [batch, 1, num_heads, 512] (line 1071: q.unsqueeze(1)).
    q_2d = q.squeeze(1) if q.ndim == 4 else q
    num_tokens, num_heads, head_dim = q_2d.shape

    # k_cache : sglang reshapes the underlying uint8 page buffer to
    # [num_pages, swa_window=128, 1, k_cache_total_dim=584] (line 1039).
    swa_cache = k_cache.squeeze(2) if k_cache.ndim == 4 else k_cache
    extra_cache = (
        extra_k_cache.squeeze(2)
        if extra_k_cache is not None and extra_k_cache.ndim == 4
        else extra_k_cache
    )

    # The kernel reinterprets swa_cache as fp8 / bf16 / u8 via .view(); this
    # only works if the underlying tensor is uint8. The V4 memory pool stores
    # buf as uint8 (deepseekv4_memory_pool.py:108), so we get this for free.
    if swa_cache.dtype != torch.uint8:
        swa_cache = swa_cache.view(torch.uint8)
    if extra_cache is not None and extra_cache.dtype != torch.uint8:
        extra_cache = extra_cache.view(torch.uint8)

    # The Triton kernel expects 1-D length tensors of shape [num_tokens].
    swa_lens = topk_length.reshape(-1)[:num_tokens].contiguous()
    extra_lens = (
        extra_topk_length.reshape(-1)[:num_tokens].contiguous()
        if extra_topk_length is not None
        else None
    )

    # The kernel emits bf16; allocate accordingly.
    out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=torch.bfloat16,
        device=q_2d.device,
    )
    if num_tokens == 0:
        return (out.unsqueeze(1), None)

    decode_sparse_attention_triton(
        q=q_2d if q_2d.dtype == torch.bfloat16 else q_2d.to(torch.bfloat16),
        swa_cache=swa_cache,
        swa_indices=indices,
        swa_lens=swa_lens,
        scale=float(softmax_scale),
        attn_sink=attn_sink,
        out=out,
        extra_cache=extra_cache,
        extra_indices=extra_indices_in_kvcache,
        extra_lens=extra_lens,
    )
    return (out.unsqueeze(1), None)


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if backend == "kernel":
        # Auto-fall back on architectures the upstream CUDA kernel does not
        # cover. Set SGLANG_FLASHMLA_BACKEND_OVERRIDE=triton to force-bypass even
        # on supported arches (useful for numerical comparison).
        if _should_use_triton_fallback():
            return _v4_triton_decode_dispatch(**kwargs)
        import flash_mla

        return flash_mla.flash_mla_with_kvcache(**kwargs)

    if backend == "triton":
        return _v4_triton_decode_dispatch(**kwargs)

    raise ValueError(
        f"unsupported backend {backend!r}; expected 'kernel' or 'triton'"
    )
