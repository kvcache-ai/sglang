"""Correctness fallback for GLM-5-Next's native zero-RoPE sparse MLA.

FlashInfer first exposed the H512 kernel ABI in 0.6.17, but its TRTLLM-GEN
implementation does not run on SM120 (for example RTX 5090).  Session AB also
keeps CUDA graphs disabled.  This small, model-local eager implementation
therefore provides a deterministic path without changing the pinned
FlashInfer dependency or any existing DeepSeek NSA backend.

The fallback deliberately processes a bounded number of query tokens at once.
That keeps the gathered ``[query, topk, 512]`` workspace bounded during short
correctness prefills.  A graph-safe high-performance kernel remains a Session-C
replacement, not an implicit promise of this reference implementation.
"""

from __future__ import annotations

import torch


def glm5_next_sparse_mla_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Compute sparse latent attention over a raw FP8/BF16 H512 cache.

    Args:
        query: ``[num_query_tokens, num_heads, 512]``.
        kv_cache: Any contiguous paged/flat layout whose last dimension is 512.
        indices: ``[num_query_tokens, sparse_capacity]`` physical token indices;
            invalid padding entries are ``-1``.
        sm_scale: Scale applied before softmax.
        chunk_size: Maximum query rows gathered at once.

    Returns:
        BF16 tensor ``[num_query_tokens, num_heads, 512]``.
    """

    if query.ndim != 3 or query.shape[-1] != 512:
        raise ValueError(
            "GLM-5-Next sparse MLA query must have shape [tokens, heads, 512], "
            f"got {tuple(query.shape)}"
        )
    if kv_cache.shape[-1] != 512:
        raise ValueError(
            "GLM-5-Next sparse MLA KV cache must have width 512, "
            f"got {kv_cache.shape[-1]}"
        )
    if indices.ndim != 2 or indices.shape[0] != query.shape[0]:
        raise ValueError(
            "GLM-5-Next sparse MLA indices must have shape [tokens, capacity], "
            f"got {tuple(indices.shape)} for {query.shape[0]} query tokens"
        )
    if indices.dtype != torch.int32:
        raise TypeError(
            f"GLM-5-Next sparse MLA indices must be int32, got {indices.dtype}"
        )
    if query.device != kv_cache.device or query.device != indices.device:
        raise ValueError("GLM-5-Next sparse MLA inputs must be on the same device")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    flat_kv = kv_cache.reshape(-1, 512)
    output = torch.empty(
        (*query.shape[:-1], 512),
        dtype=torch.bfloat16,
        device=query.device,
    )

    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        chunk_indices = indices[start:end]
        valid = chunk_indices >= 0
        if bool(torch.any(chunk_indices[valid] >= flat_kv.shape[0]).item()):
            bad_index = int(chunk_indices[valid].max().item())
            raise IndexError(
                "GLM-5-Next sparse MLA index exceeds the KV cache: "
                f"max index {bad_index}, cache tokens {flat_kv.shape[0]}"
            )

        nonempty = valid.any(dim=-1)
        safe_indices = chunk_indices.masked_fill(~valid, 0).to(torch.long)
        selected_kv = flat_kv[safe_indices].to(torch.bfloat16)
        query_chunk = query[start:end].to(torch.bfloat16)

        scores = torch.bmm(query_chunk, selected_kv.transpose(1, 2)).float()
        scores.mul_(float(sm_scale))
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        # Empty rows occur only for padded eager work.  Give softmax a finite
        # dummy slot and explicitly zero the resulting output below.
        if not bool(torch.all(nonempty).item()):
            scores[~nonempty] = 0.0

        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        chunk_output = torch.bmm(probabilities, selected_kv)
        chunk_output[~nonempty] = 0
        output[start:end] = chunk_output

    return output


__all__ = ["glm5_next_sparse_mla_reference"]
