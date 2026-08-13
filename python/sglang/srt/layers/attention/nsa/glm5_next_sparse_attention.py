"""GLM-5-Next native zero-RoPE sparse MLA.

FlashInfer first exposed the H512 kernel ABI in 0.6.17, but its TRTLLM-GEN
implementation does not run on SM120 (for example RTX 5090).  This small,
model-local implementation provides a deterministic graph-safe path without
changing the pinned FlashInfer dependency or any existing DeepSeek NSA backend.

CUDA decode uses a BF16-query/scaled-FP8-KV Triton kernel with online softmax.
It consumes the model's physical token indices directly, has fixed launch
geometry for graph batch sizes 1/2/4, and does not materialize
``[query, topk, 512]``.  The small PyTorch implementation remains the CPU
correctness oracle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


GLM5_NEXT_SPARSE_HEAD_DIM = 512
GLM5_NEXT_LATENT_SCALE_GROUP_SIZE = 128
GLM5_NEXT_LATENT_SCALE_GROUPS = (
    GLM5_NEXT_SPARSE_HEAD_DIM // GLM5_NEXT_LATENT_SCALE_GROUP_SIZE
)
_TRITON_TOKEN_BLOCK = 16


@triton.jit
def _glm5_next_sparse_mla_decode_kernel(
    query_ptr,
    kv_ptr,
    kv_scale_ptr,
    indices_ptr,
    output_ptr,
    sm_scale: tl.float32,
    num_kv_tokens: tl.int64,
    topk: tl.int32,
    stride_query_row: tl.constexpr,
    stride_query_head: tl.constexpr,
    stride_query_dim: tl.constexpr,
    stride_kv_row: tl.constexpr,
    stride_kv_dim: tl.constexpr,
    stride_kv_scale_row: tl.constexpr,
    stride_kv_scale_group: tl.constexpr,
    stride_indices_row: tl.constexpr,
    stride_indices_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_head: tl.constexpr,
    stride_output_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    USE_KV_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    dims = tl.arange(0, HEAD_DIM)
    query = tl.load(
        query_ptr
        + row * stride_query_row
        + head * stride_query_head
        + dims * stride_query_dim
    ).to(tl.float32)

    running_max: tl.float32 = -1.0e30
    running_sum: tl.float32 = 0.0
    accumulator = tl.zeros([HEAD_DIM], tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)

    # ``topk`` is a static tensor width at graph capture (2051 for the released
    # checkpoint), while validity remains data-driven through the -1 sentinel.
    for start in range(0, topk, BLOCK_T):
        cols = start + token_offsets
        in_bounds = cols < topk
        physical = tl.load(
            indices_ptr + row * stride_indices_row + cols * stride_indices_col,
            mask=in_bounds,
            other=-1,
        ).to(tl.int64)
        valid = in_bounds & (physical >= 0) & (physical < num_kv_tokens)
        safe_physical = tl.where(valid, physical, 0)

        kv = tl.load(
            kv_ptr
            + safe_physical[:, None] * stride_kv_row
            + dims[None, :] * stride_kv_dim,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        if USE_KV_SCALE:
            kv_scale = tl.load(
                kv_scale_ptr
                + safe_physical[:, None] * stride_kv_scale_row
                + (dims[None, :] // SCALE_GROUP_SIZE) * stride_kv_scale_group,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            kv *= kv_scale
        scores = tl.sum(kv * query[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, -1.0e30)

        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        probabilities = tl.where(valid, probabilities, 0.0)

        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        accumulator = accumulator * old_scale + tl.sum(
            probabilities[:, None] * kv, axis=0
        )
        running_max = next_max

    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    output = accumulator / denominator
    tl.store(
        output_ptr
        + row * stride_output_row
        + head * stride_output_head
        + dims * stride_output_dim,
        output.to(tl.bfloat16),
    )


def _glm5_next_sparse_mla_cuda(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the graph-safe BF16-Q/raw-or-scaled-FP8-KV decode kernel."""

    flat_kv = kv_cache.reshape(-1, GLM5_NEXT_SPARSE_HEAD_DIM)
    query_3d = query.contiguous()
    indices_2d = indices.contiguous()
    use_kv_scale = kv_scale is not None
    if use_kv_scale:
        kv_scale_2d = kv_scale.contiguous()
    else:
        # Triton requires a pointer argument even when the constexpr branch is
        # dead.  Reuse a live input pointer; its strides are never consumed.
        kv_scale_2d = flat_kv
    output = torch.empty_like(query_3d, dtype=torch.bfloat16)
    grid = (query_3d.shape[0], query_3d.shape[1])
    _glm5_next_sparse_mla_decode_kernel[grid](
        query_3d,
        flat_kv,
        kv_scale_2d,
        indices_2d,
        output,
        float(sm_scale),
        flat_kv.shape[0],
        indices_2d.shape[1],
        query_3d.stride(0),
        query_3d.stride(1),
        query_3d.stride(2),
        flat_kv.stride(0),
        flat_kv.stride(1),
        kv_scale_2d.stride(0),
        kv_scale_2d.stride(1),
        indices_2d.stride(0),
        indices_2d.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        HEAD_DIM=GLM5_NEXT_SPARSE_HEAD_DIM,
        SCALE_GROUP_SIZE=GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
        USE_KV_SCALE=use_kv_scale,
        BLOCK_T=_TRITON_TOKEN_BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return output


def glm5_next_sparse_mla_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    chunk_size: int = 8,
    use_cuda_decode_kernel: bool = True,
    kv_scale: torch.Tensor | None = None,
    current_chunk_kv: torch.Tensor | None = None,
    current_chunk_locs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute sparse latent attention over a raw/scaled FP8 or BF16 cache.

    Args:
        query: ``[num_query_tokens, num_heads, 512]``.
        kv_cache: Any contiguous paged/flat layout whose last dimension is 512.
        indices: ``[num_query_tokens, sparse_capacity]`` physical token indices;
            invalid padding entries are ``-1``.
        sm_scale: Scale applied before softmax.
        chunk_size: Maximum query rows gathered at once.
        use_cuda_decode_kernel: Use the graph-safe raw-FP8 Triton kernel for
            CUDA decode.  EXTEND/prefill must pass ``False`` so its BF16
            matmul/softmax rounding remains aligned with the reference model.
        kv_scale: Optional FP32 ``[physical_tokens, 4]`` block descales.
            It uses 128 channels/group.  Production exact GLM keeps ephemeral
            Q in BF16 and scales only its persistent FP8 latent cache.
        current_chunk_kv: Optional pre-quantization BF16 latent rows for the
            current EXTEND chunk.  Persistent history is still read from the
            scaled-FP8 cache; only selected physical rows whose indices occur
            in ``current_chunk_locs`` use these values.  Decode must not pass
            this argument.
        current_chunk_locs: Physical cache locations corresponding one-to-one
            with ``current_chunk_kv``.

    Returns:
        BF16 tensor ``[num_query_tokens, num_heads, 512]``.
    """

    if query.ndim != 3 or query.shape[-1] != GLM5_NEXT_SPARSE_HEAD_DIM:
        raise ValueError(
            "GLM-5-Next sparse MLA query must have shape [tokens, heads, 512], "
            f"got {tuple(query.shape)}"
        )
    if kv_cache.shape[-1] != GLM5_NEXT_SPARSE_HEAD_DIM:
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

    use_kv_scale = kv_scale is not None
    if use_kv_scale:
        if kv_scale.ndim != 2 or kv_scale.shape != (
            kv_cache.numel() // GLM5_NEXT_SPARSE_HEAD_DIM,
            GLM5_NEXT_LATENT_SCALE_GROUPS,
        ):
            raise ValueError(
                "GLM-5-Next KV scale must have shape [physical_tokens, 4], "
                f"got {tuple(kv_scale.shape)}"
            )
        if kv_scale.device != query.device:
            raise ValueError("GLM-5-Next KV scale must share the input device")

    use_current_chunk_kv = current_chunk_kv is not None
    if use_current_chunk_kv != (current_chunk_locs is not None):
        raise ValueError(
            "GLM-5-Next current-chunk KV and locations must be provided together"
        )
    if use_current_chunk_kv:
        assert current_chunk_kv is not None and current_chunk_locs is not None
        if use_cuda_decode_kernel:
            raise ValueError(
                "GLM-5-Next current-chunk BF16 KV is valid only for EXTEND"
            )
        if not use_kv_scale or kv_cache.dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            raise ValueError(
                "GLM-5-Next current-chunk BF16 KV requires a scaled-FP8 cache"
            )
        if current_chunk_kv.ndim != 2 or current_chunk_kv.shape[-1] != (
            GLM5_NEXT_SPARSE_HEAD_DIM
        ):
            raise ValueError(
                "GLM-5-Next current-chunk KV must have shape [tokens, 512], "
                f"got {tuple(current_chunk_kv.shape)}"
            )
        if current_chunk_kv.dtype != torch.bfloat16:
            raise TypeError(
                "GLM-5-Next current-chunk KV must be BF16, "
                f"got {current_chunk_kv.dtype}"
            )
        if (
            current_chunk_locs.ndim != 1
            or current_chunk_locs.numel() != current_chunk_kv.shape[0]
            or current_chunk_locs.dtype == torch.bool
            or current_chunk_locs.is_floating_point()
        ):
            raise ValueError(
                "GLM-5-Next current-chunk locations must be a one-dimensional "
                "integer tensor matching the KV row count"
            )
        if (
            current_chunk_kv.device != query.device
            or current_chunk_locs.device != query.device
        ):
            raise ValueError(
                "GLM-5-Next current-chunk KV metadata must share the input device"
            )
        if current_chunk_kv.shape[0] == 0:
            raise ValueError("GLM-5-Next current-chunk KV must not be empty")

        # Physical cache locations are not guaranteed to be monotonic after
        # allocator reuse.  Sort once per DSA layer, then use searchsorted for
        # the bounded selected set instead of materializing a pool-sized BF16
        # lookup table.  This keeps the 500k persistent cache in FP8.
        sorted_current_locs, current_order = torch.sort(
            current_chunk_locs.to(torch.int64)
        )
    else:
        sorted_current_locs = None
        current_order = None

    if query.is_cuda and use_cuda_decode_kernel:
        return _glm5_next_sparse_mla_cuda(
            query,
            kv_cache,
            indices,
            sm_scale=sm_scale,
            kv_scale=kv_scale,
        )

    flat_kv = kv_cache.reshape(-1, GLM5_NEXT_SPARSE_HEAD_DIM)
    # PyTorch CPU does not implement advanced indexing for float8 tensors, so
    # its tiny unit-test oracle dequantizes the flat cache once.  CUDA prefill
    # must not do that: the physical pool can hold millions of token slots.
    # Gather its bounded sparse rows in FP8 and dequantize only that selection.
    gather_kv = (
        flat_kv.to(torch.bfloat16)
        if not flat_kv.is_cuda
        and flat_kv.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else flat_kv
    )
    output = torch.empty(
        (*query.shape[:-1], GLM5_NEXT_SPARSE_HEAD_DIM),
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
        selected_kv = gather_kv[safe_indices]
        query_chunk = query[start:end]
        if use_kv_scale:
            selected_scale = kv_scale[safe_indices]
            selected_kv = (
                selected_kv.float()
                .reshape(
                    *selected_kv.shape[:-1],
                    GLM5_NEXT_LATENT_SCALE_GROUPS,
                    GLM5_NEXT_LATENT_SCALE_GROUP_SIZE,
                )
                .mul(selected_scale.float().unsqueeze(-1))
                .reshape(*selected_kv.shape[:-1], GLM5_NEXT_SPARSE_HEAD_DIM)
                .to(torch.bfloat16)
            )
        else:
            selected_kv = selected_kv.to(torch.bfloat16)
        if use_current_chunk_kv:
            assert (
                current_chunk_kv is not None
                and sorted_current_locs is not None
                and current_order is not None
            )
            insertion = torch.searchsorted(
                sorted_current_locs, safe_indices.to(torch.int64)
            )
            bounded = insertion.clamp_max(sorted_current_locs.numel() - 1)
            is_current = valid & (
                sorted_current_locs[bounded] == safe_indices.to(torch.int64)
            )
            current_rows = current_order[bounded]
            selected_kv = torch.where(
                is_current.unsqueeze(-1),
                current_chunk_kv[current_rows],
                selected_kv,
            )
        query_chunk = query_chunk.to(torch.bfloat16)

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
