"""GLM-5-Next KPool indexer logits.

DeepGEMM's FP8 MQA-logits kernels do not currently support SM120.  These
helpers implement the same score contract with ordinary PyTorch operations:

    sum_h(weight[q, h] * relu(dot(q_fp8[q, h], k_fp8[k]))) * k_scale[k]

The query scale and the indexer softmax scale are already folded into
``weight`` by ``IndexerKPool._get_logits_head_gate``.  This module is private
to ``IndexerKPool``; the shared DeepSeek NSA indexer keeps its existing
backends and behavior.

The paged decode path is deliberately vectorized and device-length driven so
it can be captured with the fixed buffers used by SGLang CUDA graphs.  The
ragged prefill path remains statically chunked to bound its temporary memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F


GLM5_NEXT_INDEX_HEAD_DIM = 128
GLM5_NEXT_INDEX_PAGE_SIZE = 64
GLM5_NEXT_INDEX_PAGE_NBYTES = GLM5_NEXT_INDEX_PAGE_SIZE * (GLM5_NEXT_INDEX_HEAD_DIM + 4)
GLM5_NEXT_INDEX_SCALE_OFFSET = GLM5_NEXT_INDEX_PAGE_SIZE * GLM5_NEXT_INDEX_HEAD_DIM

# Keep the largest temporary score tile bounded.  With GLM's 32 index heads,
# the default tile is 32 * 2048 * 32 BF16 values (4 MiB), followed by an FP32
# weighted temporary (8 MiB).
_DEFAULT_QUERY_CHUNK = 32
_DEFAULT_KEY_CHUNK = 2048

# The ragged prefill caller consumes each row chunk before requesting the next
# one.  Keeping this fixed (rather than deriving it from free GPU memory) makes
# the allocation bound deterministic: at the final 500K boundary, one FP32
# logits chunk is about 15.3 MiB instead of a 1.9-GiB [4096, 125056] matrix.
GLM5_NEXT_PREFILL_QUERY_ROW_CHUNK = _DEFAULT_QUERY_CHUNK


def _validate_flat_inputs(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> None:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must have shape [Q, H, D], got {q_fp8.shape}")
    if k_fp8.ndim != 2:
        raise ValueError(f"k_fp8 must have shape [K, D], got {k_fp8.shape}")
    num_queries, num_heads, head_dim = q_fp8.shape
    if head_dim != GLM5_NEXT_INDEX_HEAD_DIM:
        raise ValueError(
            f"GLM-5-Next indexer fallback requires head_dim=128, got {head_dim}"
        )
    if k_fp8.shape[1] != head_dim:
        raise ValueError(f"K head_dim ({k_fp8.shape[1]}) does not match Q ({head_dim})")
    if weights.shape != (num_queries, num_heads):
        raise ValueError(
            f"weights must have shape {(num_queries, num_heads)}, got {weights.shape}"
        )
    if k_scale.shape != (k_fp8.shape[0],):
        raise ValueError(
            f"k_scale must have shape {(k_fp8.shape[0],)}, got {k_scale.shape}"
        )
    if ks.shape != (num_queries,) or ke.shape != (num_queries,):
        raise ValueError(
            f"ks/ke must both have shape {(num_queries,)}, got {ks.shape}/{ke.shape}"
        )
    if q_fp8.device != k_fp8.device or q_fp8.device != k_scale.device:
        raise ValueError("Q, K, and K scale must be on the same device")
    if q_fp8.device != weights.device or q_fp8.device != ks.device:
        raise ValueError("weights and ragged bounds must be on the Q device")
    if ke.device != q_fp8.device:
        raise ValueError("ragged end bounds must be on the Q device")


def glm5_next_prefill_query_row_ranges(
    num_queries: int,
    *,
    query_chunk_size: int = GLM5_NEXT_PREFILL_QUERY_ROW_CHUNK,
) -> Iterator[tuple[int, int]]:
    """Yield deterministic half-open query-row chunks without tensor allocation."""

    if num_queries < 0:
        raise ValueError("num_queries must be non-negative")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    for q_start in range(0, num_queries, query_chunk_size):
        yield q_start, min(q_start + query_chunk_size, num_queries)


def iter_glm5_next_eager_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    query_chunk_size: int = _DEFAULT_QUERY_CHUNK,
    key_chunk_size: int = _DEFAULT_KEY_CHUNK,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield bounded row chunks of non-paged FP8 MQA logits.

    Entries outside each row's half-open ``[ks, ke)`` interval are ``-inf``.
    The caller's KPool top-k transform already consumes those same bounds, but
    initializing them here makes the eager correctness route deterministic.
    K and its scale are converted exactly once and retained across yielded row
    chunks; consumers should finish top-k and release a chunk before advancing.
    """

    k_fp8, k_scale = kv_fp8
    _validate_flat_inputs(q_fp8, k_fp8, k_scale, weights, ks, ke)
    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError("query_chunk_size and key_chunk_size must be positive")

    num_queries, _, _ = q_fp8.shape
    num_keys = k_fp8.shape[0]
    if num_queries == 0 or num_keys == 0:
        for q_start, q_end in glm5_next_prefill_query_row_ranges(
            num_queries, query_chunk_size=query_chunk_size
        ):
            yield (
                q_start,
                q_end,
                torch.full(
                    (q_end - q_start, num_keys),
                    float("-inf"),
                    dtype=torch.float32,
                    device=q_fp8.device,
                ),
            )
        return

    q_bf16 = q_fp8.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)
    weights_fp32 = weights.float()
    scales_fp32 = k_scale.float()
    key_positions = torch.arange(num_keys, device=q_fp8.device)

    for q_start, q_end in glm5_next_prefill_query_row_ranges(
        num_queries, query_chunk_size=query_chunk_size
    ):
        q_chunk = q_bf16[q_start:q_end]
        weight_chunk = weights_fp32[q_start:q_end]
        ks_chunk = ks[q_start:q_end].to(torch.int64)
        ke_chunk = ke[q_start:q_end].to(torch.int64)
        logits_chunk = torch.full(
            (q_end - q_start, num_keys),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )

        for k_start in range(0, num_keys, key_chunk_size):
            k_end = min(k_start + key_chunk_size, num_keys)
            key_chunk = k_bf16[k_start:k_end]

            # [Q, K, H].  BF16 operands match SGLang's existing SM120 torch
            # reference for paged MQA; weighting/reduction are FP32.
            scores = torch.bmm(
                key_chunk.unsqueeze(0).expand(q_end - q_start, -1, -1),
                q_chunk.transpose(1, 2),
            )
            scores = F.relu(scores)
            scores = (scores * weight_chunk.unsqueeze(1)).sum(dim=2)
            scores = scores * scales_fp32[k_start:k_end].unsqueeze(0)

            positions = key_positions[k_start:k_end].unsqueeze(0)
            valid = (positions >= ks_chunk.unsqueeze(1)) & (
                positions < ke_chunk.unsqueeze(1)
            )
            logits_chunk[:, k_start:k_end] = scores.masked_fill(~valid, float("-inf"))

        yield q_start, q_end, logits_chunk
        del logits_chunk


def glm5_next_eager_fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    query_chunk_size: int = _DEFAULT_QUERY_CHUNK,
    key_chunk_size: int = _DEFAULT_KEY_CHUNK,
) -> torch.Tensor:
    """Materialize the complete logits matrix for small tests and callers.

    Production GLM-5-Next ragged prefill consumes
    :func:`iter_glm5_next_eager_fp8_mqa_logits` directly so its memory is
    bounded by one query-row chunk.
    """

    k_fp8, k_scale = kv_fp8
    _validate_flat_inputs(q_fp8, k_fp8, k_scale, weights, ks, ke)
    logits = torch.full(
        (q_fp8.shape[0], k_fp8.shape[0]),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    for q_start, q_end, logits_chunk in iter_glm5_next_eager_fp8_mqa_logits(
        q_fp8,
        kv_fp8,
        weights,
        ks,
        ke,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
    ):
        logits[q_start:q_end] = logits_chunk
    return logits


def _paged_cache_keys_and_scales(
    kv_cache_fp8: torch.Tensor,
    page_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed GLM index pages and return FP8 keys plus FP32 scales."""

    if kv_cache_fp8.ndim < 2:
        raise ValueError("packed KPool cache must have at least two dimensions")
    cache_u8 = kv_cache_fp8.view(torch.uint8)
    cache_flat = cache_u8.reshape(cache_u8.shape[0], -1)
    if cache_flat.shape[1] != GLM5_NEXT_INDEX_PAGE_NBYTES:
        raise ValueError(
            "packed GLM index page must contain 64x128 FP8 bytes and 64 FP32 "
            f"scales ({GLM5_NEXT_INDEX_PAGE_NBYTES} bytes), got "
            f"{cache_flat.shape[1]}"
        )

    # Only pages below a valid logical sequence length are selected.  Clamping
    # mirrors SGLang's existing torch paged-MQA reference and prevents a stale
    # padding sentinel from becoming a negative Python-style index.
    gathered = cache_flat.index_select(0, page_ids.to(torch.int64).clamp_min(0))
    key_bytes = gathered[:, :GLM5_NEXT_INDEX_SCALE_OFFSET].contiguous()
    scale_bytes = gathered[:, GLM5_NEXT_INDEX_SCALE_OFFSET:].contiguous()
    keys = key_bytes.view(torch.float8_e4m3fn).reshape(-1, GLM5_NEXT_INDEX_HEAD_DIM)
    scales = scale_bytes.view(torch.float32).reshape(-1)
    return keys, scales


def glm5_next_eager_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    *,
    key_chunk_size: int = _DEFAULT_KEY_CHUNK,
) -> torch.Tensor:
    """Graph-safe paged FP8 MQA logits for GLM decode on SM120.

    ``max_seq_len`` and the page-table shape are capture-time constants.  The
    logical lengths stay on device and only mask the fixed-width result.  In
    particular, this function must not branch or choose launch geometry from
    the dummy sequence lengths used while a graph is captured.
    """

    if q_fp8.ndim != 4 or q_fp8.shape[1] != 1:
        raise ValueError(f"paged q_fp8 must have shape [B, 1, H, D], got {q_fp8.shape}")
    batch_size, _, num_heads, head_dim = q_fp8.shape
    if head_dim != GLM5_NEXT_INDEX_HEAD_DIM:
        raise ValueError(
            f"GLM-5-Next paged indexer fallback requires head_dim=128, got {head_dim}"
        )
    if weights.shape != (batch_size, num_heads):
        raise ValueError(
            f"weights must have shape {(batch_size, num_heads)}, got {weights.shape}"
        )
    if seq_lens.ndim == 2 and seq_lens.shape[1] == 1:
        seq_lens = seq_lens.squeeze(1)
    if seq_lens.shape != (batch_size,):
        raise ValueError(
            f"seq_lens must have shape {(batch_size,)}, got {seq_lens.shape}"
        )
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError(
            f"page_table must have batch size {batch_size}, got {page_table.shape}"
        )
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if key_chunk_size <= 0:
        raise ValueError("key_chunk_size must be positive")

    # ``key_chunk_size`` is retained for source/API compatibility with the
    # eager Session-AB helper.  Decode batch sizes are capped at 1/2/4, so a
    # single vectorized launch is both graph-stable and bounded.
    del key_chunk_size

    max_pages = (
        max_seq_len + GLM5_NEXT_INDEX_PAGE_SIZE - 1
    ) // GLM5_NEXT_INDEX_PAGE_SIZE
    if max_pages > page_table.shape[1]:
        raise ValueError(
            f"page_table has {page_table.shape[1]} pages but {max_pages} are "
            f"required for max_seq_len={max_seq_len}"
        )

    # Gather every capture-width page.  Invalid logical positions are masked
    # below using the replay-updated device lengths.  Clamp padding/stale table
    # cells so even an entirely empty padded row can never address outside the
    # physical index cache.
    cache_u8 = kv_cache_fp8.view(torch.uint8)
    cache_flat = cache_u8.reshape(cache_u8.shape[0], -1)
    if cache_flat.shape[1] != GLM5_NEXT_INDEX_PAGE_NBYTES:
        raise ValueError(
            "packed GLM index page must contain 64x128 FP8 bytes and 64 FP32 "
            f"scales ({GLM5_NEXT_INDEX_PAGE_NBYTES} bytes), got "
            f"{cache_flat.shape[1]}"
        )

    page_ids = page_table[:, :max_pages].to(torch.int64)
    page_ids = page_ids.clamp(min=0, max=cache_flat.shape[0] - 1)
    gathered = cache_flat[page_ids]
    key_bytes = gathered[..., :GLM5_NEXT_INDEX_SCALE_OFFSET].contiguous()
    scale_bytes = gathered[..., GLM5_NEXT_INDEX_SCALE_OFFSET:].contiguous()

    padded_seq_len = max_pages * GLM5_NEXT_INDEX_PAGE_SIZE
    keys = key_bytes.view(torch.float8_e4m3fn).reshape(
        batch_size, padded_seq_len, GLM5_NEXT_INDEX_HEAD_DIM
    )
    scales = scale_bytes.view(torch.float32).reshape(batch_size, padded_seq_len)

    q_bf16 = q_fp8[:, 0].to(torch.bfloat16)
    scores = torch.bmm(keys.to(torch.bfloat16), q_bf16.transpose(1, 2))
    scores = F.relu(scores)
    scores = (scores.float() * weights.float().unsqueeze(1)).sum(dim=2)
    scores.mul_(scales.float())

    logits = scores[:, :max_seq_len].contiguous()
    positions = torch.arange(max_seq_len, dtype=torch.int32, device=q_fp8.device)
    logical_lengths = seq_lens.to(torch.int32).clamp(min=0, max=max_seq_len)
    logits.masked_fill_(
        positions.unsqueeze(0) >= logical_lengths.unsqueeze(1), float("-inf")
    )
    return logits


def use_glm5_next_eager_logits_on_device(device: torch.device) -> bool:
    """Return true only for NVIDIA SM120, where current DeepGEMM is unsupported."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) == (12, 0)
