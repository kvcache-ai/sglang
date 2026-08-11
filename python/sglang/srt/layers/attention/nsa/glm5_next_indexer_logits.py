"""Eager correctness logits for the GLM-5-Next KPool indexer.

DeepGEMM's FP8 MQA-logits kernels do not currently support SM120.  These
helpers implement the same score contract with ordinary PyTorch operations:

    sum_h(weight[q, h] * relu(dot(q_fp8[q, h], k_fp8[k]))) * k_scale[k]

The query scale and the indexer softmax scale are already folded into
``weight`` by ``IndexerKPool._get_logits_head_gate``.  This module is private
to ``IndexerKPool``; the shared DeepSeek NSA indexer keeps its existing
backends and behavior.

This path is deliberately eager and chunked.  CUDA-graph capture and a fused
high-performance SM120 implementation are Session C work.
"""

from __future__ import annotations

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
    """Chunked non-paged FP8 MQA logits with DeepGEMM-compatible math.

    Entries outside each row's half-open ``[ks, ke)`` interval are ``-inf``.
    The caller's KPool top-k transform already consumes those same bounds, but
    initializing them here makes the eager correctness route deterministic.
    """

    k_fp8, k_scale = kv_fp8
    _validate_flat_inputs(q_fp8, k_fp8, k_scale, weights, ks, ke)
    if query_chunk_size <= 0 or key_chunk_size <= 0:
        raise ValueError("query_chunk_size and key_chunk_size must be positive")

    num_queries, _, _ = q_fp8.shape
    num_keys = k_fp8.shape[0]
    logits = torch.full(
        (num_queries, num_keys),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    if num_queries == 0 or num_keys == 0:
        return logits

    q_bf16 = q_fp8.to(torch.bfloat16)
    k_bf16 = k_fp8.to(torch.bfloat16)
    weights_fp32 = weights.float()
    scales_fp32 = k_scale.float()
    key_positions = torch.arange(num_keys, device=q_fp8.device)

    for q_start in range(0, num_queries, query_chunk_size):
        q_end = min(q_start + query_chunk_size, num_queries)
        q_chunk = q_bf16[q_start:q_end]
        weight_chunk = weights_fp32[q_start:q_end]
        ks_chunk = ks[q_start:q_end].to(torch.int64)
        ke_chunk = ke[q_start:q_end].to(torch.int64)

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
            logits[q_start:q_end, k_start:k_end] = scores.masked_fill(
                ~valid, float("-inf")
            )

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
    """Chunked paged FP8 MQA logits for eager GLM decode on SM120."""

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

    logits = torch.full(
        (batch_size, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    if batch_size == 0 or max_seq_len == 0:
        return logits

    q_bf16 = q_fp8[:, 0].to(torch.bfloat16)
    weights_fp32 = weights.float()
    # One host sync is acceptable in this explicitly eager correctness path.
    logical_lengths = seq_lens.detach().to(device="cpu", dtype=torch.int64).tolist()

    for row, requested_len in enumerate(logical_lengths):
        logical_len = min(max(int(requested_len), 0), max_seq_len)
        logical_len = min(logical_len, page_table.shape[1] * GLM5_NEXT_INDEX_PAGE_SIZE)
        if logical_len == 0:
            continue

        # Round the chunk to whole pages so cache extraction remains simple.
        keys_per_chunk = max(
            GLM5_NEXT_INDEX_PAGE_SIZE,
            (key_chunk_size // GLM5_NEXT_INDEX_PAGE_SIZE) * GLM5_NEXT_INDEX_PAGE_SIZE,
        )
        for key_start in range(0, logical_len, keys_per_chunk):
            key_end = min(key_start + keys_per_chunk, logical_len)
            page_start = key_start // GLM5_NEXT_INDEX_PAGE_SIZE
            page_end = (
                key_end + GLM5_NEXT_INDEX_PAGE_SIZE - 1
            ) // GLM5_NEXT_INDEX_PAGE_SIZE
            keys, scales = _paged_cache_keys_and_scales(
                kv_cache_fp8, page_table[row, page_start:page_end]
            )
            valid_keys = key_end - key_start
            keys = keys[:valid_keys].to(torch.bfloat16)
            scales = scales[:valid_keys].float()

            scores = torch.matmul(keys, q_bf16[row].transpose(0, 1))
            scores = F.relu(scores)
            scores = (scores * weights_fp32[row].unsqueeze(0)).sum(dim=1)
            logits[row, key_start:key_end] = scores * scales

    return logits


def use_glm5_next_eager_logits_on_device(device: torch.device) -> bool:
    """Return true only for NVIDIA SM120, where current DeepGEMM is unsupported."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) == (12, 0)
