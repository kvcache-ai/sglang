from __future__ import annotations

import os

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import topk_transform_512, topk_transform_512_v2
from sglang.srt.environ import envs

import tilelang  # noqa: E402

tilelang.set_log_level("ERROR")  # suppress TMA cp.async rewrites on SM_86
from sglang.srt.layers.deep_gemm_wrapper.configurer import DEEPGEMM_CAPS
from sglang.srt.layers.attention.compressed.metadata import (
    PagedCoreMetadata,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.indexer_topk_capturer import (
    get_global_indexer_capturer,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import C4Indexer


if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch fallback for SM < 90.

    Two strategies depending on whether we are inside a CUDA graph capture:
    - Capture  (bs is always 1, short sequences): vectorised, fixed-shape.
    - No capture (prefill / extend, larger sequences): per-batch loop
      that only materialises exactly seq_len pages — keeps peak memory
      O(actual_seq_len) instead of O(max_seq_len * num_pages).
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    in_capture = torch.cuda.is_current_stream_capturing()

    if in_capture:
        return _fp8_paged_mqa_logits_torch_capture(
            q_fp8, kvcache_fp8, weight, seq_lens, page_table,
            batch_size, num_heads, head_dim, block_size, max_seq_len,
        )
    else:
        return _fp8_paged_mqa_logits_torch_loop(
            q_fp8, kvcache_fp8, weight, seq_lens, page_table,
            batch_size, num_heads, head_dim, block_size, max_seq_len,
        )


def _fp8_paged_mqa_logits_torch_capture(
    q_fp8, kvcache_fp8, weight, seq_lens, page_table,
    batch_size, num_heads, head_dim, block_size, max_seq_len,
) -> torch.Tensor:
    """Vectorised, CUDA-graph-safe path. Only used during capture (bs=1, small seq)."""
    q = q_fp8.squeeze(1).to(torch.float32)
    q_scale = weight[:, None, :]

    max_pages = max_seq_len // block_size
    padded_seq_len = max_pages * block_size
    pages = page_table[:, :max_pages].long()

    SCALE_OFFSET = block_size * head_dim
    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    kv_vals = kvcache_flat[:, :SCALE_OFFSET]
    kv_scales = kvcache_flat[:, SCALE_OFFSET:]

    gathered_vals = kv_vals[pages].contiguous()
    gathered_scales = kv_scales[pages].contiguous()

    gathered_vals_fp8 = gathered_vals.view(
        batch_size, padded_seq_len, head_dim
    ).view(dtype=FP8_DTYPE)
    gained_vals_f32 = gathered_vals_fp8.to(torch.float32).contiguous()
    gained_vals_f32 = gained_vals_f32.view(batch_size, padded_seq_len, head_dim)

    gained_scales_f32 = (
        gathered_scales.view(dtype=torch.float32)
        .contiguous()
        .view(batch_size, padded_seq_len)
    )

    score = torch.bmm(gained_vals_f32, q.transpose(1, 2))
    score = F.relu(score) * q_scale
    score = score.sum(dim=-1) * gained_scales_f32

    sl = seq_lens.reshape(batch_size)
    mask = torch.arange(padded_seq_len, device=score.device)[None, :] < sl[:, None]
    score = score.masked_fill(~mask, float("-inf"))

    if padded_seq_len < max_seq_len:
        pad = score.new_full((batch_size, max_seq_len - padded_seq_len), float("-inf"))
        score = torch.cat([score, pad], dim=-1)
    elif padded_seq_len > max_seq_len:
        score = score[:, :max_seq_len]
    return score


def _fp8_paged_mqa_logits_torch_loop(
    q_fp8, kvcache_fp8, weight, seq_lens, page_table,
    batch_size, num_heads, head_dim, block_size, max_seq_len,
) -> torch.Tensor:
    """Per-batch-element loop — O(actual_seq_len) peak memory."""
    logits = page_table.new_full(
        (batch_size, max_seq_len), float("-inf"), dtype=torch.float32
    )
    SCALE_OFFSET = block_size * head_dim
    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))

    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        if seq_len == 0:
            continue
        num_pages = (seq_len + block_size - 1) // block_size
        padded = num_pages * block_size
        pages = page_table[i, :num_pages].long()

        kv = kvcache_flat[pages]
        kv_vals = kv[:, :SCALE_OFFSET].contiguous()
        kv_scales = kv[:, SCALE_OFFSET:].contiguous()

        kv_fp8 = kv_vals.view(padded, head_dim).view(dtype=FP8_DTYPE)
        kv_f32 = kv_fp8.to(torch.float32)
        scale_f32 = kv_scales.view(dtype=torch.float32).view(padded)

        q = q_fp8[i, 0].to(torch.float32)
        score = F.linear(kv_f32, q)
        score = F.relu(score)
        score = score * weight[i][None, :]
        score = score.sum(dim=1)
        score = score * scale_f32
        logits[i, :seq_len] = score[:seq_len]

    return logits


# ---------------------------------------------------------------------------
# SM_86 BF16 tilelang path: pre-dequant FP8→BF16, then tilelang T.gemm(BF16).
# This uses native BF16 MMA on Ampere (mma.sync.aligned.m16n8k16) and
# avoids both the Python loop of the torch fallback and the FP8 MMA
# requirement of the original tilelang kernel.
# ---------------------------------------------------------------------------

_bf16_indexer_module: Any = None


def _build_bf16_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
) -> Any:
    import tilelang as _tl
    import tilelang.language as T

    _tl.set_log_level("ERROR")  # suppress TMA cp.async rewrites on SM_86

    _cfg = {
        _tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        _tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    # Short names (without _sym suffix) so that get_type_hints in PEP563
    # mode can find them via the closure's __code__.co_freevars.
    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    d0, d1 = T.dynamic("d0, d1")

    @_tl.jit(pass_configs=_cfg)
    def bf16_paged_mqa_logits(
        q: T.Tensor[(N, H, D), "bfloat16"],
        kvcache_bf16: T.StridedTensor[(C, B, D), (d0, D, 1), "bfloat16"],
        kvcache_scale: T.StridedTensor[(C, B), (d1, 1), "float32"],
        weight: T.Tensor[(N, H), "float32"],
        seq_lens: T.Tensor[(N,), "int32"],
        page_table: T.Tensor[(N, L), "int32"],
        o: T.Tensor[(N, S), "float32"],
    ) -> None:
        _ = N, L, S, C, D, H, B, d0, d1
        with T.Kernel(N) as bx:
            seq_len = seq_lens[bx]
            q_smem = T.alloc_shared((H, D), "bfloat16")
            q_s_frag = T.alloc_fragment((H,), "float32")
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for i in T.Pipelined(T.ceildiv(seq_len, B), num_stages=2):
                page = page_table[bx, i]
                k_smem = T.alloc_shared((B, D), "bfloat16")
                k_s_frag = T.alloc_fragment((B,), "float32")
                T.copy(kvcache_bf16[page, 0, 0], k_smem)
                T.copy(kvcache_scale[page, 0], k_s_frag)

                logits = T.alloc_fragment((B, H), "float32")
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                )

                for h, j in T.Parallel(H, B):
                    logits[j, h] = T.max(logits[j, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), "float32")
                T.reduce_sum(logits, logits_sum, dim=1)
                for j in T.Parallel(B):
                    logits_sum[j] *= k_s_frag[j]
                T.copy(logits_sum, o[bx, i * B])

    return bf16_paged_mqa_logits


def _get_bf16_indexer_module(
    head_dim: int, num_heads: int, block_size: int,
) -> Any:
    global _bf16_indexer_module
    if _bf16_indexer_module is None:
        _bf16_indexer_module = _build_bf16_paged_mqa_logits_kernel(
            head_dim, num_heads, block_size,
        )
    return _bf16_indexer_module


def bf16_paged_mqa_logits_tilelang(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """SM_86 tilelang BF16 path.

    Dequants the FP8 E4M3 KV cache to BF16 on every call using PyTorch ops
    (capturable in CUDA graphs), then runs a tilelang kernel with native
    BF16 MMA (T.gemm on bfloat16).
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert clean_logits is False

    # Dequant FP8 E4M3 → BF16 on every call.  These are plain PyTorch ops
    # and therefore safe inside CUDA graph capture/replay — unlike Python
    # control flow or .item() calls.
    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim
    num_blocks = kvcache_flat.shape[0]

    kv_u8 = kvcache_flat[:, :SCALE_OFFSET].contiguous()
    kv_fp8 = kv_u8.view(-1, block_size, head_dim).view(
        dtype=torch.float8_e4m3fn
    )
    kvcache_bf16 = kv_fp8.to(torch.bfloat16)

    kv_scales = kvcache_flat[:, SCALE_OFFSET:].contiguous()
    scales = kv_scales.view(dtype=torch.float32).view(
        num_blocks, block_size,
    ).contiguous()

    q_bf16 = q_fp8.squeeze(1).to(torch.bfloat16).view(
        batch_size, num_heads, head_dim,
    )

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    kernel = _get_bf16_indexer_module(head_dim, num_heads, block_size)
    kernel(
        q_bf16,
        kvcache_bf16,
        scales,
        weight.float(),
        seq_lens.int(),
        page_table.int(),
        logits,
    )
    return logits


# ---------------------------------------------------------------------------
# SM_86 BF16 direct kernel: KV cache is already bfloat16 (no FP8 quant),
# so there are no scales to load or multiply.  Simpler and faster.
# ---------------------------------------------------------------------------

_bf16_direct_indexer_module: Any = None


def _build_bf16_direct_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
) -> Any:
    import tilelang as _tl
    import tilelang.language as T

    _tl.set_log_level("ERROR")

    _cfg = {
        _tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        _tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    d0 = T.dynamic("d0")

    @_tl.jit(pass_configs=_cfg)
    def bf16_direct_paged_mqa_logits(
        q: T.Tensor[(N, H, D), "bfloat16"],
        kvcache_bf16: T.StridedTensor[(C, B, D), (d0, D, 1), "bfloat16"],
        weight: T.Tensor[(N, H), "float32"],
        seq_lens: T.Tensor[(N,), "int32"],
        page_table: T.Tensor[(N, L), "int32"],
        o: T.Tensor[(N, S), "float32"],
    ) -> None:
        _ = N, L, S, C, D, H, B, d0
        with T.Kernel(N) as bx:
            seq_len = seq_lens[bx]
            q_smem = T.alloc_shared((H, D), "bfloat16")
            q_s_frag = T.alloc_fragment((H,), "float32")
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for i in T.Pipelined(T.ceildiv(seq_len, B), num_stages=2):
                page = page_table[bx, i]
                k_smem = T.alloc_shared((B, D), "bfloat16")
                T.copy(kvcache_bf16[page, 0, 0], k_smem)

                logits = T.alloc_fragment((B, H), "float32")
                T.gemm(
                    k_smem, q_smem, logits,
                    transpose_A=False, transpose_B=True, clear_accum=True,
                )

                for h, j in T.Parallel(H, B):
                    logits[j, h] = T.max(logits[j, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), "float32")
                T.reduce_sum(logits, logits_sum, dim=1)
                T.copy(logits_sum, o[bx, i * B])

    return bf16_direct_paged_mqa_logits


def _get_bf16_direct_indexer_module(
    head_dim: int, num_heads: int, block_size: int,
) -> Any:
    global _bf16_direct_indexer_module
    if _bf16_direct_indexer_module is None:
        _bf16_direct_indexer_module = _build_bf16_direct_paged_mqa_logits_kernel(
            head_dim, num_heads, block_size,
        )
    return _bf16_direct_indexer_module


def bf16_direct_paged_mqa_logits_tilelang(
    q_bf16: torch.Tensor,
    kvcache_bf16: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """SM_86 BF16-direct path: KV cache is already bfloat16, no dequant, no scales."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_bf16.shape
    block_size = kvcache_bf16.shape[1]
    assert head_dim == 128
    assert block_size == 64
    assert clean_logits is False

    q_bf16 = q_bf16.squeeze(1).to(torch.bfloat16).view(
        batch_size, num_heads, head_dim,
    )

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    kernel = _get_bf16_direct_indexer_module(head_dim, num_heads, block_size)
    kernel(
        q_bf16,
        kvcache_bf16,
        weight.float(),
        seq_lens.int(),
        page_table.int(),
        logits,
    )
    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = (
            torch.arange(TOPK, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(
                sequential_valid,
                sequential_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            ),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(
        valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32)
    )

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(
            valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32)
        )
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_bf16_buffer(
                layer_id=c4_indexer.layer_id,
            )
        else:
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            if cc < (8, 9):
                q_fp8 = q  # BF16 direct, kernel converts internally
                q_scale = q.float().abs().amax(dim=-1) / 448.0
            else:
                q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            if cc < (8, 9):
                weights = weights * c4_indexer.weight_scale * q_scale
                weights = weights.unsqueeze(-1)
            else:
                weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        cc = torch.cuda.get_device_capability()
        if cc < (8, 9):
            q_fp8 = q
            q_scale = q.float().abs().amax(dim=-1) / 448.0
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights = weights * c4_indexer.weight_scale * q_scale
            weights = weights.unsqueeze(-1)
        else:
            q_fp8, q_scale = act_quant(q)
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        if cc < (8, 9):
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_bf16_buffer(
                layer_id=c4_indexer.layer_id,
            )
        else:
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )
        return q_fp8, weights, c4_indexer_kv_cache

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackend)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
            DSV4AttnMetadataRadix,
        )

        assert isinstance(core_metadata, (PagedCoreMetadata, DSV4AttnMetadataRadix))
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)

        _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if _cc < (8, 9):
            # BF16 direct: cache is (num_pages, block_size, head_dim) bfloat16 — no view
            assert c4_indexer_kv_cache.shape[1:] == (64, 128)
        else:
            assert len(c4_indexer_kv_cache.shape) == 2
            block_kv = 64
            num_heads_kv = 1
            head_dim_with_sf = 132
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # The tilelang and torch reference impls expect a 1-D `seq_lens`
        # (assert ``shape == (batch_size,)``); only the deep_gemm path takes
        # the 2-D ``(batch, 1)`` form. Pick the right shape per backend.
        #
        # Backend selection (capability-driven, env override):
        #   - env SGLANG_OPT_USE_TILELANG_INDEXER=1 forces tilelang.
        #   - env SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 forces torch.
        #   - Otherwise: cap in DEEPGEMM_CAPS -> deep_gemm; else tilelang.
        # Origin: sglang 本身.
        _has_dg_caps = (torch.cuda.get_device_capability() in DEEPGEMM_CAPS
                        if torch.cuda.is_available() else False)
        _force_tilelang = envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
        _force_torch = envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        _auto_tilelang = (
            "SGLANG_OPT_USE_TILELANG_INDEXER" not in os.environ
            and "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH" not in os.environ
            and not _has_dg_caps
            and _cc >= (8, 9)  # tilelang FP8 MMA requires SM>=89; SM_86 lacks it
        )

        seq_lens_2d = True
        if _force_tilelang or _auto_tilelang:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
            seq_lens_2d = False
        elif _force_torch:
            fn = fp8_paged_mqa_logits_torch
            seq_lens_2d = False
        elif _cc < (8, 9):
            # SM_86: KV cache already bfloat16 (no act_quant in compressor),
            # so the kernel reads BF16 directly — no dequant, no scales.
            fn = bf16_direct_paged_mqa_logits_tilelang
            seq_lens_2d = False
        else:
            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    fp8_paged_mqa_logits_chunked as fn,
                )
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        _c4sl = indexer_metadata.c4_seq_lens
        if seq_lens_2d:
            if _c4sl.dim() == 1:
                _c4sl = _c4sl.unsqueeze(-1)
        else:
            if _c4sl.dim() == 2 and _c4sl.shape[-1] == 1:
                _c4sl = _c4sl.squeeze(-1)
        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            _c4sl,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer.is_enabled()

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]

        if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = token_to_kv_pool.c4_kv_pool.translate_loc_from_compressed_to_hisparse_device(
                    core_metadata.c4_sparse_page_indices
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)
