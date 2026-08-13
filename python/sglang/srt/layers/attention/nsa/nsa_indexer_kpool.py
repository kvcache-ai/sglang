from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig

from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    rotate_activation,
)
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if is_npu():
    import custom_ops  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache, NSATokenToKVPool


def _token_pool_from_batch(forward_batch: ForwardBatch) -> "KVCache":
    """Return the cache pool owned by this forward, or fail closed.

    KPool plans and sequence metadata are derived from ``forward_batch``.  Use
    the pool carried by that same batch so nested backend contexts cannot pair
    a batch-local plan with an unrelated ambient cache.  Both eager execution
    and CUDA Graph capture populate this field.
    """

    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None:
        raise RuntimeError("GLM-5-Next KPool requires ForwardBatch.token_to_kv_pool")
    return pool


class IndexerKPool(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.compress_gate_stream = None
        self.skip_rope = skip_rope

        assert config is not None, "KPool indexer requires the model config"
        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress = config.index_kpool_compress

        assert self.index_kpool == 4, (
            f"GLM-5-Next requires index_kpool=4, got {self.index_kpool}"
        )
        assert self.index_kpool_compress, "GLM-5-Next requires KPool compression"
        assert self.index_kpool_always_select_tail, (
            "GLM-5-Next requires index_kpool_always_select_tail"
        )

        assert self.index_topk % self.index_kpool == 0, (
            f"index_topk ({self.index_topk}) must be divisible by "
            f"index_kpool ({self.index_kpool})"
        )
        assert 64 % self.index_kpool == 0, (
            f"index_kpool ({self.index_kpool}) must divide page_size (64)"
        )

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, self.hidden_size, dtype=torch.bfloat16)
        )

        if is_cuda() and self.alt_stream is not None:
            self.compress_gate_stream = torch.cuda.Stream()

        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        # NOTE: weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenience
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)
        if not self.skip_rope:
            self.rotary_emb = get_rope_wrapper(
                rope_head_dim,
                rotary_dim=rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,  # type: ignore
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_global_server_args().device,
            )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    @staticmethod
    def _materialize_gate_input(x: torch.Tensor) -> torch.Tensor:
        """Materialize KT's optional grouped-FP8 activation tuple for BF16 gates."""
        if not isinstance(x, tuple):
            return x
        assert len(x) in (2, 3), (
            "tuple activations must be (x_fp8, x_scale[, residual])"
        )
        x_q, x_scale = x[0], x[1]
        if (
            x_scale is not None
            and x_q.dim() == 2
            and x_scale.dim() == 2
            and x_q.shape[0] == x_scale.shape[0]
        ):
            rows, width = x_q.shape
            groups = x_scale.shape[1]
            if groups > 0 and width % groups == 0:
                group_width = width // groups
                return (
                    x_q.to(torch.float32)
                    .view(rows, groups, group_width)
                    .mul_(x_scale.to(torch.float32).unsqueeze(-1))
                    .view(rows, width)
                    .to(torch.bfloat16)
                )
        return x_q.to(torch.bfloat16)

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        x = self._materialize_gate_input(x)
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    @staticmethod
    def _get_index_k_read_buffer(pool, layer_id: int) -> torch.Tensor:
        if hasattr(pool, "get_broadcastable_index_k_with_scale_buffer"):
            return pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        if hasattr(pool, "_get_broadcastable_index_buffer"):
            return pool._get_broadcastable_index_buffer(layer_id)
        return pool.get_index_k_with_scale_buffer(layer_id=layer_id)

    def _write_compressed_pooled_index_cache(
        self,
        slot_k,
        slot_score,
        write_locs,
        forward_batch,
        layer_id,
        write_mask=None,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        if slot_k.shape[0] == 0:
            if return_compressed:
                return (
                    torch.empty(
                        (0, self.head_dim),
                        dtype=torch.float8_e4m3fn,
                        device=slot_k.device,
                    ),
                    torch.empty((0,), dtype=torch.float32, device=slot_k.device),
                )
            return None
        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            kpool_softmax_rotate_write_cache,
        )

        pool = _token_pool_from_batch(forward_batch)
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            if not return_compressed:
                return None
            write_cache = False

        buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        return kpool_softmax_rotate_write_cache(
            pool=pool,
            buf=buf,
            slot_k=slot_k,
            slot_score=slot_score,
            ape=self.index_kpool_compress_ape,
            loc=write_locs.contiguous(),
            write_mask=write_mask.contiguous() if write_mask is not None else None,
            round_scale=self.scale_fmt is not None,
            return_compressed=return_compressed,
            write_cache=write_cache,
        )

    def _compress_write_decode(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
    ):
        batch = key.shape[0]
        if batch == 0:
            return

        pool = _token_pool_from_batch(forward_batch)
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return

        pool.kpool_decode_update_index_cache(
            layer_id=layer_id,
            key=key,
            slot_score=gate_score,
            ape=self.index_kpool_compress_ape,
            block_tables=metadata.get_page_table_64(),
            req_pool_indices=forward_batch.req_pool_indices[:batch],
            positions=positions[:batch],
            seq_lens=metadata.get_seqlens_int32()[:batch],
            out_cache_loc=forward_batch.out_cache_loc[:batch],
            round_scale=self.scale_fmt is not None,
        )

    def _compress_write_extend(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        """Apply the eager multi-pool compression plan and update the tail.

        CP, speculative decoding, and CUDA-graph write plans intentionally live
        outside this GLM-5-Next bring-up path.
        """
        assert not return_compressed, "deferred KPool cache writes are unsupported"
        assert write_cache, "eager KPool extend always writes the cache"
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        attn_metadata = getattr(metadata, "attn_metadata", None)
        plan = getattr(attn_metadata, "kpool_extend_plan", None)
        assert plan is not None, "eager KPool extend requires kpool_extend_plan"

        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            kpool_assemble_softmax_rotate_write_cache,
            scatter_kpool_tail_updates,
        )

        pool = _token_pool_from_batch(forward_batch)
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return None

        writes, tails = plan.writes, plan.tails
        if writes.is_empty and tails.is_empty:
            return None

        tail_k_buf, tail_score_buf = pool.get_compress_tail_buffers(layer_id)
        if not writes.is_empty:
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            kpool_assemble_softmax_rotate_write_cache(
                pool=pool,
                buf=buf,
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=writes.req,
                n_from_tail=writes.n_from_tail,
                chunk_src_start=writes.chunk_src,
                tail_logical_base=writes.tail_logical_base,
                ape=self.index_kpool_compress_ape,
                loc=writes.write_loc,
                write_mask=None,
                round_scale=self.scale_fmt is not None,
            )

        if not tails.is_empty:
            scatter_kpool_tail_updates(
                pool=pool,
                chunk_k=key,
                chunk_score=gate_score,
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                req_pool_idx=tails.req,
                dst_logical_start=tails.dst_logical_start,
                chunk_src_start=tails.chunk_src,
                n_write=tails.n_write,
            )
        return None

    def _compress_write(
        self,
        x,
        key,
        positions,
        forward_batch,
        layer_id,
        metadata,
        gate_score: Optional[torch.Tensor] = None,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        if key.shape[0] == 0:
            return None

        if gate_score is None:
            gate_score = F.linear(
                self._materialize_gate_input(x), self.index_kpool_compress_gate
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self._compress_write_decode(
                key=key,
                gate_score=gate_score,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
            )
        elif forward_batch.forward_mode.is_extend_without_speculative():
            return self._compress_write_extend(
                key=key,
                gate_score=gate_score,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
                return_compressed=return_compressed,
                write_cache=write_cache,
            )
        else:
            raise NotImplementedError(
                "index_kpool_compress currently supports decode and extend only."
            )
        return None

    def _compute_gate_score_if_missing(
        self, x: torch.Tensor, gate_score: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_score is not None:
            return gate_score
        return F.linear(self._materialize_gate_input(x), self.index_kpool_compress_gate)

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        precompute_compress_gate: bool = False,
    ):
        gate_score = None
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if precompute_compress_gate:
                assert self.compress_gate_stream is not None
                self.compress_gate_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            if precompute_compress_gate:
                with torch.cuda.stream(self.compress_gate_stream):
                    gate_score = F.linear(
                        self._materialize_gate_input(x),
                        self.index_kpool_compress_gate,
                    )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        if not self.skip_rope:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        query = rotate_activation(query)

        return query, key, gate_score

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if not self.skip_rope:
            _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
            key[..., : self.rope_head_dim] = k_rope
        return key

    def _full_topk_for_short_sequence(
        self, metadata: BaseIndexerMetadata, device: torch.device
    ) -> torch.Tensor:
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=device,
        )
        topk_full = metadata.topk_transform(dummy_logits, self.index_topk)
        if self.index_kpool == 1:
            return topk_full
        padding = torch.full(
            (topk_full.shape[0], self.index_kpool - 1),
            -1,
            dtype=topk_full.dtype,
            device=topk_full.device,
        )
        return torch.cat([topk_full, padding], dim=1)

    def _topk_from_kpool_logits(
        self,
        logits: torch.Tensor,
        pool_lens: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        topk_offsets: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        out_rows: Optional[int] = None,
        page_table_row_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            topk_from_pooled_history_logits,
        )

        n_rows = logits.shape[0]
        if (
            page_table is not None
            and page_table_row_index is None
            and page_table.shape[0] != n_rows
        ):
            page_table = page_table[:n_rows]
        if topk_offsets is not None and topk_offsets.shape[0] != n_rows:
            topk_offsets = topk_offsets[:n_rows]
        if page_table_row_index is not None and page_table_row_index.shape[0] != n_rows:
            page_table_row_index = page_table_row_index[:n_rows]

        return topk_from_pooled_history_logits(
            logits=logits,
            group_lengths=pool_lens,
            pool_size=self.index_kpool,
            topk=self.index_topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    def _get_kpool_decode_metadata(
        self,
        metadata: BaseIndexerMetadata,
        block_tables: torch.Tensor,
        seqlens_32: torch.Tensor,
        blocksize: int,
        build_schedule_metadata: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            build_pooled_page_table_64,
        )

        attn_metadata = getattr(metadata, "attn_metadata", None)
        pool_seqlens = getattr(attn_metadata, "pooled_cache_seqlens_int32", None)
        pool_block_tables = getattr(attn_metadata, "pooled_real_page_table", None)
        pool_schedule_metadata = getattr(
            attn_metadata, "pooled_paged_mqa_schedule_metadata", None
        )

        if (
            pool_seqlens is None
            or pool_block_tables is None
            or getattr(attn_metadata, "pooled_index_kpool", 1) != self.index_kpool
        ):
            pool_seqlens = torch.div(
                seqlens_32, self.index_kpool, rounding_mode="floor"
            ).to(torch.int32)
            pool_block_tables = build_pooled_page_table_64(
                block_tables, self.index_kpool
            ).contiguous()
            pool_schedule_metadata = None
        else:
            pool_seqlens = pool_seqlens[: seqlens_32.shape[0]]
            pool_block_tables = pool_block_tables[
                : block_tables.shape[0],
                : (block_tables.shape[1] + self.index_kpool - 1) // self.index_kpool,
            ]

        pool_context_lens = pool_seqlens.contiguous().view(-1, 1)
        if pool_schedule_metadata is None and build_schedule_metadata:
            pool_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                pool_context_lens.clamp(min=1), blocksize, self.sm_count
            )

        return (
            pool_seqlens,
            pool_context_lens,
            pool_block_tables,
            pool_schedule_metadata,
        )

    @staticmethod
    def _kpool_fused_topk_mapping(
        metadata: BaseIndexerMetadata,
        paged_page_table: Optional[torch.Tensor] = None,
        paged_page_table_row_index: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not envs.SGLANG_NSA_FUSE_TOPK.get():
            return None, None, None

        topk_method = getattr(metadata, "topk_transform_method", None)
        attn_metadata = getattr(metadata, "attn_metadata", None)
        if getattr(topk_method, "name", "") == "PAGED":
            page_table_1 = (
                paged_page_table
                if paged_page_table is not None
                else getattr(attn_metadata, "page_table_1", None)
            )
            assert page_table_1 is not None
            row_index = (
                paged_page_table_row_index if paged_page_table is not None else None
            )
            return page_table_1, None, row_index
        if getattr(topk_method, "name", "") == "RAGGED":
            return None, getattr(attn_metadata, "topk_indices_offset", None), None
        return None, None, None

    @staticmethod
    def _should_use_tilelang_paged_mqa_logits(q_fp8: torch.Tensor) -> bool:
        if not is_cuda():
            return False
        arch_major, _ = torch.cuda.get_device_capability(q_fp8.device)
        num_heads = q_fp8.shape[2]
        return arch_major == 9 and num_heads not in (32, 64)

    @staticmethod
    def _should_use_eager_logits(q_fp8: torch.Tensor) -> bool:
        """Keep the unfused SM120 route private to GLM-5-Next KPool."""

        from sglang.srt.layers.attention.nsa.glm5_next_indexer_logits import (
            use_glm5_next_eager_logits_on_device,
        )

        return use_glm5_next_eager_logits_on_device(q_fp8.device)

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(_token_pool_from_batch(forward_batch), NSATokenToKVPool)

        pool = _token_pool_from_batch(forward_batch)
        page_size = pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        kv_cache_fp8 = self._get_index_k_read_buffer(pool, layer_id)

        blocksize = page_size
        seqlens_32 = metadata.get_seqlens_int32()
        assert len(q_fp8.shape) == 3
        num_q_padded = q_fp8.shape[0]
        n_real = seqlens_32.shape[0]
        if n_real < num_q_padded:
            q_fp8 = q_fp8[:n_real]
            weights = weights[:n_real]
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        use_tilelang_paged_mqa = self._should_use_tilelang_paged_mqa_logits(q_fp8)
        use_eager_logits = self._should_use_eager_logits(q_fp8)

        pool_seqlens, pool_context_lens, pool_block_tables, pool_schedule_metadata = (
            self._get_kpool_decode_metadata(
                metadata,
                block_tables,
                seqlens_32,
                blocksize,
                build_schedule_metadata=not (
                    use_tilelang_paged_mqa or use_eager_logits
                ),
            )
        )
        pool_max_seq_len = pool_block_tables.shape[1] * blocksize
        if use_eager_logits:
            from sglang.srt.layers.attention.nsa.glm5_next_indexer_logits import (
                glm5_next_eager_fp8_paged_mqa_logits,
            )

            logits = glm5_next_eager_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                pool_seqlens,
                pool_block_tables,
                pool_max_seq_len,
            )
        elif use_tilelang_paged_mqa:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits,
            )

            logits = tilelang_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                pool_seqlens,
                pool_block_tables,
                pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )
        else:
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                pool_context_lens,
                pool_block_tables,
                pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )

        page_table_1, topk_offsets, _ = self._kpool_fused_topk_mapping(metadata)
        topk_result = self._topk_from_kpool_logits(
            logits,
            pool_seqlens,
            seq_lens=seqlens_32,
            page_table=page_table_1,
            topk_offsets=topk_offsets,
            out_rows=num_q_padded if num_q_padded != n_real else None,
        )
        return topk_result

    def _get_topk_ragged_kpool_plan(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            gather_index_k_scale_prefix_into,
        )

        plan = metadata.attn_metadata.kpool_extend_plan
        assert plan is not None, "kpool extend plan is required"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        device = q_fp8.device
        total_q = q_fp8.shape[0]
        seq_lens_expanded = plan.seq_lens_expanded
        pool_lens = plan.pooled_seq_lens_expanded
        ks_per_q = plan.ragged_q_ks
        ke_per_q = plan.ragged_q_ke
        total_k_rows = plan.ragged_total_k_rows

        n_real = seq_lens_expanded.shape[0]
        assert n_real <= total_q, (
            f"plan has more real rows ({n_real}) than q_fp8 ({total_q})"
        )

        if total_k_rows > 0:
            k_u8 = plan.ragged_k_u8
            k_scale = plan.ragged_k_scale
            assert k_u8 is not None and k_scale is not None
            pool = _token_pool_from_batch(forward_batch)
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=self._get_index_k_read_buffer(pool, layer_id),
                page_indices=plan.ragged_concat_page_table,
                seq_len=total_k_rows,
                k_out=k_u8,
                scale_out=k_scale,
            )
            k_fp8 = k_u8.view(torch.float8_e4m3fn)
            if self._should_use_eager_logits(q_fp8):
                topk_method = getattr(metadata, "topk_transform_method", None)
                attn_metadata = getattr(metadata, "attn_metadata", None)
                page_table_all = None
                page_table_row_index_all = None
                topk_offsets_all = None
                if envs.SGLANG_NSA_FUSE_TOPK.get():
                    if getattr(topk_method, "name", "") == "PAGED":
                        page_table_all = plan.ragged_paged_page_table
                        page_table_row_index_all = (
                            plan.ragged_paged_page_table_row_index
                        )
                    elif getattr(topk_method, "name", "") == "RAGGED":
                        topk_offsets_all = getattr(
                            attn_metadata, "topk_indices_offset", None
                        )

                return self._topk_from_glm5_next_eager_logits_rows(
                    q_fp8[:n_real].contiguous(),
                    (k_fp8.contiguous(), k_scale.contiguous()),
                    weights[:n_real].contiguous(),
                    pool_lens,
                    seq_lens_expanded,
                    ks_per_q,
                    ke_per_q,
                    total_q=total_q,
                    page_table=page_table_all,
                    topk_offsets=topk_offsets_all,
                    page_table_row_index=page_table_row_index_all,
                )

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[:n_real].contiguous(),
                (k_fp8.contiguous(), k_scale.contiguous()),
                weights[:n_real].contiguous(),
                ks_per_q,
                ke_per_q,
                clean_logits=True,
            )
        else:
            logits = torch.empty((n_real, 0), dtype=torch.float32, device=device)

        topk_method = getattr(metadata, "topk_transform_method", None)
        attn_metadata = getattr(metadata, "attn_metadata", None)
        page_table_all = None
        page_table_row_index_all = None
        topk_offsets_all = None
        if envs.SGLANG_NSA_FUSE_TOPK.get():
            if getattr(topk_method, "name", "") == "PAGED":
                page_table_all = plan.ragged_paged_page_table
                page_table_row_index_all = plan.ragged_paged_page_table_row_index
            elif getattr(topk_method, "name", "") == "RAGGED":
                topk_offsets_all = getattr(attn_metadata, "topk_indices_offset", None)

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
            out_rows=total_q,
            page_table_row_index=page_table_row_index_all,
        )

    def _topk_from_glm5_next_eager_logits_rows(
        self,
        q_fp8: torch.Tensor,
        kv_fp8: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        pool_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        ks: torch.Tensor,
        ke: torch.Tensor,
        *,
        total_q: int,
        page_table: Optional[torch.Tensor],
        topk_offsets: Optional[torch.Tensor],
        page_table_row_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Score/select fixed query-row chunks for exact-GLM SM120 prefill."""

        from sglang.srt.layers.attention.nsa.glm5_next_indexer_logits import (
            iter_glm5_next_eager_fp8_mqa_logits,
        )

        n_real = q_fp8.shape[0]
        if total_q < n_real:
            raise ValueError(
                "GLM-5-Next total query rows cannot be smaller than real queries"
            )
        if not (
            pool_lens.shape[0]
            == seq_lens.shape[0]
            == ks.shape[0]
            == ke.shape[0]
            == n_real
        ):
            raise ValueError("GLM-5-Next ragged row metadata must match real queries")
        if topk_offsets is not None and topk_offsets.shape[0] != n_real:
            raise ValueError("GLM-5-Next ragged topk offsets must match real queries")
        if page_table_row_index is not None and page_table_row_index.shape[0] != n_real:
            raise ValueError("GLM-5-Next paged row mapping must match real queries")
        if (page_table is None) != (page_table_row_index is None):
            raise ValueError(
                "GLM-5-Next paged table and row mapping must be provided together"
            )

        # Consume each fixed query-row logits chunk immediately. At the final
        # 500K boundary this avoids simultaneously retaining multi-GiB
        # [4096, 125056] logits/mask tensors. The complete K payload is
        # converted only once inside the iterator.
        topk_result = torch.full(
            (total_q, self.index_topk + self.index_kpool - 1),
            -1,
            dtype=torch.int32,
            device=q_fp8.device,
        )
        logits_rows = iter_glm5_next_eager_fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
        )
        for q_start, q_end, logits_chunk in logits_rows:
            topk_offsets_chunk = (
                None if topk_offsets is None else topk_offsets[q_start:q_end]
            )
            page_table_row_index_chunk = (
                None
                if page_table_row_index is None
                else page_table_row_index[q_start:q_end]
            )
            topk_result[q_start:q_end] = self._topk_from_kpool_logits(
                logits_chunk,
                pool_lens[q_start:q_end],
                seq_lens=seq_lens[q_start:q_end],
                page_table=page_table,
                topk_offsets=topk_offsets_chunk,
                row_starts=ks[q_start:q_end],
                page_table_row_index=page_table_row_index_chunk,
            )
            del logits_chunk
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        assert forward_batch.forward_mode.is_extend_without_speculative()
        assert _token_pool_from_batch(forward_batch).page_size == 64, (
            "only support page size 64"
        )
        assert getattr(metadata.attn_metadata, "kpool_extend_plan", None) is not None
        return self._get_topk_ragged_kpool_plan(
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
        )

    def _forward_cuda_skip_logits(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()

        key = self._get_k_bf16(x, positions)
        self._compress_write(
            x=x,
            key=key,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            metadata=metadata,
        )

        if not return_indices:
            return None

        x_meta = x[0] if isinstance(x, tuple) else x
        return self._full_topk_for_short_sequence(metadata, x_meta.device)

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        """GLM-5-Next KPool forward path.

        Decode (including CUDA-graph replay) and non-speculative extend are
        supported. CP and MTP remain outside this model-local contract.
        """
        if is_hip():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not is_npu():
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(_token_pool_from_batch(forward_batch), NSATokenToKVPool)

        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
        if metadata is None:
            return None

        x_meta = x[0] if isinstance(x, tuple) else x
        assert forward_batch.seq_lens_cpu is not None
        mode = forward_batch.forward_mode
        if mode.is_idle() or len(forward_batch.seq_lens_cpu) == 0:
            return torch.full(
                (x_meta.shape[0], self.index_topk + self.index_kpool - 1),
                -1,
                dtype=torch.int,
                device=x_meta.device,
            )
        if not (mode.is_decode() or mode.is_extend_without_speculative()):
            raise NotImplementedError(
                "GLM-5-Next KPool supports decode (including CUDA graphs) and "
                "non-speculative extend only"
            )

        if mode.is_extend_without_speculative():
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            if max_kv_len <= self.index_topk:
                return self._forward_cuda_skip_logits(
                    x,
                    positions,
                    forward_batch,
                    layer_id,
                    act_quant,
                    metadata,
                    return_indices,
                )

        query, key, gate_score = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream=False,
            forward_batch=forward_batch,
            precompute_compress_gate=False,
        )
        q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
        self._compress_write(
            x=x,
            key=key,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            metadata=metadata,
            gate_score=gate_score,
        )
        if not return_indices:
            return None

        weights = self._get_logits_head_gate(x, q_scale)
        if mode.is_decode():
            return self._get_topk_paged(
                forward_batch, layer_id, q_fp8, weights, metadata
            )
        return self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)
