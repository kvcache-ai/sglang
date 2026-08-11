"""GLM-5-Next's zero-RoPE DSA attention boundary.

Keep the KPool-specific construction out of the shared DeepSeek attention
class.  This lets GLM opt in to its 4-token pooled index while the existing
DeepSeek/Kimi models keep their current indexer and checkpoint namespaces.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.layers.attention.nsa.nsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.utils import add_prefix


def is_glm5_next_dsa_config(config: Any) -> bool:
    """Return whether ``config`` is the canonical GLM-5-Next text config."""

    return isinstance(config, Glm5NextTextConfig) and (
        getattr(config, "model_type", None) == Glm5NextTextConfig.model_type
    )


def _resolve_config_value(config: Glm5NextTextConfig, name: str, value: Any) -> Any:
    """Use the canonical config value and reject a divergent factory input."""

    expected = getattr(config, name)
    if value is None:
        return expected
    if value != expected:
        raise ValueError(f"GLM-5-Next DSA requires {name}={expected!r}, got {value!r}")
    return value


def _validate_glm5_next_dsa_config(config: Glm5NextTextConfig) -> None:
    if not is_glm5_next_dsa_config(config):
        raise TypeError(
            "Glm5NextDSAAttention only accepts Glm5NextTextConfig; "
            f"got {type(config).__name__}"
        )

    # These are checkpoint-format invariants, not tunable runtime options.
    assert config.index_kpool == 4, (
        f"GLM-5-Next requires index_kpool=4, got {config.index_kpool}"
    )
    assert config.index_kpool_compress is True, (
        "GLM-5-Next requires index_kpool_compress=True"
    )
    assert config.index_kpool_always_select_tail is True, (
        "GLM-5-Next requires index_kpool_always_select_tail=True"
    )

    if config.index_topk is None or config.index_topk <= 0:
        raise ValueError("GLM-5-Next DSA requires a positive index_topk")
    if config.index_topk % config.index_kpool != 0:
        raise ValueError(
            "GLM-5-Next index_topk must be divisible by index_kpool, got "
            f"{config.index_topk} and {config.index_kpool}"
        )
    if config.q_lora_rank is None:
        raise ValueError("GLM-5-Next KPool requires q_lora_rank")
    if config.qk_rope_head_dim != 0:
        raise ValueError("GLM-5-Next DSA is zero-RoPE and requires qk_rope_head_dim=0")

    # KPool's CP/MTP and cross-layer top-k reuse contracts are deliberately
    # not enabled yet.  Refuse those modes instead of silently entering the
    # generic DeepSeek paths with a wider (index_topk + 3) index tensor.
    if config.index_topk_freq != 1:
        raise ValueError("GLM-5-Next KPool does not support index_topk_freq != 1")
    if config.index_topk_pattern is not None:
        raise ValueError("GLM-5-Next KPool does not support index_topk_pattern")
    if config.index_skip_topk_offset is not None:
        raise ValueError("GLM-5-Next KPool does not support index_skip_topk_offset")
    if config.index_share_for_mtp_iteration is not False:
        raise ValueError("GLM-5-Next KPool does not support index sharing for MTP")


class Glm5NextDSAAttention(DeepseekV2AttentionMLA):
    """DeepSeek MLA tensor layout with GLM's KPool4 DSA indexer."""

    def _fuse_rope_for_trtllm_mla(self, forward_batch) -> bool:
        """Keep zero-RoPE GLM out of the base fused-RoPE metadata path.

        The KT DeepSeek base enables fused RoPE whenever the NSA TRTLLM
        backend stores FP8 KV.  GLM deliberately has no rotary embedding, so
        that generic branch would dereference ``self.rotary_emb`` even though
        the query/key RoPE width is zero.  Returning false still lets the NSA
        backend quantize the native 512-wide latent cache; it only suppresses
        nonexistent cosine/sine metadata.
        """

        del forward_batch
        return False

    def __init__(
        self,
        config: Glm5NextTextConfig,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        rope_theta: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: Optional[int] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = True,
        is_nextn: bool = False,
    ) -> None:
        _validate_glm5_next_dsa_config(config)
        if is_nsa_enable_prefill_cp():
            raise NotImplementedError(
                "GLM-5-Next KPool does not support NSA prefill context parallel"
            )
        if is_nextn:
            raise NotImplementedError("GLM-5-Next KPool does not support MTP layers")
        if not skip_rope:
            raise ValueError("GLM-5-Next DSA requires skip_rope=True")
        if layer_id is None:
            raise ValueError("GLM-5-Next DSA requires a concrete layer_id")

        hidden_size = _resolve_config_value(config, "hidden_size", hidden_size)
        num_heads = _resolve_config_value(config, "num_attention_heads", num_heads)
        qk_nope_head_dim = _resolve_config_value(
            config, "qk_nope_head_dim", qk_nope_head_dim
        )
        qk_rope_head_dim = _resolve_config_value(
            config, "qk_rope_head_dim", qk_rope_head_dim
        )
        v_head_dim = _resolve_config_value(config, "v_head_dim", v_head_dim)
        q_lora_rank = _resolve_config_value(config, "q_lora_rank", q_lora_rank)
        kv_lora_rank = _resolve_config_value(config, "kv_lora_rank", kv_lora_rank)
        rope_theta = _resolve_config_value(config, "rope_theta", rope_theta)
        max_position_embeddings = _resolve_config_value(
            config, "max_position_embeddings", max_position_embeddings
        )
        if rope_scaling is None:
            rope_scaling = config.rope_scaling
        elif rope_scaling != config.rope_scaling:
            raise ValueError("GLM-5-Next DSA rope_scaling must match the text config")

        # The subclass owns indexer construction.  Suppress the generic NSA
        # branch without mutating the real text config or allocating a
        # temporary DeepSeek Indexer; every other base forward dependency is
        # still initialized by DeepseekV2AttentionMLA.
        base_config = copy.copy(config)
        base_config.index_topk = None
        base_rope_scaling = copy.deepcopy(rope_scaling)
        super().__init__(
            config=base_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=base_rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            reduce_results=reduce_results,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
            skip_rope=True,
            is_nextn=False,
        )

        self.config = config
        self.use_nsa = True
        self.nsa_enable_prefill_cp = False
        self.is_nextn = False
        self.skip_rope = True
        self.rotary_emb = None

        # GLM computes top-k independently in every DSA layer.  Keeping these
        # attributes explicit is required by the inherited absorb forward.
        self.index_topk_freq = 1
        self.index_topk_pattern = None
        self.index_skip_topk_offset = None
        self.skip_topk = False
        self.next_skip_topk = False

        self.indexer = IndexerKPool(
            hidden_size=hidden_size,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            rope_head_dim=qk_rope_head_dim,
            index_topk=config.index_topk,
            q_lora_rank=q_lora_rank,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            scale_fmt="ue8m0",
            block_size=128,
            rope_scaling=copy.deepcopy(rope_scaling),
            is_neox_style=not config.indexer_rope_interleave,
            prefix=add_prefix("indexer", prefix),
            quant_config=quant_config,
            layer_id=layer_id,
            alt_stream=alt_stream,
            skip_rope=True,
            config=config,
        )

        # KPool always reserves three tail slots in addition to the 2048
        # pooled-history choices in the released GLM config.
        self.index_topk_output_width = config.index_topk + config.index_kpool - 1


__all__ = ["Glm5NextDSAAttention", "is_glm5_next_dsa_config"]
