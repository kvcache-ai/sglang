"""Configuration and capability gates for GLM-5-Next.

The text implementation lands before the multimodal implementation.  Keep the
vision configuration serializable here, but deliberately avoid importing the
GLM-OCR vision stack.  Stage 7 can replace the lightweight holder with the
runtime vision implementation without changing text-only config loading.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape

_GLM5_NEXT_ARCH = "Glm5NextForConditionalGeneration"


class Glm5NextGPUProfile(str, Enum):
    """Validated cache/kernel policy for one GLM-5-Next GPU family.

    Consumer GPUs deliberately use explicit profiles instead of inheriting the
    generic DeepSeek NSA capability checks.  In particular, SM86 cannot execute
    the E4M3 tensor-core index-cache path and therefore keeps both the latent and
    KPool caches in BF16.
    """

    SM86_BF16 = "sm86_bf16"
    SM89_FP8 = "sm89_fp8"
    BLACKWELL_FP8 = "blackwell_fp8"

    @property
    def kv_cache_dtype(self) -> str:
        return (
            "bfloat16"
            if self is Glm5NextGPUProfile.SM86_BF16
            else "fp8_e4m3"
        )

    @property
    def index_cache_dtype(self) -> str:
        return "bfloat16" if self is Glm5NextGPUProfile.SM86_BF16 else "fp8_e4m3"

    @property
    def is_consumer_gpu(self) -> bool:
        return self in (
            Glm5NextGPUProfile.SM86_BF16,
            Glm5NextGPUProfile.SM89_FP8,
        )


def get_glm5_next_gpu_profile(
    capability: tuple[int, int],
) -> Glm5NextGPUProfile:
    """Resolve the exact GLM runtime policy or fail closed.

    Keep the pre-existing major>=10 acceptance boundary intact while adding
    only the two requested consumer architectures.  SM90 and other Ampere/Ada
    variants remain unsupported until separately validated.
    """

    if capability == (8, 6):
        return Glm5NextGPUProfile.SM86_BF16
    if capability == (8, 9):
        return Glm5NextGPUProfile.SM89_FP8
    if capability[0] >= 10:
        return Glm5NextGPUProfile.BLACKWELL_FP8
    raise ValueError(
        "GLM-5-Next supports NVIDIA SM86, SM89, or Blackwell (SM>=100); "
        f"got SM{capability[0]}{capability[1]}."
    )

_GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS = (
    "architectures",
    "vocab_size",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "partial_rotary_factor",
    "tie_word_embeddings",
    "attention_bias",
    "attention_dropout",
    "n_routed_experts",
    "num_experts_per_tok",
    "n_shared_experts",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "routed_scaling_factor",
    "scoring_func",
    "topk_method",
    "first_k_dense_replace",
    "moe_layer_freq",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "swiglu_limit",
    "mhc",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "num_nextn_predict_layers",
    "linear_attn_config",
    "linear_head_dim",
    "linear_num_heads",
    "linear_conv_kernel_dim",
    "linear_lower_bound",
    "gate_lower_bound",
    "index_head_dim",
    "index_topk",
    "index_kpool",
    "index_kpool_always_select_tail",
    "index_kpool_compress",
    "index_n_heads",
    "index_topk_freq",
    "index_topk_pattern",
    "index_skip_topk_offset",
    "index_share_for_mtp_iteration",
    "indexer_rope_interleave",
    "layer_types",
    "mlp_layer_types",
    "quantization_config",
)


class _Glm5NextVisionConfigHolder(PretrainedConfig):
    """GLM-5-Next vision config without importing the vision runtime."""

    model_type = "glm_ocr_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 4096,
        projection_intermediate_size: int = 10240,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        image_size: int = 448,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.projection_intermediate_size = projection_intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        head_dim: Optional[int] = 0,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: Optional[int] = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = 154820,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        rope_theta: float = 800000.0,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        partial_rotary_factor: float = 1.0,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        n_routed_experts: Optional[int] = 288,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        swiglu_limit: Optional[float] = 10.0,
        mhc: bool = True,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        num_nextn_predict_layers: int = 1,
        linear_attn_config: Optional[dict] = None,
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: Optional[float] = -5.0,
        gate_lower_bound: Optional[float] = None,
        index_head_dim: Optional[int] = 128,
        index_topk: Optional[int] = 2048,
        index_kpool: int = 4,
        index_kpool_always_select_tail: bool = True,
        index_kpool_compress: bool = True,
        index_n_heads: Optional[int] = 32,
        index_topk_freq: int = 1,
        index_topk_pattern: Optional[str] = None,
        index_skip_topk_offset: Optional[int] = None,
        index_share_for_mtp_iteration: bool = False,
        indexer_rope_interleave: bool = True,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        **kwargs,
    ):
        if eos_token_id is None:
            eos_token_id = [154820, 154827, 154829]
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get(
                "partial_rotary_factor", partial_rotary_factor
            )

        if layer_types is None:
            layer_types = [
                "linear_attention"
                if layer_idx % 4 != 3
                else "deepseek_sparse_attention"
                for layer_idx in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "GLM-5-Next layer_types must contain exactly "
                f"num_hidden_layers={num_hidden_layers} entries, got {len(layer_types)}"
            )
        unsupported_layer_types = set(layer_types) - {
            "linear_attention",
            "deepseek_sparse_attention",
        }
        if unsupported_layer_types:
            raise ValueError(
                f"Unsupported GLM-5-Next layer types: {sorted(unsupported_layer_types)}"
            )

        if mlp_layer_types is None:
            mlp_layer_types = [
                "dense" if layer_idx < first_k_dense_replace else "sparse"
                for layer_idx in range(num_hidden_layers)
            ]
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError(
                "GLM-5-Next mlp_layer_types must contain exactly "
                f"num_hidden_layers={num_hidden_layers} entries, got {len(mlp_layer_types)}"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.swiglu_limit = swiglu_limit
        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        self.gate_lower_bound = (
            gate_lower_bound if gate_lower_bound is not None else linear_lower_bound
        )
        # The pinned checkpoint names DSA layers
        # ``deepseek_sparse_attention``.  transformers-kt 5.6 validates the
        # generic ``PreTrainedConfig.layer_types`` field against its closed
        # vocabulary, where the corresponding public name is ``sparse``.
        # Keep the checkpoint-native names separately for provenance while
        # exposing validator-compatible generic names to shared Transformers
        # and SGLang helpers.  GLM runtime selection remains driven by the
        # exact ``linear_attn_config`` partition below.
        self._glm5_next_checkpoint_layer_types = list(layer_types)
        self.layer_types = [
            "linear_attention"
            if layer_type == "linear_attention"
            else "sparse"
            for layer_type in layer_types
        ]
        self.mlp_layer_types = list(mlp_layer_types)

        if linear_attn_config is None:
            kda_layers = [
                layer_idx
                for layer_idx, layer_type in enumerate(
                    self._glm5_next_checkpoint_layer_types
                )
                if layer_type == "linear_attention"
            ]
            kda_layer_set = set(kda_layers)
            linear_attn_config = {
                "full_attn_layers": [
                    layer_idx
                    for layer_idx in range(num_hidden_layers)
                    if layer_idx not in kda_layer_set
                ],
                "head_dim": linear_head_dim,
                "kda_layers": kda_layers,
                "num_heads": linear_num_heads,
                "short_conv_kernel_size": linear_conv_kernel_dim,
                "gate_lower_bound": self.gate_lower_bound,
            }
        self._validate_linear_attn_config(linear_attn_config)
        self.linear_attn_config = linear_attn_config

        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.index_kpool_compress = index_kpool_compress
        self.index_n_heads = index_n_heads
        self.index_topk_freq = index_topk_freq
        self.index_topk_pattern = index_topk_pattern
        self.index_skip_topk_offset = index_skip_topk_offset
        self.index_share_for_mtp_iteration = index_share_for_mtp_iteration
        self.indexer_rope_interleave = indexer_rope_interleave

        # Compatibility aliases used by the existing KT Kimi/DeepSeek model
        # skeleton.  They mirror the same values; no non-GLM config is changed.
        self.num_experts = n_routed_experts
        self.num_experts_per_token = num_experts_per_tok
        self.num_shared_experts = n_shared_experts
        self.num_expert_group = n_group
        self.moe_renormalize = norm_topk_prob
        self.moe_router_activation_func = scoring_func
        self.use_grouped_topk = True
        self.mla_use_nope = qk_rope_head_dim == 0

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if rope_parameters is not None or rope_scaling is not None:
            self.rope_parameters = rope_parameters or rope_scaling

    def _validate_linear_attn_config(self, linear_attn_config: dict) -> None:
        required = {
            "full_attn_layers",
            "head_dim",
            "kda_layers",
            "num_heads",
            "short_conv_kernel_size",
        }
        missing = required - set(linear_attn_config)
        if missing:
            raise ValueError(
                f"GLM-5-Next linear_attn_config is missing {sorted(missing)}"
            )
        kda_layers = list(linear_attn_config["kda_layers"])
        full_attn_layers = list(linear_attn_config["full_attn_layers"])
        if set(kda_layers) & set(full_attn_layers):
            raise ValueError("KDA and full-attention layer sets must be disjoint")
        if sorted(kda_layers + full_attn_layers) != list(range(self.num_hidden_layers)):
            raise ValueError(
                "KDA and full-attention layers must partition every decoder layer"
            )

    @property
    def is_mla(self) -> bool:
        return True

    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.linear_attn_config["kda_layers"])

    def is_kda_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.linear_attn_config["kda_layers"]

    @property
    def linear_layer_ids(self) -> List[int]:
        return list(self.linear_attn_config["kda_layers"])

    @property
    def nextn_layer_ids(self) -> List[int]:
        num_nextn_layers = self.num_nextn_predict_layers or 0
        return [self.num_hidden_layers + i for i in range(num_nextn_layers)]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        return list(self.linear_attn_config["full_attn_layers"])

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        # Import lazily so config parsing remains CPU-only and side-effect free.
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = KimiLinearStateShape.create(
            tp_world_size=get_attention_tp_size(),
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
        )
        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {
        "vision_config": _Glm5NextVisionConfigHolder,
        "text_config": Glm5NextTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        **kwargs,
    ):
        kwargs.setdefault("architectures", [_GLM5_NEXT_ARCH])
        top_level_text_config = {
            key: kwargs[key]
            for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS
            if key in kwargs
        }

        if isinstance(vision_config, dict):
            self.vision_config = _Glm5NextVisionConfigHolder(**vision_config)
        elif vision_config is None:
            self.vision_config = _Glm5NextVisionConfigHolder()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = Glm5NextTextConfig(
                **{**top_level_text_config, **text_config}
            )
        elif text_config is None:
            self.text_config = Glm5NextTextConfig(**top_level_text_config)
        else:
            self.text_config = text_config

        # A legacy top-level ``layer_types`` copy is present in the pinned
        # checkpoint.  The nested text config above has already validated and
        # normalized it, so do not let the root PreTrainedConfig validate the
        # raw checkpoint spelling a second time.
        if "layer_types" in kwargs:
            kwargs["layer_types"] = list(self.text_config.layer_types)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(**kwargs)
        for key in _GLM5_NEXT_TOP_LEVEL_CONFIG_KEYS:
            if hasattr(self.text_config, key):
                setattr(self, key, getattr(self.text_config, key))


def _unwrap_glm5_next_text_config(config: Any) -> tuple[Any, Any]:
    root = getattr(config, "hf_config", config)
    return root, getattr(root, "text_config", root)


def is_glm5_next(config: Any) -> bool:
    root, text = _unwrap_glm5_next_text_config(config)
    root_type = getattr(root, "model_type", None)
    text_type = getattr(text, "model_type", None)
    architectures = getattr(root, "architectures", None) or getattr(
        text, "architectures", None
    )
    architecture_matches = architectures is None or _GLM5_NEXT_ARCH in architectures
    return architecture_matches and (
        root_type == Glm5NextConfig.model_type
        or text_type == Glm5NextTextConfig.model_type
    )


def uses_kpool4_compress(config: Any) -> bool:
    _, text = _unwrap_glm5_next_text_config(config)
    return (
        is_glm5_next(config)
        and getattr(text, "index_kpool", None) == 4
        and getattr(text, "index_kpool_compress", False) is True
        and getattr(text, "index_kpool_always_select_tail", False) is True
    )


def uses_kda_safe_gate(config: Any) -> bool:
    _, text = _unwrap_glm5_next_text_config(config)
    linear_attn_config = getattr(text, "linear_attn_config", None)
    return (
        is_glm5_next(config)
        and isinstance(linear_attn_config, dict)
        and bool(linear_attn_config.get("kda_layers"))
        and linear_attn_config.get("gate_lower_bound") is not None
    )


def uses_zero_rope_mla(config: Any) -> bool:
    _, text = _unwrap_glm5_next_text_config(config)
    full_attention_layer_ids = getattr(text, "full_attention_layer_ids", [])
    return (
        is_glm5_next(config)
        and bool(full_attention_layer_ids)
        and getattr(text, "qk_rope_head_dim", None) == 0
    )


def uses_mhc(config: Any) -> bool:
    _, text = _unwrap_glm5_next_text_config(config)
    return is_glm5_next(config) and getattr(text, "mhc", False) is True


@dataclass(frozen=True)
class Glm5NextCapabilities:
    is_glm5_next: bool
    uses_kpool4_compress: bool
    uses_kda_safe_gate: bool
    uses_zero_rope_mla: bool
    uses_mhc: bool

    @classmethod
    def from_config(cls, config: Any) -> "Glm5NextCapabilities":
        return cls(
            is_glm5_next=is_glm5_next(config),
            uses_kpool4_compress=uses_kpool4_compress(config),
            uses_kda_safe_gate=uses_kda_safe_gate(config),
            uses_zero_rope_mla=uses_zero_rope_mla(config),
            uses_mhc=uses_mhc(config),
        )


def get_glm5_next_capabilities(config: Any) -> Glm5NextCapabilities:
    return Glm5NextCapabilities.from_config(config)
