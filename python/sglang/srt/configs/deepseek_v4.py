from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class DeepSeekV4Config(PretrainedConfig):
    """DeepSeek V4 / V4-Flash config.

    Uses the traditional ``__init__`` kwargs idiom (matching the rest of
    sglang/configs/) rather than ``@dataclass`` because the auto-application
    of ``@dataclass`` by ``PretrainedConfig.__init_subclass__`` differs across
    transformers / transformers-kt versions: in some builds the
    ``field(default_factory=...)`` markers stay as raw ``Field`` objects on
    the class, leaving ``self.quantization_config`` un-defaulted and breaking
    ``to_diff_dict``. The ``__init__`` form is portable.
    """

    model_type = "deepseek_v4"

    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: Optional[int] = None,
        ep_size: int = 1,
        first_k_dense_replace: int = 0,
        hidden_act: str = "silu",
        hidden_size: int = 4096,
        index_head_dim: int = 128,
        index_n_heads: int = 64,
        index_topk: int = 512,
        initializer_range: float = 0.02,
        intermediate_size: int = 2048,
        kv_lora_rank: int = 512,
        max_position_embeddings: int = 65536,
        moe_intermediate_size: int = 2048,
        moe_layer_freq: int = 1,
        n_group: int = 8,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        num_attention_heads: int = 64,
        num_experts_per_tok: int = 6,
        num_hidden_layers: int = 43,
        num_key_value_heads: int = 1,
        q_lora_rank: int = 1024,
        qk_nope_head_dim: int = 448,
        qk_rope_head_dim: int = 64,
        quantization_config: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-6,
        rope_scaling: Optional[Dict[str, float]] = None,
        rope_theta: int = 10000,
        routed_scaling_factor: float = 1.5,
        scoring_func: str = "sqrtsoftplus",
        tie_word_embeddings: bool = False,
        topk_group: int = 8,
        topk_method: str = "noaux_tc",
        use_cache: bool = True,
        v_head_dim: int = 512,
        vocab_size: int = 129280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        window_size: int = 128,
        compress_rope_theta: int = 40000,
        compress_ratios: Optional[List[int]] = None,
        n_hash_layers: int = 3,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        **kwargs,
    ):
        # Mutable defaults are constructed inside __init__ to avoid the
        # shared-default-list bug.
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.kv_lora_rank = kv_lora_rank
        self.max_position_embeddings = max_position_embeddings
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.num_attention_heads = num_attention_heads
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.quantization_config = quantization_config if quantization_config is not None else {}
        self.rms_norm_eps = rms_norm_eps
        self.rope_scaling = rope_scaling if rope_scaling is not None else {}
        self.rope_theta = rope_theta
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.use_cache = use_cache
        self.v_head_dim = v_head_dim
        self.vocab_size = vocab_size
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.window_size = window_size
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios if compress_ratios is not None else []
        self.n_hash_layers = n_hash_layers
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
