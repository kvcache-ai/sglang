"""Text-only GLM-5-Next model skeleton.

This module intentionally reuses the KT Kimi/DeepSeek building blocks.  The
GLM-specific KDA, DSA/KPool, mHC, and vision implementations are wired through
small seams so that those later integrations do not have to modify the shared
Kimi model or loosen behavior for existing architectures.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Iterator
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.communicator import (
    LayerScatterModes,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.kimi_linear import (
    KimiDecoderLayer,
    KimiDeltaAttention,
    KimiLinearModel,
)
from sglang.srt.models.glm5_next_dsa import Glm5NextDSAAttention
from sglang.srt.models.glm5_next_moe import Glm5NextMLP, Glm5NextMoE
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.utils import make_layers
from sglang.srt.utils.common import BumpAllocator


# Vision support is scheduled for phase 7.  Keep this whitelist exact: a
# substring check such as ``"visual" in name`` can accidentally discard a
# future text parameter and hide a corrupt checkpoint.
GLM5_NEXT_PHASE7_VISUAL_WEIGHT_PREFIXES = ("visual.", "model.visual.")

_GLM5_NEXT_LAYER_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.")
_GLM5_NEXT_EXPERT_PARAMETER_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<kind>w13|w2)_(?P<leaf>.+)$"
)
_GLM5_NEXT_RUNTIME_KV_SCALE_RE = re.compile(
    r"^model\.layers\.\d+\.self_attn\.attn_m(?:ha|qa)\.[kv]_scale$"
)


def _glm5_next_checkpoint_source_contract(
    parameter_names: Iterable[str],
    *,
    num_experts: int,
    packed_modules_mapping: dict[str, list[str]],
) -> tuple[frozenset[str], frozenset[str]]:
    """Reverse the runtime parameter namespace into required HF source names.

    A target parameter is not always a complete checkpoint-loading unit.  For
    example, ``qkv_proj.weight`` needs three checkpoint shards, and a routed
    ``w13_weight`` needs gate/up shards for every logical expert.  Tracking the
    target parameter name alone would therefore miss a truncated packed shard.

    The returned sets are rank-local: PP constructs only the layers owned by
    the current rank, while TP keeps the same names and lets each parameter's
    loader validate and shard the source tensor shape.
    """

    expected_sources: set[str] = set()
    runtime_defaults: set[str] = set()

    for parameter_name in parameter_names:
        if _GLM5_NEXT_RUNTIME_KV_SCALE_RE.fullmatch(parameter_name):
            # BaseKVCacheMethod owns these scalar defaults.  The pinned FP8
            # checkpoint intentionally has no serialized K/V cache scales.
            runtime_defaults.add(parameter_name)
            continue

        expert_match = _GLM5_NEXT_EXPERT_PARAMETER_RE.fullmatch(parameter_name)
        if expert_match is not None:
            prefix = expert_match.group("prefix")
            kind = expert_match.group("kind")
            leaf = expert_match.group("leaf")
            checkpoint_projections = (
                ("gate_proj", "up_proj") if kind == "w13" else ("down_proj",)
            )
            for expert_id in range(num_experts):
                for projection in checkpoint_projections:
                    expected_sources.add(f"{prefix}.{expert_id}.{projection}.{leaf}")
            continue

        if ".mlp.experts." in parameter_name:
            raise RuntimeError(
                "GLM-5-Next Session AB encountered an unsupported routed-expert "
                f"runtime parameter: {parameter_name!r}"
            )

        for packed_name, checkpoint_names in packed_modules_mapping.items():
            packed_segment = f".{packed_name}."
            if packed_segment not in parameter_name:
                continue
            expected_sources.update(
                parameter_name.replace(
                    packed_segment,
                    f".{checkpoint_name}.",
                )
                for checkpoint_name in checkpoint_names
            )
            break
        else:
            expected_sources.add(parameter_name)

    return frozenset(expected_sources), frozenset(runtime_defaults)


def normalize_glm5_next_weight_name(name: str) -> Optional[str]:
    """Map the HF wrapper prefix to the text-only SGLang module namespace.

    ``None`` means that the tensor belongs to the explicitly deferred phase-7
    vision tower.  No other unknown weight is classified as skippable here.
    """

    if name.startswith(GLM5_NEXT_PHASE7_VISUAL_WEIGHT_PREFIXES):
        return None

    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name.startswith("language_model."):
        name = "model." + name.removeprefix("language_model.")

    return name


def _kda_construction_config(config: Glm5NextTextConfig) -> Glm5NextTextConfig:
    """Return a non-mutating Kimi compatibility view for GLM KDA layers.

    GLM uses 256-wide values in its DSA layers, while its KDA Q/K/V heads are
    all 128-wide.  Kimi's current KDA constructor reads ``v_head_dim`` from the
    common config, so passing the root config would incorrectly make only KDA
    V 256-wide.  A shallow copy keeps the source HF config untouched.
    """

    kda_config = copy.copy(config)
    kda_config.linear_num_heads = config.linear_attn_config["num_heads"]
    kda_config.linear_head_dim = config.linear_attn_config["head_dim"]
    kda_config.v_head_dim = config.linear_attn_config["head_dim"]
    return kda_config


class Glm5NextLinearAttention(KimiDeltaAttention):
    """Phase-3 construction seam backed by KT's existing KDA module."""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        construction_config = _kda_construction_config(config)
        super().__init__(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            config=construction_config,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            prefix=prefix,
        )
        # Preserve the canonical config for capability checks and diagnostics;
        # all dimensions consumed during construction are stored on the module.
        self.config = config
        self.attn.lower_bound = config.linear_attn_config["gate_lower_bound"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        """Pass GLM's raw gate and beta logits to its dedicated backend."""

        del positions, zero_allocator
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        if self.do_fuse_qkvbfg:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg_fused(
                hidden_states
            )
        else:
            mixed_qkv, beta, forget_gate, g_proj_states = self.forward_qkvbfg(
                hidden_states
            )

        # GLM applies neither fused_kda_gate nor sigmoid here.  Its dedicated
        # backend activates both raw tensors identically in prefill and decode.
        if not forward_batch.forward_mode.is_decode():
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )
        norm_gate = g_proj_states.unflatten(-1, (-1, self.head_dim))
        core_attn_out = self.o_norm(core_attn_out, norm_gate)
        core_attn_out = core_attn_out.squeeze(0).flatten(-2)
        return self.o_proj(core_attn_out)[0]


def build_glm5_next_attention(
    *,
    config: Glm5NextTextConfig,
    layer_idx: int,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
    alt_stream: Optional[torch.cuda.Stream] = None,
) -> nn.Module:
    """Build one attention layer through a GLM-only integration seam."""

    if config.is_kda_layer(layer_idx):
        return Glm5NextLinearAttention(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

    return Glm5NextDSAAttention(
        config=config,
        layer_id=layer_idx,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        quant_config=quant_config,
        prefix=prefix,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
        alt_stream=alt_stream,
        skip_rope=True,
    )


class Glm5NextDecoderLayer(KimiDecoderLayer):
    """GLM layer selection with an opt-in mHC residual state machine."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        nn.Module.__init__(self)
        if config.mhc is not True:
            raise ValueError(
                "GLM-5-Next Session AB requires the checkpoint-native mhc=True "
                "execution path."
            )
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.alt_stream = alt_stream
        self.is_moe = config.is_moe
        self.mhc_enabled = True

        self.self_attn = build_glm5_next_attention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            alt_stream=alt_stream,
        )
        if self.mhc_enabled:
            # The mHC communicator owns the attention-output TP all-reduce.
            # Changing the GLM instance leaves the shared attention factory and
            # the historical non-mHC construction path untouched.
            self.self_attn.o_proj.reduce_results = False

        if (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            # Register the sparse module only under ``mlp``.  Registering the
            # same module first as ``block_sparse_moe`` makes
            # ``named_parameters()`` keep that first namespace and prevents
            # the DeepSeek loader from matching the checkpoint's ``mlp.*``
            # tensors.
            self.mlp = Glm5NextMoE(
                config=config,
                quant_config=quant_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                alt_stream=alt_stream,
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc_enabled:
            hc_mult = config.hc_mult
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * config.hidden_size

            # These names and FP32 shapes match the checkpoint verbatim.
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )

            is_layer_sparse = self._is_layer_sparse(layer_idx)
            layer_scatter_modes = LayerScatterModes.init_new(
                layer_id=layer_idx,
                num_layers=config.num_hidden_layers,
                is_layer_sparse=is_layer_sparse,
                is_previous_layer_sparse=self._is_layer_sparse(layer_idx - 1),
                is_next_layer_sparse=self._is_layer_sparse(layer_idx + 1),
            )
            if any(
                mode is ScatterMode.SCATTERED
                for mode in (
                    layer_scatter_modes.layer_input_mode,
                    layer_scatter_modes.attn_mode,
                    layer_scatter_modes.mlp_mode,
                    layer_scatter_modes.middle_residual_mode,
                    layer_scatter_modes.layer_output_mode,
                )
            ):
                raise NotImplementedError(
                    "GLM-5-Next mHC does not support scattered layer states"
                )
            self.layer_scatter_modes = layer_scatter_modes
            self.layer_communicator = MHCLayerCommunicator(
                layer_scatter_modes=layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=False,
                is_last_layer=(layer_idx == config.num_hidden_layers - 1),
                # DSA's absorb path fetches its Q/KV latent through the
                # attention-TP context after mHC has normalized the layer
                # input.  KDA consumes the normalized input directly and must
                # not install a stale DSA callback.
                qkv_latent_func=(
                    None
                    if config.is_kda_layer(layer_idx)
                    else self.self_attn.prepare_qkv_latent
                ),
                is_first_layer=(layer_idx == 0),
                hc_mult=hc_mult,
                hc_attn_pre=self.hc_attn_pre,
                hc_ffn_pre=self.hc_ffn_pre,
                hc_post=self.hc_post,
            )

    def _is_layer_sparse(self, layer_idx: int) -> bool:
        return (
            self.is_moe
            and self.config.num_experts is not None
            and layer_idx >= self.config.first_k_dense_replace
            and layer_idx % self.config.moe_layer_freq == 0
        )

    def hc_attn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        from sglang.kernels.ops.layernorm.mhc import hc_pre

        return hc_pre(
            x=hidden_states,
            hc_fn=self.hc_attn_fn,
            hc_scale=self.hc_attn_scale,
            hc_base=self.hc_attn_base,
            hc_mult=self.config.hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            post_mult_value=2.0,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
        )

    def hc_ffn_pre(self, hidden_states, out_norm_weight, out_norm_eps):
        from sglang.kernels.ops.layernorm.mhc import hc_pre

        return hc_pre(
            x=hidden_states,
            hc_fn=self.hc_ffn_fn,
            hc_scale=self.hc_ffn_scale,
            hc_base=self.hc_ffn_base,
            hc_mult=self.config.hc_mult,
            rms_eps=self.config.rms_norm_eps,
            hc_eps=self.config.hc_eps,
            sinkhorn_iters=self.config.hc_sinkhorn_iters,
            post_mult_value=2.0,
            hc_norm_weight=None,
            out_norm_weight=out_norm_weight,
            out_norm_eps=out_norm_eps,
        )

    def hc_post(self, hidden_states, residual, h_res, h_post):
        from sglang.kernels.ops.layernorm.mhc import hc_post

        if not self.mhc_enabled:
            raise RuntimeError("hc_post is only valid when config.mhc is True")
        return hc_post(
            x=hidden_states,
            residual=residual,
            h_post=h_post,
            h_res=h_res,
            hc_mult=self.config.hc_mult,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.mhc_enabled:
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
            )

        if residual is not None:
            raise RuntimeError(
                "GLM-5-Next mHC expects its cross-layer residual state to be None"
            )
        expected_width = self.hidden_size * (
            1 if self.layer_idx == 0 else self.config.hc_mult
        )
        if hidden_states.shape[-1] != expected_width:
            raise RuntimeError(
                "GLM-5-Next mHC layer input has width "
                f"{hidden_states.shape[-1]}, expected {expected_width}"
            )

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
        )
        if isinstance(self.mlp, Glm5NextMoE):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch=forward_batch,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return self.layer_communicator.postprocess_layer(
            hidden_states,
            residual,
            forward_batch,
        )


class Glm5NextModel(KimiLinearModel):
    """45-layer text model containing 34 KDA and 11 DSA construction seams."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        if config.mhc is not True:
            raise ValueError(
                "GLM-5-Next Session AB requires the checkpoint-native mhc=True "
                "execution path."
            )
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.layer_types = tuple(config.layer_types)

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Kimi currently assumes CUDA unconditionally.  Keeping the alternate
        # stream optional makes config/model construction tests CPU-safe and
        # matches the existing modules' Optional stream contract.
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Glm5NextDecoderLayer(
                config=config,
                layer_idx=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            if pp_proxy_tensors is None:
                raise RuntimeError("mHC pipeline rank requires pp_proxy_tensors")
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            if residual is not None:
                raise RuntimeError("mHC pipeline residual state must be None")

        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for layer_idx in range(self.start_layer, self.end_layer):
            ctx = get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            )
            with ctx:
                hidden_states, residual = self.layers[layer_idx](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if residual is not None:
            raise RuntimeError("final mHC residual state must be None")
        if hidden_states.shape[0] != 0:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Glm5NextForConditionalGeneration(nn.Module, DeepseekV2WeightLoaderMixin):
    """GLM-5-Next text runtime; the vision tower remains deferred to phase 7."""

    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkvbfg_a_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
        "fused_fg_b_proj": ["f_b_proj", "g_b_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Glm5NextConfig | Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mm_config = config if hasattr(config, "text_config") else None
        self.vision_config = getattr(config, "vision_config", None)
        self.visual = None

        text_config = getattr(config, "text_config", config)
        self.config = text_config
        self.quant_config = quant_config
        self.num_fused_shared_experts = 0
        self.pp_group = get_pp_group()
        self.model = Glm5NextModel(
            text_config,
            quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and text_config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config=text_config)
        self.skipped_phase7_visual_weights: tuple[str, ...] = ()
        self.skipped_session_ab_mtp_weights: tuple[str, ...] = ()
        self.skipped_pipeline_parallel_weight_count = 0
        self.checkpoint_runtime_default_parameters: tuple[str, ...] = ()
        self._checkpoint_expected_source_names: Optional[frozenset[str]] = None
        self._checkpoint_runtime_default_parameter_names: Optional[frozenset[str]] = (
            None
        )
        self._checkpoint_source_contract_complete = False

        # GLM's DSA layers use the same latent hand-off contract as the NSA
        # DeepSeek path.  NSA keeps scattered-input mode disabled, but the
        # context still owns the per-forward latent lifetime.
        get_attn_tp_context().init_context(text_config.q_lora_rank, True)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @staticmethod
    def _require_phase7_vision() -> None:
        raise RuntimeError(
            "GLM-5-Next vision inputs are intentionally unavailable in the "
            "phase-3 text runtime; enable them after the phase-7 vision integration."
        )

    def get_image_feature(self, *args, **kwargs):
        self._require_phase7_vision()

    def get_video_feature(self, *args, **kwargs):
        self._require_phase7_vision()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        contains_mm_inputs = getattr(forward_batch, "contains_mm_inputs", None)
        if callable(contains_mm_inputs) and contains_mm_inputs():
            self._require_phase7_vision()

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                pp_proxy_tensors,
            )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

    def _load_kda_stacked_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        """Load a KDA shard that maps onto a packed KT parameter."""

        mappings = (
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        )
        for packed_name, checkpoint_name, shard_id in mappings:
            if checkpoint_name not in name:
                continue
            candidate = name.replace(checkpoint_name, packed_name)
            param = params_dict.get(candidate)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is None:
                continue
            weight_loader(param, loaded_weight, shard_id)
            return True
        return False

    def _normalized_text_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        require_complete: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        params_dict = dict(self.named_parameters())
        expected_sources = getattr(self, "_checkpoint_expected_source_names", None)
        runtime_defaults = getattr(
            self,
            "_checkpoint_runtime_default_parameter_names",
            None,
        )
        if expected_sources is None or runtime_defaults is None:
            expected_sources, runtime_defaults = _glm5_next_checkpoint_source_contract(
                params_dict,
                num_experts=self.config.n_routed_experts,
                packed_modules_mapping=self.packed_modules_mapping,
            )
            self._checkpoint_expected_source_names = expected_sources
            self._checkpoint_runtime_default_parameter_names = runtime_defaults
        seen_sources: set[str] = set()
        seen_normalized_names: set[str] = set()
        skipped_visual: list[str] = []
        skipped_mtp: list[str] = []
        skipped_pp_count = 0

        num_hidden_layers = self.config.num_hidden_layers
        num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)

        for source_name, loaded_weight in weights:
            normalized_name = normalize_glm5_next_weight_name(source_name)
            if normalized_name is None:
                skipped_visual.append(source_name)
                continue

            if normalized_name in seen_normalized_names:
                raise RuntimeError(
                    "GLM-5-Next checkpoint source contract found a duplicate "
                    f"normalized text tensor {normalized_name!r}; latest raw "
                    f"name is {source_name!r}."
                )
            seen_normalized_names.add(normalized_name)

            if normalized_name in expected_sources:
                seen_sources.add(normalized_name)
            else:
                layer_match = _GLM5_NEXT_LAYER_WEIGHT_RE.match(normalized_name)
                layer_id = int(layer_match.group(1)) if layer_match else None

                # MTP is deliberately absent from the Session-AB runtime.  The
                # only legal decoder-layer exclusion is the one configured
                # appended layer (45 in the pinned checkpoint); a layer 46 or
                # broader ``>= num_hidden_layers`` skip must fail closed.
                if num_nextn_layers == 1 and layer_id == num_hidden_layers:
                    skipped_mtp.append(source_name)
                    continue

                # Every in-range layer is checked by its owning PP rank.  A
                # non-owner has a PPMissingLayer and must ignore that rank-local
                # source without weakening the owner rank's strict check.
                if (
                    layer_id is not None
                    and 0 <= layer_id < num_hidden_layers
                    and not self.model.start_layer <= layer_id < self.model.end_layer
                ):
                    skipped_pp_count += 1
                    continue

                owned_by_another_pp_rank = (
                    (
                        normalized_name.startswith("model.embed_tokens.")
                        and not self.pp_group.is_first_rank
                    )
                    or (
                        normalized_name.startswith("model.norm.")
                        and not self.pp_group.is_last_rank
                    )
                    or (
                        normalized_name.startswith("lm_head.")
                        and not self.pp_group.is_last_rank
                    )
                )
                if owned_by_another_pp_rank:
                    skipped_pp_count += 1
                    continue

                raise RuntimeError(
                    "GLM-5-Next checkpoint source contract rejected an unknown "
                    f"text tensor: raw={source_name!r}, "
                    f"normalized={normalized_name!r}. Only current-rank runtime "
                    "parameters, exact PP non-owner namespaces, the single "
                    "configured MTP layer, and raw visual./model.visual. "
                    "prefixes may be skipped."
                )

            if normalized_name.endswith(".A_log") and loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view(1, 1, -1, 1)

            if self._load_kda_stacked_weight(
                normalized_name, loaded_weight, params_dict
            ):
                continue
            yield normalized_name, loaded_weight

        self.skipped_phase7_visual_weights = tuple(skipped_visual)
        self.skipped_session_ab_mtp_weights = tuple(skipped_mtp)
        self.skipped_pipeline_parallel_weight_count = skipped_pp_count
        self.checkpoint_runtime_default_parameters = tuple(sorted(runtime_defaults))

        if require_complete:
            missing_sources = sorted(expected_sources - seen_sources)
            if missing_sources:
                examples = ", ".join(repr(name) for name in missing_sources[:16])
                raise RuntimeError(
                    "GLM-5-Next checkpoint source contract is missing "
                    f"{len(missing_sources)} required current-rank tensor(s); "
                    f"examples: {examples}"
                )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ) -> None:
        if is_nextn:
            raise RuntimeError(
                "GLM-5-Next MTP/NextN loading is outside the Session-AB boundary."
            )

        require_complete = not self._checkpoint_source_contract_complete
        DeepseekV2WeightLoaderMixin.do_load_weights(
            self,
            self._normalized_text_weights(
                weights,
                require_complete=require_complete,
            ),
            is_nextn=False,
        )
        self._checkpoint_source_contract_complete = True

    def post_load_weights(self, is_nextn: bool = False, weight_names=None) -> None:
        DeepseekV2WeightLoaderMixin.post_load_weights(
            self, is_nextn=is_nextn, weight_names=weight_names
        )


EntryClass = [Glm5NextForConditionalGeneration]
