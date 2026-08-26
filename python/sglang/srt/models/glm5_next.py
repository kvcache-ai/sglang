"""GLM-5-Next text runtime with an isolated single-image vision path.

This module intentionally reuses the KT Kimi/DeepSeek building blocks.  The
GLM-specific KDA, DSA/KPool, mHC, and vision implementations are wired through
small seams so the integration does not modify the shared Kimi model or loosen
behavior for existing architectures.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Iterable, Iterator
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
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
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    is_allocation_symmetric,
)
from sglang.srt.layers.linear import ReplicatedLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
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
from sglang.srt.models.glm5_next_norm import Glm5NextRMSNorm
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.utils import make_layers
from sglang.srt.utils.common import BumpAllocator


logger = logging.getLogger(__name__)


# This historical constant name is retained for compatibility with Session AB
# loader tests.  Session D now consumes the same exact prefixes when vision is
# enabled and skips them only in the explicit text-only fallback.  A substring
# check such as ``"visual" in name`` could discard a future text parameter and
# hide a corrupt checkpoint.
GLM5_NEXT_PHASE7_VISUAL_WEIGHT_PREFIXES = ("visual.", "model.visual.")
GLM5_NEXT_PINNED_VISION_SOURCE_TENSOR_COUNT = 347
_GLM5_NEXT_DEAD_VISION_RUNTIME_PREFIXES = (
    "visual.embeddings.",
    "visual.post_conv_layernorm.",
)

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
    """Map the HF wrapper prefix to the SGLang text-module namespace.

    Visual sources are handled by the strict Session D loader before this
    helper when multimodal mode is active.  ``None`` therefore means either an
    already-classified visual source or a visual tensor skipped by the explicit
    text-only fallback.  No other unknown weight is classified as skippable.
    """

    if name.startswith(GLM5_NEXT_PHASE7_VISUAL_WEIGHT_PREFIXES):
        return None

    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name.startswith("language_model."):
        name = "model." + name.removeprefix("language_model.")

    return name


def _canonical_glm5_next_visual_source_name(name: str) -> Optional[str]:
    if name.startswith("model.visual."):
        return name
    if name.startswith("visual."):
        return "model." + name
    return None


def _glm5_next_vision_checkpoint_source_contract(
    parameter_names: Iterable[str],
) -> frozenset[str]:
    """Reverse live GLM vision parameters into checkpoint source names."""

    expected_sources: set[str] = set()
    for parameter_name in parameter_names:
        if not parameter_name.startswith("visual."):
            continue
        if parameter_name.startswith(_GLM5_NEXT_DEAD_VISION_RUNTIME_PREFIXES):
            # GlmOcrVisionModel inherits these GLM4V modules, but its forward
            # never reads them and the GLM5 checkpoint intentionally omits them.
            continue

        source_name = "model." + parameter_name
        if ".attn.qkv_proj." in source_name:
            expected_sources.add(
                source_name.replace(".attn.qkv_proj.", ".attn.qkv.")
            )
        elif ".gate_up_proj." in source_name:
            expected_sources.add(
                source_name.replace(".gate_up_proj.", ".gate_proj.")
            )
            expected_sources.add(
                source_name.replace(".gate_up_proj.", ".up_proj.")
            )
        else:
            expected_sources.add(source_name)
    return frozenset(expected_sources)


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


def _glm5_next_hc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    h_res: torch.Tensor,
    h_post: torch.Tensor,
    hc_mult: int,
) -> torch.Tensor:
    """Apply GLM's checkpoint-native BF16 hyper-connection update.

    The learned mix metadata is kept in FP32 by ``hc_pre``, but the released
    model casts it to the hidden-state dtype *before* the branch multiply and
    residual matmul.  Keep this arithmetic model-local: the shared mHC helper
    is also used by existing models whose historical FP32 accumulation must
    remain unchanged.
    """

    if hidden_states.ndim != 2 or residual.ndim != 2:
        raise ValueError("GLM-5-Next mHC post expects flat 2-D tensors")
    tokens, hidden_size = hidden_states.shape
    if residual.shape != (tokens, hc_mult * hidden_size):
        raise ValueError(
            "GLM-5-Next mHC residual has shape "
            f"{tuple(residual.shape)}, expected {(tokens, hc_mult * hidden_size)}"
        )
    residual_streams = residual.reshape(tokens, hc_mult, hidden_size)
    post = h_post.reshape(tokens, hc_mult).to(hidden_states.dtype)
    comb = h_res.reshape(tokens, hc_mult, hc_mult).to(hidden_states.dtype)
    output = post.unsqueeze(-1) * hidden_states.unsqueeze(1) + torch.matmul(
        comb.transpose(-1, -2), residual_streams
    )
    return output.reshape(tokens, hc_mult * hidden_size)


def _glm5_next_kda_can_use_fp32_o_proj(
    o_proj: nn.Module,
    hidden_states: torch.Tensor,
) -> bool:
    """Keep the higher-precision KDA partial behind a GLM-only contract."""

    weight = getattr(o_proj, "weight", None)
    return (
        isinstance(o_proj, RowParallelLinear)
        and isinstance(getattr(o_proj, "quant_method", None), UnquantizedLinearMethod)
        and hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and weight is not None
        and weight.is_cuda
        and weight.device == hidden_states.device
        and weight.dtype is torch.bfloat16
        and getattr(o_proj, "bias", None) is None
        and getattr(o_proj, "reduce_results", None) is False
        and getattr(o_proj, "input_is_parallel", None) is True
    )


def _glm5_next_kda_o_proj(
    o_proj: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one GLM KDA TP partial without an intermediate BF16 round."""

    if not hidden_states.is_cuda:
        return o_proj(hidden_states)[0]
    if not _glm5_next_kda_can_use_fp32_o_proj(o_proj, hidden_states):
        raise RuntimeError(
            "GLM-5-Next CUDA KDA o_proj requires an unquantized BF16 "
            "RowParallelLinear with no bias, input_is_parallel=True, and "
            "reduce_results=False"
        )

    # Match RowParallelLinear's allocator contract so CUDA Graph capture and
    # symmetric-allocation bookkeeping remain unchanged.  mHC performs the TP
    # all-reduce and casts its result once before applying hc_post.
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        return torch.mm(
            hidden_states,
            o_proj.weight.t(),
            out_dtype=torch.float32,
        )


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

        # The released checkpoint keeps b_proj in BF16 and evaluates its full
        # 64-output GEMM before slicing heads.  A ColumnParallelLinear instead
        # evaluates eight independent 8-output GEMMs at TP=8.  Although the
        # weights and input are identical, CUDA's BF16 GEMM choice changes at
        # that output geometry and the beta logits are observably different.
        # Keep this replacement GLM-local and tiny (64 * hidden_size BF16 per
        # rank), then slice the exact full result to the local attention heads.
        if not self.do_fuse_qkvbfg:
            self.b_proj = ReplicatedLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )
            if self.num_heads % self.attn_tp_size != 0:
                raise ValueError(
                    "GLM-5-Next beta heads must divide attention TP size: "
                    f"heads={self.num_heads}, attention_tp={self.attn_tp_size}"
                )
            self._beta_heads_per_rank = self.num_heads // self.attn_tp_size
            self._beta_head_start = (
                get_attention_tp_rank() * self._beta_heads_per_rank
            )
        else:
            self._beta_heads_per_rank = self.local_num_heads
            self._beta_head_start = 0

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
            beta = beta.narrow(
                -1,
                self._beta_head_start,
                self._beta_heads_per_rank,
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
        return _glm5_next_kda_o_proj(self.o_proj, core_attn_out)


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

        self.input_layernorm = Glm5NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Glm5NextRMSNorm(
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
                attn_all_reduce_output_dtype=(
                    torch.bfloat16 if config.is_kda_layer(layer_idx) else None
                ),
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
        if not self.mhc_enabled:
            raise RuntimeError("hc_post is only valid when config.mhc is True")
        return _glm5_next_hc_post(
            hidden_states=hidden_states,
            residual=residual,
            h_res=h_res,
            h_post=h_post,
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
        self.layer_types = tuple(
            getattr(
                config,
                "_glm5_next_checkpoint_layer_types",
                config.layer_types,
            )
        )

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
            self.norm = Glm5NextRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
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
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            hidden_states = (
                input_embeds
                if input_embeds is not None
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
    """GLM-5-Next runtime with isolated multi-image and single-video paths."""

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
        self.multimodal_enabled = bool(
            getattr(config, "_glm5_next_multimodal_active", False)
        )
        self.visual = None

        if self.multimodal_enabled:
            if self.mm_config is None or self.vision_config is None:
                raise ValueError(
                    "GLM-5-Next multimodal mode requires the root config and "
                    "its vision_config."
                )
            # Keep imports and the ~1 GiB allocation out of the explicit
            # programmatic text-only path.
            from sglang.srt.layers.attention import vision_utils
            from sglang.srt.models.glm_ocr import GlmOcrVisionModel

            vision_utils.update_vit_attn_dummy_heads_config(config)
            self.visual = GlmOcrVisionModel(
                self.vision_config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=False,
                num_dummy_heads=getattr(self.vision_config, "num_dummy_heads", 0),
                merger_context_dim=self.vision_config.projection_intermediate_size,
            )

        text_config = getattr(config, "text_config", config)
        self.config = text_config
        # The processor/scheduler may retain GLM4V-style MRoPE metadata, but
        # this pinned checkpoint has qk_rope_head_dim=0.  KT's DSA KPool uses
        # the 1-D scheduler positions for cache/index updates, so forward must
        # not replace them with the [3, N] metadata tensor used by the generic
        # upstream VLM path (which would also break CUDA-graph batch size 4).
        self.is_mrope_enabled = "mrope_section" in (
            getattr(self.config, "rope_scaling", None) or {}
        )
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
        self._checkpoint_expected_visual_source_names: Optional[
            frozenset[str]
        ] = None
        self._checkpoint_seen_visual_source_names: frozenset[str] = frozenset()
        self._checkpoint_visual_source_contract_complete = False
        if self.multimodal_enabled:
            visual_sources = _glm5_next_vision_checkpoint_source_contract(
                name for name, _ in self.named_parameters()
            )
            if len(visual_sources) != GLM5_NEXT_PINNED_VISION_SOURCE_TENSOR_COUNT:
                raise RuntimeError(
                    "GLM-5-Next pinned vision runtime must reverse-map to "
                    f"{GLM5_NEXT_PINNED_VISION_SOURCE_TENSOR_COUNT} checkpoint "
                    f"tensors; got {len(visual_sources)}."
                )
            self._checkpoint_expected_visual_source_names = visual_sources

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

    def _require_multimodal_enabled(self) -> None:
        if self.multimodal_enabled and self.visual is not None:
            return
        raise RuntimeError(
            "GLM-5-Next received vision input while multimodal support is disabled."
        )

    def pad_input_ids(self, input_ids, mm_inputs):
        self._require_multimodal_enabled()
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items) -> torch.Tensor:
        self._require_multimodal_enabled()
        if not items:
            raise ValueError("GLM-5-Next image feature extraction needs image items.")
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        # GLM4V/GLM-OCR build the spatial index with CPU ``torch.arange`` and
        # move only the resulting position ids into the vision tower.  Keep
        # the tiny grid descriptor on CPU; placing it on CUDA would mix device
        # scalars with those CPU allocations before the explicit transfer.
        image_grid_thw = torch.cat(
            [item.image_grid_thw for item in items], dim=0
        ).to(device="cpu", dtype=torch.int32)
        if pixel_values.ndim != 2 or image_grid_thw.ndim != 2:
            raise ValueError(
                "GLM-5-Next vision expects 2-D pixel patches and image_grid_thw."
            )
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        expected_rows = sum(
            int(grid.prod().item())
            // (self.vision_config.spatial_merge_size**2)
            for grid in image_grid_thw
        )
        if image_embeds.ndim != 2 or image_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "GLM-5-Next visual embedding rows do not match image placeholders: "
                f"expected {expected_rows}, got {tuple(image_embeds.shape)}."
            )
        return image_embeds

    def get_video_feature(self, items) -> torch.Tensor:
        """Run timestamped video frames through bounded vision microbatches."""

        self._require_multimodal_enabled()
        if not items:
            raise ValueError("GLM-5-Next video feature extraction needs video items.")
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        video_grids = torch.cat([item.video_grid_thw for item in items], dim=0).to(
            device="cpu", dtype=torch.int32
        )
        if pixel_values.ndim != 2 or video_grids.ndim != 2:
            raise ValueError(
                "GLM-5-Next video vision expects 2-D pixel patches and video_grid_thw."
            )

        frame_grids = []
        for grid in video_grids.tolist():
            grid_t, grid_h, grid_w = (int(value) for value in grid)
            if min(grid_t, grid_h, grid_w) <= 0:
                raise ValueError(f"Invalid GLM-5-Next video grid: {grid!r}.")
            frame_grids.extend([(1, grid_h, grid_w)] * grid_t)
        expected_patch_rows = sum(t * h * w for t, h, w in frame_grids)
        if pixel_values.shape[0] != expected_patch_rows:
            raise RuntimeError(
                "GLM-5-Next video patch/grid mismatch before vision: "
                f"expected={expected_patch_rows}, got={pixel_values.shape[0]}."
            )

        # Bound the temporary vision activation footprint.  A spatial frame
        # larger than the budget is kept intact and runs alone because the
        # vision tower cannot split one grid without changing positional ids.
        max_patch_rows = 32_768
        embeddings = []
        patch_cursor = 0
        batch_start = 0
        batch_rows = 0
        for frame_index, (_, grid_h, grid_w) in enumerate(frame_grids):
            frame_rows = grid_h * grid_w
            if batch_rows and batch_rows + frame_rows > max_patch_rows:
                batch_grid = torch.tensor(
                    frame_grids[batch_start:frame_index], dtype=torch.int32
                )
                embeddings.append(
                    self.visual(
                        pixel_values[patch_cursor - batch_rows : patch_cursor],
                        grid_thw=batch_grid,
                    )
                )
                batch_start = frame_index
                batch_rows = 0
            batch_rows += frame_rows
            patch_cursor += frame_rows
        if batch_rows:
            batch_grid = torch.tensor(frame_grids[batch_start:], dtype=torch.int32)
            embeddings.append(
                self.visual(
                    pixel_values[patch_cursor - batch_rows : patch_cursor],
                    grid_thw=batch_grid,
                )
            )

        video_embeds = torch.cat(embeddings, dim=0)
        merge_area = self.vision_config.spatial_merge_size**2
        expected_embed_rows = sum(t * h * w // merge_area for t, h, w in frame_grids)
        if video_embeds.ndim != 2 or video_embeds.shape[0] != expected_embed_rows:
            raise RuntimeError(
                "GLM-5-Next visual embedding rows do not match video placeholders: "
                f"expected={expected_embed_rows}, got={tuple(video_embeds.shape)}."
            )
        return video_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        contains_image_inputs = getattr(
            forward_batch, "contains_image_inputs", None
        )
        has_image_inputs = bool(
            callable(contains_image_inputs) and contains_image_inputs()
        )
        contains_video_inputs = getattr(
            forward_batch, "contains_video_inputs", None
        )
        has_video_inputs = bool(
            callable(contains_video_inputs) and contains_video_inputs()
        )
        has_multimodal_inputs = has_image_inputs or has_video_inputs
        forward_mode_name = getattr(
            getattr(forward_batch, "forward_mode", None), "name", None
        )
        is_multimodal_extend = has_multimodal_inputs and forward_mode_name == "EXTEND"
        # Real scheduler batches expose this Session-D marker as a dataclass
        # field.  Keep the historical lightweight/unit-test forward contract
        # usable for text-only calls, while refusing to silently lose the
        # marker for an actual image batch.
        try:
            forward_batch.glm5_next_has_image_inputs = (
                has_image_inputs and forward_mode_name == "EXTEND"
            )
            forward_batch.glm5_next_force_hybrid_prefill = bool(
                is_multimodal_extend
                or getattr(forward_batch, "glm5_next_force_hybrid_prefill", False)
            )
        except (AttributeError, TypeError):
            if is_multimodal_extend:
                raise RuntimeError(
                    "GLM-5-Next multimodal dispatch requires a mutable ForwardBatch "
                    "with the multimodal scheduling marker."
                )

        contains_audio_inputs = getattr(
            forward_batch, "contains_audio_inputs", None
        )
        if callable(contains_audio_inputs) and contains_audio_inputs():
            raise ValueError("GLM-5-Next does not support audio input.")

        if has_multimodal_inputs and not self.multimodal_enabled:
            self._require_multimodal_enabled()
        if has_multimodal_inputs and forward_mode_name not in (
            "EXTEND",
            "DECODE",
            "IDLE",
        ):
            raise RuntimeError(
                "GLM-5-Next accepts multimodal data only in plain EXTEND; "
                f"got ForwardMode.{forward_mode_name}."
            )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            if self.multimodal_enabled and is_multimodal_extend:
                from sglang.srt.managers.mm_utils import general_mm_embed_routine

                hidden_states = general_mm_embed_routine(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    language_model=self.model,
                    multimodal_model=self,
                    positions=positions,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            else:
                # Preserve the Session-C text/decode call shape exactly.  In
                # particular, pure text must not be forced through the generic
                # multimodal embedding routine merely because the vision tower
                # was enabled at startup; this also keeps its CUDA-graph and
                # layerwise-prefill behavior isolated from Session D.
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

    def _load_visual_weight(
        self,
        canonical_source_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> None:
        """Load one canonical visual source without buffering checkpoint data."""

        if loaded_weight.dtype is not torch.bfloat16:
            raise RuntimeError(
                "GLM-5-Next pinned visual tensors must be BF16; "
                f"{canonical_source_name!r} has dtype {loaded_weight.dtype}."
            )

        runtime_name = canonical_source_name.removeprefix("model.")
        shard_id = None
        if ".attn.qkv." in runtime_name:
            runtime_name = runtime_name.replace(
                ".attn.qkv.", ".attn.qkv_proj."
            )
        elif ".gate_proj." in runtime_name:
            runtime_name = runtime_name.replace(
                ".gate_proj.", ".gate_up_proj."
            )
            shard_id = 0
        elif ".up_proj." in runtime_name:
            runtime_name = runtime_name.replace(
                ".up_proj.", ".gate_up_proj."
            )
            shard_id = 1

        param = params_dict.get(runtime_name)
        if param is None or runtime_name.startswith(
            _GLM5_NEXT_DEAD_VISION_RUNTIME_PREFIXES
        ):
            raise RuntimeError(
                "GLM-5-Next visual checkpoint source mapped to an unknown or "
                f"dead runtime parameter: source={canonical_source_name!r}, "
                f"runtime={runtime_name!r}."
            )

        from sglang.srt.layers.attention import vision_utils
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        loaded_weight = vision_utils.pad_vit_attn_dummy_heads(
            self.mm_config, runtime_name, loaded_weight
        )
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            weight_loader(param, loaded_weight)
        else:
            weight_loader(param, loaded_weight, shard_id)

    def _normalized_text_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        require_complete: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        all_params_dict = dict(self.named_parameters())
        params_dict = {
            name: parameter
            for name, parameter in all_params_dict.items()
            if not name.startswith("visual.")
        }
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
        multimodal_enabled = bool(getattr(self, "multimodal_enabled", False))
        expected_visual_sources = getattr(
            self, "_checkpoint_expected_visual_source_names", None
        )
        if multimodal_enabled and expected_visual_sources is None:
            expected_visual_sources = _glm5_next_vision_checkpoint_source_contract(
                all_params_dict
            )
            self._checkpoint_expected_visual_source_names = expected_visual_sources
        # Duplicate detection is scoped to one load transaction, matching the
        # text-source contract above.  The first complete startup load remains
        # strict, while later online weight updates may legally present the
        # same visual names again (or update only a subset).
        seen_visual_sources: set[str] = set()

        num_hidden_layers = self.config.num_hidden_layers
        num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)

        for source_name, loaded_weight in weights:
            visual_source_name = _canonical_glm5_next_visual_source_name(source_name)
            if visual_source_name is not None:
                if not multimodal_enabled:
                    skipped_visual.append(source_name)
                    continue
                if visual_source_name in seen_visual_sources:
                    raise RuntimeError(
                        "GLM-5-Next visual checkpoint source contract found a "
                        f"duplicate tensor {visual_source_name!r}; latest raw "
                        f"name is {source_name!r}."
                    )
                if visual_source_name not in expected_visual_sources:
                    raise RuntimeError(
                        "GLM-5-Next visual checkpoint source contract rejected "
                        f"unknown tensor: raw={source_name!r}, "
                        f"canonical={visual_source_name!r}."
                    )
                self._load_visual_weight(
                    visual_source_name, loaded_weight, all_params_dict
                )
                seen_visual_sources.add(visual_source_name)
                continue

            normalized_name = normalize_glm5_next_weight_name(source_name)
            if normalized_name is None:
                # All exact visual prefixes were handled above.  Retain this
                # defensive branch for direct calls to the historical helper.
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
        self._checkpoint_seen_visual_source_names = frozenset(seen_visual_sources)

        if require_complete:
            missing_sources = sorted(expected_sources - seen_sources)
            if missing_sources:
                examples = ", ".join(repr(name) for name in missing_sources[:16])
                raise RuntimeError(
                    "GLM-5-Next checkpoint source contract is missing "
                    f"{len(missing_sources)} required current-rank tensor(s); "
                    f"examples: {examples}"
                )
            if multimodal_enabled:
                missing_visual_sources = sorted(
                    expected_visual_sources - seen_visual_sources
                )
                if missing_visual_sources:
                    examples = ", ".join(
                        repr(name) for name in missing_visual_sources[:16]
                    )
                    raise RuntimeError(
                        "GLM-5-Next visual checkpoint source contract is missing "
                        f"{len(missing_visual_sources)} required tensor(s); "
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
        self._checkpoint_visual_source_contract_complete = bool(
            self.multimodal_enabled
        )
        if require_complete and self.multimodal_enabled:
            logger.info(
                "GLM-5-Next visual checkpoint source contract complete: "
                "loaded=%d expected=%d dtype=BF16",
                len(self._checkpoint_seen_visual_source_names),
                len(self._checkpoint_expected_visual_source_names or ()),
            )

    def post_load_weights(self, is_nextn: bool = False, weight_names=None) -> None:
        DeepseekV2WeightLoaderMixin.post_load_weights(
            self, is_nextn=is_nextn, weight_names=weight_names
        )


EntryClass = [Glm5NextForConditionalGeneration]
