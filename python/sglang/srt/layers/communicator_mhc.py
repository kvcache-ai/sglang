# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Minimal non-CP mHC layer communicator for GLM-5-Next.

This module deliberately supports the first bring-up topology only:
``attention DP == 1`` and ``attention CP == 1``.  It reuses the existing
communicator's TP decisions and replaces only the pre/post residual algebra.
DP/CP buffer sizing and hybrid-CP routing remain separate integration seams.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.communicator import (
    AttentionInputs,
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    get_attn_tp_context,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class MHCState:
    """Parameter-free runtime state shared by attention and FFN mHC stages."""

    hc_mult: int
    hc_attn_pre: Callable
    hc_ffn_pre: Callable
    hc_post: Callable
    h_res: Optional[torch.Tensor] = None
    h_post: Optional[torch.Tensor] = None

    @staticmethod
    def _resolve_out_norm(out_norm):
        if out_norm is None:
            return None, None
        return out_norm.weight.data, out_norm.variance_epsilon

    @staticmethod
    def _apply_unfused_norm(hidden_states, out_norm, norm_fused):
        if out_norm is not None and not norm_fused and hidden_states.shape[0] != 0:
            return out_norm(hidden_states)
        return hidden_states

    def attn_split(
        self,
        hidden_states: torch.Tensor,
        out_norm: Optional[torch.nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_attn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        hidden_states = self._apply_unfused_norm(hidden_states, out_norm, norm_fused)
        return hidden_states, residual

    def attn_to_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        out_norm: Optional[torch.nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.h_res is not None and self.h_post is not None
        hidden_states = self.hc_post(hidden_states, residual, self.h_res, self.h_post)
        residual = hidden_states
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_ffn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        hidden_states = self._apply_unfused_norm(hidden_states, out_norm, norm_fused)
        return hidden_states, residual

    def mlp_combine(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert self.h_res is not None and self.h_post is not None
        return self.hc_post(hidden_states, residual, self.h_res, self.h_post)

    def reset_aux(self) -> None:
        self.h_res = None
        self.h_post = None


class MHCCommunicateWithAllReduceAndLayerNormFn(CommunicateWithAllReduceAndLayerNormFn):
    """Non-CP replacements for attention-output communication."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode,
            residual_input_mode,
            hidden_states_output_mode,
            residual_output_mode,
            context,
        )
        replacements = {
            CommunicateWithAllReduceAndLayerNormFn._simple: (
                MHCCommunicateWithAllReduceAndLayerNormFn._simple
            ),
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual: (
                MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual
            ),
        }
        base_fn = fn.func if isinstance(fn, partial) else fn
        replacement = replacements.get(base_fn)
        if replacement is None:
            raise NotImplementedError(
                "minimal MHCLayerCommunicator does not support scattered/DP/CP "
                f"mode transition via {getattr(base_fn, '__name__', base_fn)}"
            )
        if isinstance(fn, partial):
            return partial(replacement, *fn.args, **(fn.keywords or {}))
        return replacement

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        del forward_batch, context
        return mhc.attn_to_mlp(hidden_states, residual, out_norm=layernorm)

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
        mhc: MHCState,
    ):
        del forward_batch
        assert context.attn_dp_size == 1 and context.attn_cp_size == 1
        if residual_input_mode == ScatterMode.SCATTERED:
            raise NotImplementedError(
                "minimal MHCLayerCommunicator requires an unscattered residual"
            )
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return mhc.attn_to_mlp(hidden_states, residual, out_norm=layernorm)


class MHCCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """mHC replacement for the ordinary no-scatter layer-output transition."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        fn = CommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode,
            residual_input_mode,
            output_mode,
            context,
        )
        if fn is not CommunicateSummableTensorPairFn._trivial:
            raise NotImplementedError(
                "minimal MHCLayerCommunicator does not support DP/CP/scattered "
                f"layer output via {getattr(fn, '__name__', fn)}"
            )
        return MHCCommunicateSummableTensorPairFn._trivial

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        del forward_batch, context, kwargs
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if is_last_layer:
            hidden_states = hc_contract(hidden_states, mhc.hc_mult)
        return hidden_states, None


class MHCLayerCommunicator(LayerCommunicator):
    """Explicitly selected mHC communicator for TP-only, non-CP execution."""

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        *,
        is_first_layer: bool,
        hc_mult: int,
        hc_attn_pre: Callable,
        hc_ffn_pre: Callable,
        hc_post: Callable,
    ):
        self.is_first_layer = is_first_layer
        self.mhc = MHCState(
            hc_mult=hc_mult,
            hc_attn_pre=hc_attn_pre,
            hc_ffn_pre=hc_ffn_pre,
            hc_post=hc_post,
        )
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        if self._context.attn_dp_size != 1 or self._context.attn_cp_size != 1:
            raise NotImplementedError(
                "minimal MHCLayerCommunicator requires attention DP=1 and CP=1"
            )
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            MHCCommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            MHCCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )

    def _assert_non_scattered(self) -> None:
        if get_attn_tp_context().input_scattered:
            raise NotImplementedError(
                "minimal MHCLayerCommunicator does not support scattered TP input"
            )

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        del residual, quant_format
        assert post_residual_addition is None
        self._assert_non_scattered()
        if self.is_first_layer:
            hidden_states = hc_expand(hidden_states, self.mhc.hc_mult)

        hidden_states, residual = self.mhc.attn_split(
            hidden_states, out_norm=self.input_layernorm
        )
        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        if self.qkv_latent_func is not None:
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(hidden_states, forward_batch, self.qkv_latent_func)
            )
        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        self._assert_non_scattered()
        if cache is not None:
            self._context.cache = cache
        return self._communicate_with_all_reduce_and_layer_norm_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            layernorm=self.post_attention_layernorm,
            context=self._context,
            mhc=self.mhc,
        )

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        self._assert_non_scattered()
        hidden_states, residual = self._communicate_summable_tensor_pair_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=False,
            mhc=self.mhc,
            is_last_layer=self.is_last_layer,
        )
        self.mhc.reset_aux()
        return hidden_states, residual

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        del forward_batch
        return False

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch) -> bool:
        del forward_batch
        return False
