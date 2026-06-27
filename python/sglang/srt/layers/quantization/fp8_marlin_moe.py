from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Fp8MarlinMoEMethod:
    """Standard FP8 MoE method for GPUs without native FP8 hardware (SM < 89).

    Uses the Marlin kernel for weight-only FP8 quantization on Ampere GPUs
    (SM 80-88).  Weights are created as ``float8_e4m3fn`` on CPU during
    init/loading (where that dtype is always legal), then repacked to
    Marlin int32 format in ``process_weights_after_loading`` **before**
    ``model.to(device)``, so the GPU never sees a native fp8 tensor.
    """

    _logged_once = False

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        self.runner = None
        if not Fp8MarlinMoEMethod._logged_once:
            Fp8MarlinMoEMethod._logged_once = True
            logger.warning(
                "Enabling FP8 Marlin MoE for Ampere GPU — using weight-only "
                "FP8 quantization with Marlin kernel."
            )

    # -- MoeRunner wiring ------------------------------------------------

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    # -- Weight creation -------------------------------------------------

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Delegate to the inner Fp8MoEMethod.

        Tensors are created on CPU at this point, so ``float8_e4m3fn`` is
        perfectly legal even on Ampere hosts — the Marlin repack runs
        before ``model.to(device)``.
        """
        self._fp8.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        # Fp8MoEMethod stores block_quant / weight_block_size on itself
        # but never on the layer.  prepare_moe_fp8_layer_for_marlin reads
        # these from the layer, so copy them over.
        if not hasattr(layer, "orig_dtype"):
            layer.orig_dtype = params_dtype
        if getattr(self._fp8, "block_quant", False):
            layer.weight_block_size = getattr(
                self._fp8, "weight_block_size",
                self._fp8.quant_config.weight_block_size,
            )

    # -- Post-load processing --------------------------------------------

    def process_weights_after_loading(self, layer: Module) -> None:
        """Run base FP8 post-processing, then repack for Marlin."""
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            prepare_moe_fp8_layer_for_marlin,
        )

        # Let the base Fp8MoEMethod handle ROCm normalization,
        # static-scale collapsing, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        # num_gpu_experts=0 means all experts run on CPU — weights are
        # empty (shape[0]==0) and the Marlin repack loop has no work.
        if layer.w13_weight.shape[0] == 0:
            return

        logger.debug(
            "Preparing FP8 MoE weights for Marlin backend (layer: %s, "
            "experts: %d)...",
            self.prefix,
            layer.num_experts,
        )

        # Fp8MoEMethod.create_weights produces (e, N, K) layout for weights
        # and (e, N_blocks, K_blocks) for block-quant scales.  But
        # prepare_moe_fp8_layer_for_marlin expects everything in (e, K, *)
        # layout.  Transpose weight dims 1↔2 AND scale dims 1↔2 so the
        # pack/repack/permute logic produces correct Marlin format.
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        for _sn in ("w13_weight_scale_inv", "w2_weight_scale_inv",
                     "w13_weight_scale", "w2_weight_scale"):
            if hasattr(layer, _sn):
                _s = getattr(layer, _sn)
                if _s.dim() >= 3:
                    _s.data = _s.data.transpose(1, 2).contiguous()

        # Repack float8_e4m3fn weights -> Marlin int32, permute scales,
        # and set ``layer.workspace``.
        prepare_moe_fp8_layer_for_marlin(layer, size_k_first=True)

    # -- Forward ---------------------------------------------------------

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(
                f"Unsupported topk output format: {topk_output.format}"
            )

        from sgl_kernel.scalar_type import scalar_types

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=8,
            is_k_full=True,
            b_q_type=scalar_types.float8_e4m3fn,
        )

        runner_output = self.runner.run(dispatch_output, quant_info=quant_info)

        return StandardCombineInput(hidden_states=runner_output.hidden_states)
