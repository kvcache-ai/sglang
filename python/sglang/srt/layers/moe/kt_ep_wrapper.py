# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import copy
import ctypes
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.utils import get_compiler_backend, is_cuda

if is_cuda():
    from sgl_kernel import gptq_marlin_repack

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


# Global flag to track if attention LoRA has been loaded
_ATTN_LORA_LOADED = False

# Global reference to the model for attention LoRA loading
_MODEL_REF = None


def set_model_reference(model: torch.nn.Module) -> None:
    """Set global model reference for attention LoRA loading."""
    global _MODEL_REF
    _MODEL_REF = model


def get_model_reference() -> Optional[torch.nn.Module]:
    """Get global model reference."""
    return _MODEL_REF


class LoRALinear(torch.nn.Module):
    """A simple LoRA wrapper for Linear layers used in attention/MLP.

    This applies LoRA as: output = base(x) + (x @ lora_a.T @ lora_b.T) * scaling

    Handles both cases:
    - Base layers that return a single tensor
    - SGLang linear layers that return (output, output_bias) tuples
    """

    def __init__(self, base: torch.nn.Module, lora_a: torch.Tensor, lora_b: torch.Tensor, lora_alpha: float):
        super().__init__()
        self.base = base

        # Get device from base module's parameters
        device = next(base.parameters()).device

        self.lora_a = torch.nn.Parameter(lora_a.to(device=device, dtype=torch.bfloat16), requires_grad=False)
        self.lora_b = torch.nn.Parameter(lora_b.to(device=device, dtype=torch.bfloat16), requires_grad=False)
        self.scaling = lora_alpha / float(self.lora_a.shape[0])

    def forward(self, x: torch.Tensor, **kwargs):
        base_out = self.base(x, **kwargs)

        # LoRA computation in higher precision for accuracy
        x_lora = x.to(self.lora_a.dtype)
        lora_out = (x_lora @ self.lora_a.t()) @ self.lora_b.t()

        # Handle both tensor and tuple (output, output_bias) returns from SGLang layers
        if isinstance(base_out, tuple):
            output, output_bias = base_out
            output = output + lora_out.to(output.dtype) * self.scaling
            return output, output_bias
        else:
            return base_out + lora_out.to(base_out.dtype) * self.scaling


def load_attention_lora_from_converted_file(model: torch.nn.Module, lora_path: str, lora_alpha: float = 16.0) -> int:
    """Load attention (MLA) and layer 0 MLP LoRA weights from converted .pt file.

    This function should be called once after model initialization to load
    attention LoRA weights that are stored in the converted MoE LoRA file.

    DeepSeek-V2 MLA (Multi-Head Latent Attention) projections:
        - q_proj: Query projection (ColumnParallelLinear)
        - kv_a_proj_with_mqa: KV latent projection (ReplicatedLinear)
        - kv_b_proj: KV up projection (ColumnParallelLinear)
        - o_proj: Output projection (RowParallelLinear)

    Args:
        model: The base model to apply LoRA to
        lora_path: Path to the converted .pt file (same as --kt-moe-lora-path)
        lora_alpha: LoRA alpha scaling factor

    Returns:
        Number of LoRA layers applied

    Raises:
        FileNotFoundError: If the LoRA file does not exist
        RuntimeError: If any expected LoRA weights fail to load
    """
    global _ATTN_LORA_LOADED
    if _ATTN_LORA_LOADED:
        return 0

    if not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LoRA file not found: {lora_path}. "
            "Please run scripts/convert_moe_lora.py to convert the adapter first."
        )

    lora_weights = torch.load(lora_path, map_location="cpu", weights_only=True)

    # Validate metadata
    if "metadata" not in lora_weights:
        raise KeyError(
            f"Metadata not found in LoRA file: {lora_path}. "
            "The file may be corrupted or in wrong format."
        )

    metadata = lora_weights["metadata"]
    lora_rank = metadata.get("lora_rank")
    logger.info(f"[Attention LoRA] Loading from {lora_path}, lora_rank={lora_rank}, lora_alpha={lora_alpha}")

    loaded_count = 0
    failed_loads = []

    # MLA projection names (DeepSeek-V2 specific)
    ATTN_PROJ_NAMES = ["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    MLP0_PROJ_NAMES = ["gate", "up", "down"]

    # Load attention LoRA
    attn_layers = sorted([k for k in lora_weights.keys() if k.startswith("attn_layer_")])
    logger.info(f"[Attention LoRA] Found {len(attn_layers)} attention layers in file")

    for attn_key in attn_layers:
        layer_idx = int(attn_key.split("_")[-1])
        layer_data = lora_weights[attn_key]

        if layer_idx >= len(model.model.layers):
            failed_loads.append(
                f"attn_layer_{layer_idx}: layer index out of range "
                f"(model has {len(model.model.layers)} layers)"
            )
            continue

        layer = model.model.layers[layer_idx]

        # Check which projections are available in the file
        available_projs = set()
        for proj_name in ATTN_PROJ_NAMES:
            a_key = f"{proj_name}_lora_a"
            b_key = f"{proj_name}_lora_b"
            if a_key in layer_data and b_key in layer_data:
                available_projs.add(proj_name)

        if not available_projs:
            failed_loads.append(f"attn_layer_{layer_idx}: no valid projection LoRA found")
            continue

        # Load each available projection
        for proj_name in ATTN_PROJ_NAMES:
            a_key = f"{proj_name}_lora_a"
            b_key = f"{proj_name}_lora_b"

            if a_key not in layer_data or b_key not in layer_data:
                # Skip if not in file (not all projections may have LoRA)
                continue

            lora_a = layer_data[a_key]
            lora_b = layer_data[b_key]

            # Validate tensor values
            if torch.isnan(lora_a).any() or torch.isnan(lora_b).any():
                failed_loads.append(f"layer_{layer_idx}.self_attn.{proj_name}: contains NaN values")
                continue
            if torch.isinf(lora_a).any() or torch.isinf(lora_b).any():
                failed_loads.append(f"layer_{layer_idx}.self_attn.{proj_name}: contains Inf values")
                continue

            # Get base layer
            base_layer = getattr(layer.self_attn, proj_name, None)
            if base_layer is None:
                failed_loads.append(f"layer_{layer_idx}.self_attn.{proj_name}: base layer not found in model")
                continue

            if isinstance(base_layer, LoRALinear):
                logger.warning(f"layer_{layer_idx}.self_attn.{proj_name}: already wrapped with LoRA, skipping")
                continue

            try:
                wrapped = LoRALinear(base_layer, lora_a, lora_b, lora_alpha)
                setattr(layer.self_attn, proj_name, wrapped)
                loaded_count += 1
                logger.debug(
                    f"[Attention LoRA] Loaded layer_{layer_idx}.self_attn.{proj_name}: "
                    f"lora_a={lora_a.shape}, lora_b={lora_b.shape}"
                )
            except Exception as e:
                failed_loads.append(f"layer_{layer_idx}.self_attn.{proj_name}: {e}")

    # Load layer 0 MLP LoRA (non-MoE layer)
    if "mlp0" in lora_weights:
        mlp0_data = lora_weights["mlp0"]
        layer0_mlp = model.model.layers[0].mlp
        logger.info("[MLP0 LoRA] Loading layer 0 MLP LoRA")

        # Check if model uses fused gate_up_proj (MergedColumnParallelLinear)
        has_fused_gate_up = hasattr(layer0_mlp, "gate_up_proj") and not hasattr(layer0_mlp, "gate_proj")

        if has_fused_gate_up:
            # Handle fused gate_up_proj case
            logger.info("[MLP0 LoRA] Model uses fused gate_up_proj, applying merged LoRA")

            # Check if we have gate and up LoRA in file
            has_gate = "gate_lora_a" in mlp0_data and "gate_lora_b" in mlp0_data
            has_up = "up_lora_a" in mlp0_data and "up_lora_b" in mlp0_data
            has_down = "down_lora_a" in mlp0_data and "down_lora_b" in mlp0_data

            if has_gate and has_up:
                gate_lora_a = mlp0_data["gate_lora_a"]
                gate_lora_b = mlp0_data["gate_lora_b"]
                up_lora_a = mlp0_data["up_lora_a"]
                up_lora_b = mlp0_data["up_lora_b"]

                # Validate tensors
                for name, tensor in [("gate_lora_a", gate_lora_a), ("gate_lora_b", gate_lora_b),
                                     ("up_lora_a", up_lora_a), ("up_lora_b", up_lora_b)]:
                    if torch.isnan(tensor).any():
                        failed_loads.append(f"layer_0.mlp.gate_up_proj ({name}): contains NaN values")
                    if torch.isinf(tensor).any():
                        failed_loads.append(f"layer_0.mlp.gate_up_proj ({name}): contains Inf values")

                if not failed_loads:
                    base_layer = layer0_mlp.gate_up_proj
                    if isinstance(base_layer, LoRALinear):
                        logger.warning("layer_0.mlp.gate_up_proj: already wrapped with LoRA, skipping")
                    else:
                        try:
                            # For MergedColumnParallelLinear, we need to concatenate gate and up LoRA
                            # lora_a: [rank, hidden] -> same for both, just use gate
                            # lora_b: [intermediate*2, rank] -> concat gate_b and up_b
                            merged_lora_a = gate_lora_a  # Both should be the same shape
                            merged_lora_b = torch.cat([gate_lora_b, up_lora_b], dim=0)

                            wrapped = LoRALinear(base_layer, merged_lora_a, merged_lora_b, lora_alpha)
                            layer0_mlp.gate_up_proj = wrapped
                            loaded_count += 1
                            logger.debug(
                                f"[MLP0 LoRA] Loaded layer_0.mlp.gate_up_proj (merged): "
                                f"lora_a={merged_lora_a.shape}, lora_b={merged_lora_b.shape}"
                            )
                        except Exception as e:
                            failed_loads.append(f"layer_0.mlp.gate_up_proj: {e}")
            else:
                if not has_gate:
                    failed_loads.append("layer_0.mlp.gate_up_proj: missing gate LoRA in file")
                if not has_up:
                    failed_loads.append("layer_0.mlp.gate_up_proj: missing up LoRA in file")

            # Handle down_proj separately
            if has_down:
                down_lora_a = mlp0_data["down_lora_a"]
                down_lora_b = mlp0_data["down_lora_b"]

                if torch.isnan(down_lora_a).any() or torch.isnan(down_lora_b).any():
                    failed_loads.append("layer_0.mlp.down_proj: contains NaN values")
                elif torch.isinf(down_lora_a).any() or torch.isinf(down_lora_b).any():
                    failed_loads.append("layer_0.mlp.down_proj: contains Inf values")
                else:
                    base_layer = getattr(layer0_mlp, "down_proj", None)
                    if base_layer is None:
                        failed_loads.append("layer_0.mlp.down_proj: base layer not found in model")
                    elif isinstance(base_layer, LoRALinear):
                        logger.warning("layer_0.mlp.down_proj: already wrapped with LoRA, skipping")
                    else:
                        try:
                            wrapped = LoRALinear(base_layer, down_lora_a, down_lora_b, lora_alpha)
                            layer0_mlp.down_proj = wrapped
                            loaded_count += 1
                            logger.debug(
                                f"[MLP0 LoRA] Loaded layer_0.mlp.down_proj: "
                                f"lora_a={down_lora_a.shape}, lora_b={down_lora_b.shape}"
                            )
                        except Exception as e:
                            failed_loads.append(f"layer_0.mlp.down_proj: {e}")
        else:
            # Handle separate gate_proj, up_proj, down_proj case
            for proj_name in MLP0_PROJ_NAMES:
                a_key = f"{proj_name}_lora_a"
                b_key = f"{proj_name}_lora_b"

                if a_key not in mlp0_data or b_key not in mlp0_data:
                    failed_loads.append(f"layer_0.mlp.{proj_name}_proj: missing lora_a or lora_b in file")
                    continue

                lora_a = mlp0_data[a_key]
                lora_b = mlp0_data[b_key]

                # Validate tensor values
                if torch.isnan(lora_a).any() or torch.isnan(lora_b).any():
                    failed_loads.append(f"layer_0.mlp.{proj_name}_proj: contains NaN values")
                    continue
                if torch.isinf(lora_a).any() or torch.isinf(lora_b).any():
                    failed_loads.append(f"layer_0.mlp.{proj_name}_proj: contains Inf values")
                    continue

                proj_attr = f"{proj_name}_proj"
                base_layer = getattr(layer0_mlp, proj_attr, None)
                if base_layer is None:
                    failed_loads.append(f"layer_0.mlp.{proj_attr}: base layer not found in model")
                    continue

                if isinstance(base_layer, LoRALinear):
                    logger.warning(f"layer_0.mlp.{proj_attr}: already wrapped with LoRA, skipping")
                    continue

                try:
                    wrapped = LoRALinear(base_layer, lora_a, lora_b, lora_alpha)
                    setattr(layer0_mlp, proj_attr, wrapped)
                    loaded_count += 1
                    logger.debug(
                        f"[MLP0 LoRA] Loaded layer_0.mlp.{proj_attr}: "
                        f"lora_a={lora_a.shape}, lora_b={lora_b.shape}"
                    )
                except Exception as e:
                    failed_loads.append(f"layer_0.mlp.{proj_attr}: {e}")

    # Check for failures - any failure is fatal
    if failed_loads:
        error_msg = (
            f"Failed to load LoRA weights from {lora_path}:\n"
            + "\n".join(f"  - {f}" for f in failed_loads)
            + "\n\nThis indicates a mismatch between the LoRA file and the model structure."
        )
        raise RuntimeError(error_msg)

    # Verify we loaded something if file has attention/mlp0 data
    has_attn_data = len(attn_layers) > 0
    has_mlp0_data = "mlp0" in lora_weights

    if loaded_count == 0:
        if has_attn_data or has_mlp0_data:
            raise RuntimeError(
                f"No LoRA layers were loaded from {lora_path}, "
                f"but file contains {len(attn_layers)} attention layers and "
                f"{'MLP0 data' if has_mlp0_data else 'no MLP0 data'}. "
                "This may indicate a model/LoRA structure mismatch."
            )
        else:
            logger.info("[Attention LoRA] No attention/MLP0 LoRA data in file, skipping")
            _ATTN_LORA_LOADED = True
            return 0

    logger.info(
        f"[Attention LoRA] Successfully loaded {loaded_count} layers: "
        f"{len(attn_layers)} attention layers, "
        f"{'3 MLP0 projections' if has_mlp0_data else 'no MLP0'}"
    )
    _ATTN_LORA_LOADED = True

    return loaded_count


# Thread-local storage for forward_batch (to avoid breaking API changes)
_thread_local = threading.local()


def set_current_forward_batch(forward_batch: Optional["ForwardBatch"]) -> None:
    """Set the current forward_batch for this thread (used for LoRA switching)."""
    _thread_local.forward_batch = forward_batch


def get_current_forward_batch() -> Optional["ForwardBatch"]:
    """Get the current forward_batch for this thread."""
    return getattr(_thread_local, "forward_batch", None)


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        gpu_prefill_token_threshold: token threshold for enabling full GPU fallback
        moe_lora_enabled: Whether MoE expert LoRA is enabled
        moe_lora_path: Path to converted MoE LoRA weights (.pt file)
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        sft_method: SFT quantization method (e.g., "AMXBF16_SFT")
        model_path: Path to HuggingFace model (for SFT BF16 mode)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None
    gpu_prefill_token_threshold: Optional[int] = None
    # MoE LoRA configuration
    moe_lora_enabled: bool = False
    moe_lora_path: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: float = 32.0
    sft_method: str = "AMXBF16_SFT"
    model_path: Optional[str] = None


_SHARED_FULL_CONTEXT = None


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        init_args: tuple,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
    ):
        self._build_layers(layer, init_args, global_num_experts, moe_runner_config)

        # Capture original tensors to support restoration before loading
        self.original_params = {
            name: param for name, param in self.gpu_layer.named_parameters()
        }
        self.original_buffers = {
            name: buf for name, buf in self.gpu_layer.named_buffers()
        }

        # Create CPU buffers once for weight loading (shared across layers)
        self._create_cpu_buffers()

    def _build_layers(self, layer, init_args, global_num_experts, moe_runner_config):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        hidden_size, intermediate_size_per_partition, params_dtype = init_args
        target_device = next(layer.parameters()).device

        # Create gpu_layer as a shallow copy, then override specific attributes
        self.gpu_layer = copy.copy(layer)
        # Clear module state that shouldn't be shared
        self.gpu_layer._parameters = {}
        self.gpu_layer._buffers = {}
        self.gpu_layer._modules = {}

        # Override expert counts for full GPU execution
        self.gpu_layer.num_experts = global_num_experts
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.num_gpu_experts = global_num_experts

        # Create quant_method for gpu_layer
        if self.gpu_layer.quant_config is not None:
            self.gpu_method = self.gpu_layer.quant_config.get_quant_method(
                self.gpu_layer, prefix=""
            )
        else:
            self.gpu_method = UnquantizedFusedMoEMethod(
                self.gpu_layer.use_triton_kernels
            )
        self.gpu_layer.quant_method = self.gpu_method

        # Detect quantization type for weight loading
        self.is_fp8_quant = self._detect_fp8_quant()
        self.is_fp8_channel_quant = self._detect_fp8_channel_quant()
        self.is_bf16_quant = self._detect_bf16_quant()

        self.gpu_method.create_weights(
            layer=self.gpu_layer,
            num_experts=global_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
        )

        # Move all parameters to target device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)

        # Create runner config - update both num_experts and num_local_experts for full GPU fallback
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
        )
        self.gpu_layer.moe_runner_config = runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, runner_config)

    def _detect_fp8_quant(self) -> bool:
        """Detect if the quantization method is FP8 block quant.

        Returns:
            True if FP8 block quant, False otherwise (INT4 Marlin, BF16, etc.)
        """
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

        method = self.gpu_method
        # Check for Fp8MoEMethod with block_quant
        if isinstance(method, Fp8MoEMethod) and getattr(method, "block_quant", False):
            return True

        # Check for CompressedTensorsW8A8Fp8MoEMethod with block_quant
        method_name = method.__class__.__name__
        if "W8A8Fp8" in method_name and getattr(method, "block_quant", False):
            return True

        return False

    def _detect_fp8_channel_quant(self) -> bool:
        """Detect if the quantization method is FP8 per-channel quant.

        Per-channel FP8 differs from block FP8:
        - Per-channel: scale shape is (num_experts, output_dim, 1), weight_scale name
        - Block FP8: scale shape is (num_experts, blocks_n, blocks_k), weight_scale_inv name

        Returns:
            True if FP8 per-channel quant, False otherwise
        """
        try:
            from compressed_tensors.quantization import QuantizationStrategy
        except ImportError:
            return False

        method = self.gpu_method
        method_name = method.__class__.__name__

        # Check for CompressedTensorsW8A8Fp8MoEMethod with channel strategy
        if "W8A8Fp8" in method_name:
            weight_quant = getattr(method, "weight_quant", None)
            if weight_quant is not None:
                if weight_quant.strategy == QuantizationStrategy.CHANNEL:
                    return True

        return False

    def _detect_bf16_quant(self) -> bool:
        """Detect if the quantization method is BF16/unquantized.

        Returns:
            True if BF16/unquantized, False otherwise (INT4 Marlin, FP8, etc.)
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            UnquantizedFusedMoEMethod,
        )

        method = self.gpu_method
        # Check for UnquantizedFusedMoEMethod
        if isinstance(method, UnquantizedFusedMoEMethod):
            return True

        return False

    @property
    def weight_names(self) -> list:
        """Get weight names based on quantization type."""
        if self.is_fp8_quant:
            return self.WEIGHT_NAMES_FP8
        elif self.is_fp8_channel_quant:
            return self.WEIGHT_NAMES_FP8_CHANNEL
        elif self.is_bf16_quant:
            return self.WEIGHT_NAMES_BF16
        else:
            return self.WEIGHT_NAMES_INT4

    # Weight names for shared memory buffers (INT4 Marlin format)
    WEIGHT_NAMES_INT4 = [
        "w13_weight_packed",
        "w13_weight_scale",
        "w2_weight_packed",
        "w2_weight_scale",
    ]

    # Weight names for FP8 block quant format
    WEIGHT_NAMES_FP8 = [
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    ]

    # Weight names for FP8 per-channel quant format
    # Per-channel differs from block quant:
    # - Scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
    # - Weight name: w13_weight_scale vs w13_weight_scale_inv
    WEIGHT_NAMES_FP8_CHANNEL = [
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
    ]

    # Weight names for BF16/unquantized format (no scales)
    WEIGHT_NAMES_BF16 = [
        "w13_weight",
        "w2_weight",
    ]

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses double buffering (2 experts) to reduce memory usage while maintaining
        pipeline efficiency: write(e+1) || copy(e) only needs 2 buffers.
        """
        # Set NUMA local allocation policy to allocate on local NUMA node
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            raise RuntimeError("NUMA is not available on this system")
        libnuma.numa_set_localalloc()

        self.cpu_buffers = {}
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}
        tp_rank = get_tensor_model_parallel_rank()
        num_experts = self.gpu_layer.num_experts

        # Generate unique ID on rank 0 and broadcast to all ranks
        if tp_rank == 0:
            self.shm_unique_id = uuid.uuid4().hex[:8]
        else:
            self.shm_unique_id = None
        if dist.is_initialized():
            unique_id_list = [self.shm_unique_id]
            dist.broadcast_object_list(
                unique_id_list, src=0, group=get_tp_group().cpu_group
            )
            self.shm_unique_id = unique_id_list[0]

        for name in self.weight_names:
            gpu_tensor = getattr(self.gpu_layer, name)
            # Only allocate 2 experts worth of buffer (double buffering)
            expert_shape = gpu_tensor.shape[1:]  # Shape per expert
            expert_nbytes = (
                gpu_tensor.numel() // num_experts * gpu_tensor.element_size()
            )
            double_buf_nbytes = expert_nbytes * 2

            shm_name = f"kt_buf_{name}_r{tp_rank}_{self.shm_unique_id}"
            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=double_buf_nbytes
            )
            self.shm_handles[name] = shm

            # Shape: [2, ...expert_shape...]
            cpu_buffer = torch.frombuffer(shm.buf, dtype=gpu_tensor.dtype).reshape(
                (2,) + expert_shape
            )

            # Register as pinned memory for fast DMA
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostRegister(
                    cpu_buffer.data_ptr(), double_buf_nbytes, 0
                )

            self.cpu_buffers[name] = cpu_buffer

        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()

        # Unlink shared memory after all ranks have collected pointers.
        # The memory remains accessible as long as we hold references via mmap.
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)
        for shm in self.shm_handles.values():
            shm.unlink()

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    ptr = self.cpu_buffers[name].data_ptr()
                elif tp_rank == 0:
                    shm_name = f"kt_buf_{name}_r{rank}_{self.shm_unique_id}"
                    try:
                        shm = shared_memory.SharedMemory(name=shm_name)
                        self._opened_shm_refs[f"{name}_r{rank}"] = shm
                        ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    except FileNotFoundError:
                        logger.error(
                            "Rank %d: Failed to open shared memory '%s'",
                            tp_rank,
                            shm_name,
                        )
                        ptr = 0
                else:
                    ptr = 0
                all_rank_ptrs[name].append(ptr)

        return all_rank_ptrs

    def _prepare_weight_int4(self, wrapper):
        """Prepare INT4 Marlin weights by writing from KT, copying to GPU, and postprocessing.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        Postprocessing extracted from CompressedTensorsWNA16MoEMethod.process_weights_after_loading
        in python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        method = self.gpu_method

        num_bits = method.num_bits
        packed_factor = method.packed_factor
        group_size = method.group_size
        actorder = getattr(method, "actorder", None)
        num_experts = layer.num_experts
        device = layer.w13_weight_packed.device

        # Create empty g_idx tensors for non-grouped actorder
        if actorder != "group":
            for name in [
                "w13_weight_g_idx",
                "w2_weight_g_idx",
                "w13_g_idx_sort_indices",
                "w2_g_idx_sort_indices",
            ]:
                setattr(
                    layer,
                    name,
                    torch.nn.Parameter(
                        torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                        requires_grad=False,
                    ),
                )

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_INT4:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            # Reshape gpu_t to match expert shape for per-expert copy
            expert_shape = cpu_buf.shape[1:]
            gpu_t.set_(gpu_t.view((num_experts,) + expert_shape))
            weight_infos.append((cpu_buf, gpu_t))

        w13_p, w13_s = layer.w13_weight_packed, layer.w13_weight_scale
        w2_p, w2_s = layer.w2_weight_packed, layer.w2_weight_scale
        w13_k, w13_n = w13_p.shape[1] * packed_factor, w13_p.shape[2]
        w2_k, w2_n = w2_p.shape[1] * packed_factor, w2_p.shape[2]
        w2_sk = w2_s.shape[1] * (group_size if group_size != -1 else packed_factor)
        perm = torch.empty(0, dtype=torch.int32, device=device)

        # Tmp buffers for transpose
        tmp_bufs = [
            torch.empty(t.size(1), t.size(2), dtype=t.dtype, device=device)
            for _, t in weight_infos
        ]

        def postprocess_expert(e):
            # Transpose
            for (_, gpu_t), tmp in zip(weight_infos, tmp_bufs):
                d1, d2 = gpu_t.size(1), gpu_t.size(2)
                tmp.copy_(gpu_t[e].reshape(d2, d1).T, non_blocking=True)
                gpu_t[e].copy_(tmp, non_blocking=True)
            # Repack weights
            w13_p[e].copy_(
                gptq_marlin_repack(w13_p[e], perm, w13_k, w13_n, num_bits).view(
                    w13_p[e].shape
                )
            )
            w2_p[e].copy_(
                gptq_marlin_repack(w2_p[e], perm, w2_k, w2_n, num_bits).view(
                    w2_p[e].shape
                )
            )
            # Permute scales
            w13_s[e].copy_(
                marlin_permute_scales(w13_s[e], w13_n, w13_s.shape[2], group_size).view(
                    w13_s[e].shape
                )
            )
            w2_s[e].copy_(
                marlin_permute_scales(w2_s[e], w2_sk, w2_s.shape[2], group_size).view(
                    w2_s[e].shape
                )
            )

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_packed_buf = self.cpu_buffers["w13_weight_packed"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_packed_buf = self.cpu_buffers["w2_weight_packed"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_packed_expert_nbytes = (
                w13_packed_buf.numel() // 2 * w13_packed_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_packed_expert_nbytes = (
                w2_packed_buf.numel() // 2 * w2_packed_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_packed_ptrs = [
                    ptr + slot * w13_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_packed"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_packed_ptrs = [
                    ptr + slot * w2_packed_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_packed"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_packed_ptrs,
                    w13_scale_ptrs,
                    w2_packed_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

        # Reshape to final shape
        w13_p.set_(w13_p.view(num_experts, w13_k // 16, w13_n * (num_bits // 2)))
        w2_p.set_(w2_p.view(num_experts, w2_k // 16, w2_n * (num_bits // 2)))

    def _prepare_weight_fp8(self, wrapper):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 block quant is simpler than INT4 Marlin:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optional DeepGemm ue8m0 conversion is handled after all experts are loaded.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((cpu_buf, gpu_t))

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        def postprocess_expert(e):
            # FP8 doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, ensure copy from that slot is complete.
                    # Since (e+1)%2 == (e-1)%2 for e >= 1, we need to wait for copy(e-1).
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            # Postprocess expert e-1: provides pipeline structure for future extensions
            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        # Process last expert
        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_fp8_channel(self, wrapper):
        """Prepare FP8 per-channel quant weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        FP8 per-channel quant differs from FP8 block quant:
        - Per-channel scale shape: (num_experts, output_dim, 1) vs (num_experts, blocks_n, blocks_k)
        - Weight name: w13_weight_scale vs w13_weight_scale_inv
        - Both use float8_e4m3fn weights

        Similar to block FP8:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)

        The postprocess stage is a no-op for FP8 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_FP8_CHANNEL:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((cpu_buf, gpu_t))

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        def postprocess_expert(e):
            # FP8 per-channel doesn't need actual postprocessing (no repack/permute).
            # This function provides a pipeline synchronization point and
            # can be extended for future FP8-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w13_scale_buf = self.cpu_buffers["w13_weight_scale"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]
            w2_scale_buf = self.cpu_buffers["w2_weight_scale"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w13_scale_expert_nbytes = (
                w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )
            w2_scale_expert_nbytes = (
                w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w13_scale_ptrs = [
                    ptr + slot * w13_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight_scale"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                w2_scale_ptrs = [
                    ptr + slot * w2_scale_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight_scale"]
                ]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, ensure copy from that slot is complete.
                    # Since (e+1)%2 == (e-1)%2 for e >= 1, we need to wait for copy(e-1).
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            # Postprocess expert e-1: provides pipeline structure for future extensions
            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        # Process last expert
        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.
        """
        # Bind Python thread to specific CPU core (last cores for each rank)
        tp_rank = get_tensor_model_parallel_rank()
        num_cpus = os.cpu_count()
        target_cpu = num_cpus - 1 - tp_rank
        os.sched_setaffinity(0, {target_cpu})

        layer = self.gpu_layer
        num_experts = layer.num_experts
        device = layer.w13_weight.device

        # Prepare weight tensors (cpu_buf is double-buffered with shape [2, ...])
        weight_infos = []
        for name in self.WEIGHT_NAMES_BF16:
            cpu_buf = self.cpu_buffers[name]  # Shape: [2, ...expert_shape...]
            gpu_t = getattr(layer, name)  # Shape: [num_experts, ...expert_shape...]
            weight_infos.append((cpu_buf, gpu_t))

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        events = [torch.cuda.Event() for _ in range(num_experts)]

        def postprocess_expert(e):
            # BF16 doesn't need actual postprocessing (no repack/permute/transpose).
            # This function provides a pipeline synchronization point and
            # can be extended for future BF16-specific processing if needed.
            pass

        # Prepare write pipeline (rank 0 only)
        tp_world_size = get_tensor_model_parallel_world_size()
        do_write = tp_rank == 0 and wrapper is not None

        if do_write:
            # Calculate per-expert byte sizes (buffer is double-buffered: [2, ...])
            w13_weight_buf = self.cpu_buffers["w13_weight"]
            w2_weight_buf = self.cpu_buffers["w2_weight"]

            # Buffer shape is [2, ...], so numel() // 2 gives per-expert size
            w13_weight_expert_nbytes = (
                w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
            )
            w2_weight_expert_nbytes = (
                w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
            )

            def submit_write_expert(expert_id):
                # Use expert_id % 2 for double buffering slot selection
                slot = expert_id % 2
                w13_weight_ptrs = [
                    ptr + slot * w13_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w13_weight"]
                ]
                w2_weight_ptrs = [
                    ptr + slot * w2_weight_expert_nbytes
                    for ptr in self.all_rank_buffer_ptrs["w2_weight"]
                ]
                # For BF16, we pass empty scale pointer lists (no scales)
                w13_scale_ptrs = [0] * tp_world_size
                w2_scale_ptrs = [0] * tp_world_size
                wrapper.submit_write_weight_scale_to_buffer(
                    tp_world_size,
                    expert_id,
                    w13_weight_ptrs,
                    w13_scale_ptrs,
                    w2_weight_ptrs,
                    w2_scale_ptrs,
                )

            # Submit expert 0 ahead of time
            submit_write_expert(0)

        for e in range(num_experts):
            # Sync write for expert e, submit write for expert e+1
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if e + 1 < num_experts:
                    # Before writing to slot (e+1)%2, ensure copy from that slot is complete.
                    # Since (e+1)%2 == (e-1)%2 for e >= 1, we need to wait for copy(e-1).
                    if e > 0:
                        events[e - 1].synchronize()
                    submit_write_expert(e + 1)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                slot = e % 2  # Double buffering
                for cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[e].record(copy_stream)

            # Postprocess expert e-1: provides pipeline structure for future extensions
            if e > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[e - 1])
                    postprocess_expert(e - 1)

        # Process last expert
        with torch.cuda.stream(post_stream):
            post_stream.wait_event(events[-1])
            postprocess_expert(num_experts - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def load(self, layer_idx, wrapper):
        """Load weights from disk to GPU via shared memory."""
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        t0 = time.perf_counter()

        # Select appropriate prepare_weight method based on quantization type
        if self.is_fp8_quant:
            self._prepare_weight_fp8(wrapper)
        elif self.is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper)
        elif self.is_bf16_quant:
            self._prepare_weight_bf16(wrapper)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT fallback: layer %d prepare weight = %.2f ms",
                layer_idx,
                total_time,
            )


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Get num_layers from model config
    hf_config = server_args.get_hf_config()
    num_layers = getattr(hf_config, "num_hidden_layers", None)

    # Check for MoE LoRA configuration
    moe_lora_enabled = getattr(server_args, "kt_moe_lora_path", None) is not None
    moe_lora_path = getattr(server_args, "kt_moe_lora_path", None)
    lora_rank = getattr(server_args, "kt_moe_lora_rank", 16)
    lora_alpha = getattr(server_args, "kt_moe_lora_alpha", 32.0)
    sft_method = getattr(server_args, "kt_moe_sft_method", "AMXBF16_SFT")

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_prefill_token_threshold=server_args.kt_gpu_prefill_token_threshold,
        # MoE LoRA configuration
        moe_lora_enabled=moe_lora_enabled,
        moe_lora_path=moe_lora_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        sft_method=sft_method,
        model_path=server_args.model_path,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        New tensor with CPU expert IDs masked as -1 (original tensor is not modified)
    """
    # Clone to avoid in-place modification of the original tensor
    # This is critical for SFT mode where the original topk_ids is needed later
    result = topk_ids.clone()
    result[result >= num_gpu_experts] = -1
    return result


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoEMethod(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()

        self.gpu_prefill_token_threshold = kt_config.gpu_prefill_token_threshold or 0
        self._full_init_args = None
        self.wrapper: Optional[KTMoEWrapper] = None

        # Track current loaded LoRA for dynamic switching
        self.current_lora_id: Optional[str] = None
        self.lora_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # Cache loaded LoRA weights

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            if self.kt_config.moe_lora_enabled:
                # SFT mode with MoE LoRA support
                # Determine weight path based on SFT method:
                # - AMXBF16_SFT: use model_path (HuggingFace BF16 weights)
                # - AMXINT8_SFT/AMXINT4_SFT: use weight_path (pre-quantized weights)
                if self.kt_config.sft_method == "AMXBF16_SFT":
                    sft_weight_path = self.kt_config.model_path
                else:
                    sft_weight_path = self.kt_config.weight_path

                self.wrapper = KTMoEWrapper(
                    layer_idx=self.kt_config.layer_idx,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    hidden_size=hidden_size,
                    moe_intermediate_size=intermediate_size_full,
                    num_gpu_experts=self.num_gpu_experts,
                    cpuinfer_threads=self.kt_config.cpuinfer_threads,
                    threadpool_count=self.kt_config.threadpool_count,
                    weight_path=sft_weight_path,
                    chunked_prefill_size=self.kt_config.chunked_prefill_size,
                    mode="sft",
                    method=self.kt_config.sft_method,
                    lora_rank=self.kt_config.lora_rank,
                    lora_alpha=self.kt_config.lora_alpha,
                    max_cache_depth=1,  # Inference mode: no need for deep cache
                )
            else:
                # Standard inference mode
                self.wrapper = KTMoEWrapper(
                    layer_idx=self.kt_config.layer_idx,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    hidden_size=hidden_size,
                    moe_intermediate_size=intermediate_size_full,
                    num_gpu_experts=self.num_gpu_experts,
                    cpuinfer_threads=self.kt_config.cpuinfer_threads,
                    threadpool_count=self.kt_config.threadpool_count,
                    weight_path=self.kt_config.weight_path,
                    chunked_prefill_size=self.kt_config.chunked_prefill_size,
                    method=self.kt_config.method,
                    max_deferred_experts_per_token=layer_max_deferred,
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

            # 3. Load MoE LoRA weights if enabled
            if self.kt_config.moe_lora_enabled and self.kt_config.moe_lora_path:
                # Load default LoRA from kt_config.moe_lora_path
                # Use "default" as lora_id for initial load
                self._load_moe_lora_weights(lora_path=self.kt_config.moe_lora_path, lora_id="default")

                # 4. Load attention and layer 0 MLP LoRA (only once, when processing first MoE layer)
                model = get_model_reference()
                if model is not None:
                    load_attention_lora_from_converted_file(
                        model,
                        self.kt_config.moe_lora_path,
                        lora_alpha=self.kt_config.lora_alpha,
                    )

    def _load_moe_lora_weights(self, lora_path: Optional[str] = None, lora_id: Optional[str] = None) -> None:
        """Load MoE LoRA weights from converted .pt file.

        This method loads the pre-converted MoE LoRA weights and initializes
        them in the kt-kernel SFT wrapper.

        Args:
            lora_path: Path to LoRA weights file. If None, uses self.kt_config.moe_lora_path
            lora_id: LoRA identifier for tracking currently loaded LoRA

        Raises:
            FileNotFoundError: If the LoRA file does not exist
            KeyError: If required weights are missing from the file
            RuntimeError: If weight shapes are invalid
        """
        import os

        if lora_path is None:
            lora_path = self.kt_config.moe_lora_path
        layer_idx = self.kt_config.layer_idx

        logger.info(f"[MoE LoRA] Loading layer {layer_idx} from {lora_path}, lora_id={lora_id}")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"MoE LoRA file not found: {lora_path}. "
                "Please run scripts/convert_moe_lora.py to convert the adapter first."
            )

        # Load the converted LoRA weights
        lora_weights = torch.load(lora_path, map_location="cpu", weights_only=True)

        # Check metadata exists
        if "metadata" not in lora_weights:
            raise KeyError(
                f"Metadata not found in MoE LoRA file: {lora_path}. "
                "The file may be corrupted or in wrong format."
            )

        metadata = lora_weights["metadata"]

        # Get layer-specific weights
        layer_key = f"layer_{layer_idx}"
        if layer_key not in lora_weights:
            available_layers = sorted([k for k in lora_weights.keys() if k.startswith("layer_")])
            raise KeyError(
                f"Layer {layer_idx} not found in MoE LoRA file: {lora_path}. "
                f"Available MoE layers: {available_layers}"
            )

        layer_data = lora_weights[layer_key]

        # Verify all required keys exist
        required_keys = [
            "gate_lora_a", "gate_lora_b",
            "up_lora_a", "up_lora_b",
            "down_lora_a", "down_lora_b",
        ]
        missing_keys = [k for k in required_keys if k not in layer_data]
        if missing_keys:
            raise KeyError(
                f"Missing LoRA weights for layer {layer_idx}: {missing_keys}. "
                f"Available keys: {list(layer_data.keys())}"
            )

        # Extract LoRA weight tensors (ensure contiguous and correct dtype)
        gate_lora_a = layer_data["gate_lora_a"].contiguous().to(torch.bfloat16)
        gate_lora_b = layer_data["gate_lora_b"].contiguous().to(torch.bfloat16)
        up_lora_a = layer_data["up_lora_a"].contiguous().to(torch.bfloat16)
        up_lora_b = layer_data["up_lora_b"].contiguous().to(torch.bfloat16)
        down_lora_a = layer_data["down_lora_a"].contiguous().to(torch.bfloat16)
        down_lora_b = layer_data["down_lora_b"].contiguous().to(torch.bfloat16)

        # Verify shapes are consistent
        num_experts = gate_lora_a.shape[0]
        lora_rank = gate_lora_a.shape[1]

        expected_num_experts = metadata.get("num_experts")
        expected_lora_rank = metadata.get("lora_rank")

        if expected_num_experts and num_experts != expected_num_experts:
            raise RuntimeError(
                f"Layer {layer_idx}: num_experts mismatch. "
                f"Got {num_experts} but metadata says {expected_num_experts}"
            )

        if expected_lora_rank and lora_rank != expected_lora_rank:
            raise RuntimeError(
                f"Layer {layer_idx}: lora_rank mismatch. "
                f"Got {lora_rank} but metadata says {expected_lora_rank}"
            )

        # Verify all tensors have the same number of experts
        all_tensors = [
            ("gate_lora_a", gate_lora_a), ("gate_lora_b", gate_lora_b),
            ("up_lora_a", up_lora_a), ("up_lora_b", up_lora_b),
            ("down_lora_a", down_lora_a), ("down_lora_b", down_lora_b),
        ]
        for name, tensor in all_tensors:
            if tensor.shape[0] != num_experts:
                raise RuntimeError(
                    f"Layer {layer_idx}: {name} has {tensor.shape[0]} experts, "
                    f"but expected {num_experts}"
                )

        # Check for NaN/Inf values
        for name, tensor in all_tensors:
            if torch.isnan(tensor).any():
                raise RuntimeError(f"Layer {layer_idx}: {name} contains NaN values")
            if torch.isinf(tensor).any():
                raise RuntimeError(f"Layer {layer_idx}: {name} contains Inf values")

        # Initialize LoRA weights in the kt-kernel wrapper
        self.wrapper.init_lora_weights(
            gate_lora_a,
            gate_lora_b,
            up_lora_a,
            up_lora_b,
            down_lora_a,
            down_lora_b,
        )

        logger.info(
            f"[MoE LoRA] Layer {layer_idx} loaded successfully: "
            f"num_experts={num_experts}, lora_rank={lora_rank}, "
            f"lora_alpha={metadata.get('lora_alpha')}, lora_id={lora_id}"
        )

        # Update current loaded LoRA ID
        if lora_id is not None:
            self.current_lora_id = lora_id

    def _get_lora_path_by_id(self, lora_id: str) -> str:
        """Get LoRA file path from lora_id using LoRARegistry.

        NOTE: This method is currently unused. We only support the default MoE LoRA for now.
        TODO: Implement dynamic multi-LoRA switching by querying the LoRA registry.

        Args:
            lora_id: LoRA identifier (UUID from SGLang's LoRA registry)

        Returns:
            Path to converted MoE LoRA .pt file

        Raises:
            ValueError: If lora_id is not found in registry or path is invalid
        """
        # This would require access to the LoRA registry to map lora_id (UUID) to lora_path
        # However, the LoRA registry is in the tokenizer manager process, which may not be
        # easily accessible from the worker process.
        # For now, we only support the default MoE LoRA specified at server startup.
        raise NotImplementedError(
            "Dynamic multi-LoRA switching is not yet supported for KT MoE. "
            "Currently only the default MoE LoRA (--kt-moe-lora-path) can be used."
        )

    def _clear_lora_weights(self) -> None:
        """Clear LoRA weights and return to base model.

        This is done by setting all LoRA weights to zero.
        """
        import torch
        print("triggered _clear_lora_weights")
        layer_idx = self.kt_config.layer_idx

        # Create zero LoRA weights with correct shapes
        # [num_experts, lora_rank, hidden_size/intermediate_size]
        num_experts = self.global_num_experts - self.num_gpu_experts
        lora_rank = self.kt_config.lora_rank
        hidden_size, intermediate_size_per_partition, _ = self._full_init_args
        intermediate_size_full = intermediate_size_per_partition * get_tensor_model_parallel_world_size()

        zero_gate_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        zero_gate_lora_b = torch.zeros(num_experts, intermediate_size_full, lora_rank, dtype=torch.bfloat16)
        zero_up_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        zero_up_lora_b = torch.zeros(num_experts, intermediate_size_full, lora_rank, dtype=torch.bfloat16)
        zero_down_lora_a = torch.zeros(num_experts, lora_rank, intermediate_size_full, dtype=torch.bfloat16)
        zero_down_lora_b = torch.zeros(num_experts, hidden_size, lora_rank, dtype=torch.bfloat16)

        self.wrapper.init_lora_weights(
            zero_gate_lora_a,
            zero_gate_lora_b,
            zero_up_lora_a,
            zero_up_lora_b,
            zero_down_lora_a,
            zero_down_lora_b,
        )

        if layer_idx == 0:
            logger.info("MoE LoRA cleared - using base model")

        self.current_lora_id = None

    def _update_lora_if_needed(self, forward_batch: "ForwardBatch") -> None:
        """Check if LoRA needs to be switched and update if necessary.

        IMPORTANT: Currently only supports switching between base model and the default MoE LoRA
        specified at server startup. Dynamic multi-LoRA switching is not yet supported.

        Args:
            forward_batch: Batch information containing lora_ids
        """
        layer_idx = self.kt_config.layer_idx

        # DEBUG: Track calls to this method
        if layer_idx == 1:
            print(f"[DEBUG _update_lora_if_needed ENTRY] layer={layer_idx} forward_batch={forward_batch is not None}")

        if not self.kt_config.moe_lora_enabled:
            if layer_idx == 1:
                print(f"[DEBUG _update_lora_if_needed] MoE LoRA not enabled, skipping")
            return

        # Get unique LoRA IDs in current batch
        if layer_idx == 1:
            print(f"[DEBUG _update_lora_if_needed] forward_batch.lora_ids: {forward_batch.lora_ids}")

        lora_ids = set(forward_batch.lora_ids)

        # Remove None from set
        lora_ids.discard(None)

        # Determine target state: "default" (use default MoE LoRA) or None (base model)
        if len(lora_ids) == 0:
            # No LoRA requested - use base model
            target_lora_id = None
        else:
            # Any LoRA requested - use default MoE LoRA (for now)
            # TODO: Support dynamic multi-LoRA switching by mapping lora_id to MoE LoRA paths
            target_lora_id = "default"
            if len(lora_ids) > 1 and layer_idx == 1:
                logger.warning(
                    f"KT MoE LoRA does not support multiple LoRAs per batch yet. "
                    f"Requested: {lora_ids}. Using default MoE LoRA."
                )

        # Switch if needed
        if target_lora_id != self.current_lora_id:
            if layer_idx == 1:
                logger.info(f"Switching MoE LoRA: {self.current_lora_id} -> {target_lora_id}")

            if target_lora_id is None:
                # Clear LoRA
                self._clear_lora_weights()
            else:
                # Load default MoE LoRA (specified at server startup)
                default_lora_path = self.kt_config.moe_lora_path
                if default_lora_path is None:
                    raise ValueError(
                        "MoE LoRA enabled but no default MoE LoRA path specified. "
                        "Please provide --kt-moe-lora-path at server startup."
                    )
                self._load_moe_lora_weights(lora_path=default_lora_path, lora_id=target_lora_id)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config

        # Create a separate config for GPU method without routed_scaling_factor.
        # This is because:
        # 1. GPU method's moe_sum_reduce would apply routed_scaling_factor internally
        # 2. KT CPU kernel does NOT apply routed_scaling_factor
        # 3. The combined output (GPU + CPU) would have inconsistent scaling
        # 4. routed_scaling_factor is applied uniformly in deepseek_v2.py forward_normal
        # So we disable it in GPU method to avoid double scaling on GPU part.
        gpu_runner_config = replace(moe_runner_config, routed_scaling_factor=None)
        if self.override_num_local_experts:
            gpu_runner_config = replace(
                gpu_runner_config, num_local_experts=self.num_gpu_experts
            )

        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, gpu_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Note: In SFT mode (MoE LoRA enabled), this method is a no-op because
        forward_sft() is synchronous and will be called in apply() directly.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        # SFT mode uses synchronous forward_sft, skip async submit
        if self.kt_config.moe_lora_enabled:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(
        self,
        x: torch.Tensor,
        dispatch_output: "StandardDispatchOutput" = None,
    ) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information
            dispatch_output: Dispatched tokens and routing information (required for SFT mode)

        Returns:
            CPU expert computation results
        """

        # DEBUG: Track sync() calls
        layer_idx = self.kt_config.layer_idx
        if layer_idx == 1:
            print(f"[DEBUG sync ENTRY] layer={layer_idx} tp_rank={self.tp_rank} wrapper={self.wrapper is not None} dispatch_output={dispatch_output is not None}")

        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # DEBUG: Log LoRA enabled status (only once per layer)
        layer_idx = self.kt_config.layer_idx
        if not hasattr(self, '_logged_lora_status'):
            print(f"[DEBUG Layer {layer_idx}] moe_lora_enabled: {self.kt_config.moe_lora_enabled}")
            print(f"[DEBUG Layer {layer_idx}] current_lora_id: {self.current_lora_id}")
            self._logged_lora_status = True

        # SFT mode: use synchronous forward_sft with LoRA
        if self.kt_config.moe_lora_enabled:
            if dispatch_output is None:
                raise ValueError(
                    "dispatch_output is required for SFT mode (MoE LoRA enabled)"
                )

            # Get forward_batch from thread-local storage (set by model forward())
            forward_batch = get_current_forward_batch()

            # DEBUG: Check forward_batch availability
            layer_idx = self.kt_config.layer_idx
            if layer_idx == 1:
                print(f"[DEBUG sync] forward_batch from thread-local: {forward_batch is not None}")
                if forward_batch is not None:
                    print(f"[DEBUG sync] forward_batch.lora_ids: {forward_batch.lora_ids}")

            # Check and switch LoRA if needed based on batch lora_ids
            if forward_batch is not None:
                self._update_lora_if_needed(forward_batch)
            elif layer_idx == 1:
                print(f"[DEBUG sync] WARNING: No forward_batch available in thread-local storage")

            topk_output = dispatch_output.topk_output
            topk_weights, topk_ids, _ = topk_output

            # forward_sft is synchronous and includes LoRA computation
            result = self.wrapper.forward_sft(
                x,
                topk_ids,
                topk_weights,
                save_for_backward=False,  # Inference mode: no gradient needed
            )

            return result

        # Standard inference mode: wait for async CPU computation
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # DEBUG: Track apply() calls
        layer_idx = self.kt_config.layer_idx
        if layer_idx == 1:
            print(f"[DEBUG apply ENTRY] layer={layer_idx} dispatch_output={dispatch_output is not None}")
            if dispatch_output is not None:
                print(f"[DEBUG apply] dispatch_output type: {type(dispatch_output)}")
                print(f"[DEBUG apply] has forward_batch attr: {hasattr(dispatch_output, 'forward_batch')}")

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0

        # SFT mode (MoE LoRA): save a copy of hidden_states before GPU computation
        # because GPU method may modify hidden_states in-place (used as output buffer)
        if self.kt_config.moe_lora_enabled:
            x_for_cpu = x.clone()
        else:
            x_for_cpu = x

        # Check for full GPU fallback
        if (
            self.gpu_prefill_token_threshold > 0
            and num_tokens >= self.gpu_prefill_token_threshold
        ):
            ctx = self._build_full_context(layer)

            t_compute = time.perf_counter()
            result = ctx.gpu_method.apply(ctx.gpu_layer, dispatch_output)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compute_time = (time.perf_counter() - t_compute) * 1000.0
            if self.tp_rank == 0:
                logger.info(
                    "KT fallback: layer %d compute = %.2f ms",
                    self.kt_config.layer_idx,
                    compute_time,
                )
            return result

        # Step 1: Submit CPU expert computation (non-blocking)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)

        # Step 4: Synchronize CPU results and merge with GPU results
        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:
            # Pass dispatch_output for SFT mode (needed for forward_sft)
            # Use x_for_cpu which is a clone in SFT mode (original x may be modified by GPU)
            cpu_output = self.sync(x_for_cpu, dispatch_output)
            # SFT mode returns CPU tensor, need to move to GPU before adding
            if cpu_output.device != output.device:
                cpu_output = cpu_output.to(output.device, non_blocking=True)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)

    def _build_full_context(self, layer: torch.nn.Module) -> "SharedFullContext":
        global _SHARED_FULL_CONTEXT

        if _SHARED_FULL_CONTEXT is None:
            _SHARED_FULL_CONTEXT = SharedFullContext(
                layer=layer,
                init_args=self._full_init_args,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )

        _SHARED_FULL_CONTEXT.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
        )
        return _SHARED_FULL_CONTEXT