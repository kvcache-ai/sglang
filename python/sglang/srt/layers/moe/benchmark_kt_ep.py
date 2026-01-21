#!/usr/bin/env python3
"""
Benchmark script for KTEPWrapperMethod.apply() performance testing.

This script measures the execution time of hybrid CPU-GPU MoE computation
under various workload configurations.

Usage:
    python benchmark_kt_ep.py \
        --model /path/to/model \
        --kt-weight-path /path/to/kt_weights \
        --kt-num-gpu-experts 2 \
        --num-tokens 128 \
        --gpu-slots 100 \
        --gpu-experts-active 2 \
        --cpu-experts-active 4
"""

import argparse
import glob
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from transformers import AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_minimal_server_args(model_path: str):
    """Initialize minimal server args for MoE computation to work."""
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    # Create a minimal ServerArgs with required model_path
    server_args = ServerArgs(model_path=model_path)
    set_global_server_args_for_scheduler(server_args)
    logger.info("Global server args initialized")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KTEPWrapperMethod.apply()")

    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory (HuggingFace format)")

    # KT configuration
    parser.add_argument("--kt-weight-path", type=str, required=True,
                        help="Path to KT CPU quantized weights")
    parser.add_argument("--kt-cpuinfer", type=int, default=60,
                        help="Number of CPU inference threads")
    parser.add_argument("--kt-threadpool-count", type=int, default=2,
                        help="Number of thread pools for CPU computation")
    parser.add_argument("--kt-method", type=str, default="AMXINT4",
                        choices=["AMXINT4", "AMXINT8", "RAWINT4", "FP8", "BF16", "FP8_PERCHANNEL", "LLAMAFILE", "MOE_INT4", "MOE_INT8"],
                        help="CPU computation method (must match kt-weight-path quantization)")
    parser.add_argument("--kt-num-gpu-experts", type=int, required=True,
                        help="Number of experts on GPU")
    parser.add_argument("--kt-chunked-prefill-size", type=int, default=512,
                        help="Chunk size for prefill computation")

    # Workload configuration
    parser.add_argument("--num-tokens", type=int, required=True,
                        help="Total number of input tokens")
    parser.add_argument("--gpu-slots", type=int, required=True,
                        help="Number of slots routed to GPU experts")
    parser.add_argument("--gpu-experts-active", type=int, required=True,
                        help="Number of GPU experts with load")
    parser.add_argument("--cpu-experts-active", type=int, required=True,
                        help="Number of CPU experts with load")

    # Benchmark configuration
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Enable CUDA graph mode")
    parser.add_argument("--warmup-iters", type=int, default=10,
                        help="Number of warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=100,
                        help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-weight-loading", action="store_true",
                        help="Skip loading weights (use for quick framework testing)")
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Starting layer index to benchmark (for weight loading)")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of layers to load and alternate between (reduces caching effects)")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Use GPU-only apply (no CPU stream, no CPU operations)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Use CPU-only apply (no GPU computation, saves GPU memory)")
    parser.add_argument("--throughput-mode", action="store_true",
                        help="Measure throughput instead of per-iteration latency (no sync between iterations)")
    parser.add_argument("--override-top-k", type=int, default=None,
                        help="Override model's top_k to control total slots (total_slots = num_tokens * override_top_k)")
    parser.add_argument("--zero-cpu-slots", action="store_true",
                        help="Set cpu_slots=0, all slots go to GPU (use with --override-top-k to control GPU slot count)")
    parser.add_argument("--zero-gpu-slots", action="store_true",
                        help="Set gpu_slots=0, all slots go to CPU (use with --override-top-k to control CPU slot count)")

    return parser.parse_args()


# =============================================================================
# Model Configuration Loading
# =============================================================================

@dataclass
class MoEModelConfig:
    """MoE model configuration extracted from HuggingFace config."""
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_layers: int
    params_dtype: torch.dtype
    first_moe_layer: int = 0  # First layer index that has MoE experts


def detect_first_moe_layer(model_path: str, max_layers_to_check: int = 10) -> int:
    """Detect the first layer that has MoE experts by scanning safetensors files.

    Some models (like DeepSeek) have dense layers before MoE layers.
    This function finds the first layer with experts.

    Args:
        model_path: Path to model directory
        max_layers_to_check: Maximum number of layers to scan

    Returns:
        Index of the first MoE layer (0 if all layers have MoE or detection fails)
    """
    import re

    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensors_files:
        return 0

    # Check each layer for expert keys
    for layer_idx in range(max_layers_to_check):
        expert_pattern = f"model.layers.{layer_idx}.mlp.experts."

        for sf_file in safetensors_files:
            try:
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if expert_pattern in key:
                            logger.info(f"Detected first MoE layer at index {layer_idx}")
                            return layer_idx
            except Exception:
                continue

    # Default to 0 if no experts found (might be a different naming convention)
    return 0


def load_model_config(model_path: str) -> MoEModelConfig:
    """Load MoE configuration from model."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Extract hidden_size
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Model config missing 'hidden_size'")

    # Extract intermediate_size for MoE experts
    # For MoE models, moe_intermediate_size is the per-expert FFN size
    # intermediate_size is typically for shared experts or non-MoE FFN
    moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
    if moe_intermediate_size is not None:
        intermediate_size = moe_intermediate_size
    else:
        intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        raise ValueError("Model config missing 'intermediate_size' or 'moe_intermediate_size'")

    # Extract num_experts (try multiple attribute names)
    num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "n_routed_experts", None)
    if num_experts is None:
        raise ValueError("Model config missing num_experts")

    # Extract top_k
    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", None)
    if top_k is None:
        top_k = 2  # Default to 2

    # Extract num_layers
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Model config missing 'num_hidden_layers'")

    # Determine dtype
    torch_dtype = getattr(config, "torch_dtype", None)
    if torch_dtype is None or torch_dtype == "auto":
        params_dtype = torch.bfloat16
    elif isinstance(torch_dtype, str):
        params_dtype = getattr(torch, torch_dtype.replace("torch.", ""))
    else:
        params_dtype = torch_dtype

    # Detect first MoE layer (some models have dense layers before MoE)
    first_moe_layer = detect_first_moe_layer(model_path)

    logger.info(f"Model config: hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
                f"num_experts={num_experts}, top_k={top_k}, num_layers={num_layers}, "
                f"first_moe_layer={first_moe_layer}, params_dtype={params_dtype}")

    return MoEModelConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=num_layers,
        params_dtype=params_dtype,
        first_moe_layer=first_moe_layer,
    )


# =============================================================================
# Quantization Detection
# =============================================================================

def detect_quantization_type(model_path: str) -> str:
    """Detect quantization type from model files.

    Returns: "bf16", "fp8", "int4", etc.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Check for quantization_config
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is not None:
        quant_method = quant_config.get("quant_method", "")
        if "fp8" in quant_method.lower():
            return "fp8"
        elif "gptq" in quant_method.lower() or "awq" in quant_method.lower():
            return "int4"
        elif "compressed" in quant_method.lower():
            # Check for specific bit width
            bits = quant_config.get("bits", 16)
            if bits == 4:
                return "int4"
            elif bits == 8:
                return "fp8"

    # Check safetensors files for weight names
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensors_files:
        with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Check for quantized weight names
            for key in keys:
                if "weight_packed" in key or "qweight" in key:
                    return "int4"
                if "weight_scale_inv" in key:
                    return "fp8"

    return "bf16"


# =============================================================================
# topk_ids Generation
# =============================================================================

def generate_workload_topk_ids(
    num_tokens: int,
    top_k: int,
    gpu_slots: int,
    gpu_experts_active: int,
    cpu_experts_active: int,
    gpu_expert_ids: List[int],
    cpu_expert_ids: List[int],
    device: torch.device,
    seed: int = 42,
) -> torch.Tensor:
    """Generate topk_ids with specified GPU/CPU slot distribution.

    Args:
        num_tokens: Total number of input tokens
        top_k: Number of experts per token
        gpu_slots: Number of slots routed to GPU experts
        gpu_experts_active: Number of GPU experts with load
        cpu_experts_active: Number of CPU experts with load
        gpu_expert_ids: List of logical expert IDs on GPU
        cpu_expert_ids: List of logical expert IDs on CPU
        device: Target device
        seed: Random seed

    Returns:
        topk_ids tensor of shape [num_tokens, top_k]
    """
    total_slots = num_tokens * top_k
    cpu_slots = total_slots - gpu_slots

    assert gpu_slots + cpu_slots == total_slots, \
        f"gpu_slots ({gpu_slots}) + cpu_slots ({cpu_slots}) != total_slots ({total_slots})"
    assert gpu_experts_active <= len(gpu_expert_ids), \
        f"gpu_experts_active ({gpu_experts_active}) > len(gpu_expert_ids) ({len(gpu_expert_ids)})"
    assert cpu_experts_active <= len(cpu_expert_ids), \
        f"cpu_experts_active ({cpu_experts_active}) > len(cpu_expert_ids) ({len(cpu_expert_ids)})"

    # Create slot list
    all_slots = []

    # GPU slots distributed to active GPU experts (round-robin)
    active_gpu_ids = gpu_expert_ids[:gpu_experts_active]
    for i in range(gpu_slots):
        expert_id = active_gpu_ids[i % gpu_experts_active]
        all_slots.append(expert_id)

    # CPU slots distributed to active CPU experts (round-robin)
    active_cpu_ids = cpu_expert_ids[:cpu_experts_active]
    for i in range(cpu_slots):
        expert_id = active_cpu_ids[i % cpu_experts_active]
        all_slots.append(expert_id)

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(all_slots)

    # Convert to tensor
    topk_ids = torch.tensor(all_slots, dtype=torch.int32, device=device)
    topk_ids = topk_ids.reshape(num_tokens, top_k)

    return topk_ids


def generate_topk_weights(
    num_tokens: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate normalized topk_weights."""
    weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


# =============================================================================
# Simplified KT Benchmark Wrapper
# =============================================================================

class MockMoELayer(torch.nn.Module):
    """Mock MoE layer with necessary attributes for GPU method."""

    def __init__(
        self,
        num_experts: int,
        num_gpu_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        params_dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_gpu_experts
        self.num_gpu_experts = num_gpu_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.top_k = top_k

        # Set EP/TP parameters (single GPU, no parallelism)
        self.moe_ep_size = 1
        self.moe_ep_rank = 0
        self.moe_tp_size = 1
        self.moe_tp_rank = 0

        # Expert mask (not used in benchmark)
        self.expert_mask_gpu = None

        # Create moe_runner_config
        from sglang.srt.layers.moe import MoeRunnerConfig
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            layer_id=0,
            top_k=top_k,
            num_fused_shared_experts=0,
            params_dtype=params_dtype,
            activation="silu",
            apply_router_weight_on_input=False,
            inplace=True,
            no_combine=False,
            routed_scaling_factor=None,
            is_gated=True,
        )


class BenchmarkKTWrapper:
    """Simplified KT wrapper for benchmarking.

    This class implements the core hybrid CPU-GPU MoE computation logic
    without the complexity of the full KTEPWrapperMethod (no distributed
    support, no dynamic expert updates, etc.)
    """

    def __init__(
        self,
        model_config: MoEModelConfig,
        kt_weight_path: str,
        kt_num_gpu_experts: int,
        kt_cpuinfer: int,
        kt_threadpool_count: int,
        kt_method: str,
        kt_chunked_prefill_size: int,
        device: torch.device,
        layer_idx: int = 0,
        skip_gpu_weights: bool = False,
        override_top_k: Optional[int] = None,
    ):
        from kt_kernel import KTMoEWrapper

        self.model_config = model_config
        self.num_experts = model_config.num_experts
        self.num_gpu_experts = kt_num_gpu_experts
        self.num_cpu_experts = self.num_experts - kt_num_gpu_experts
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        # Use override_top_k if provided, otherwise use model's top_k
        self.top_k = override_top_k if override_top_k is not None else model_config.top_k
        self.params_dtype = model_config.params_dtype
        self.device = device
        self.layer_idx = layer_idx
        self.skip_gpu_weights = skip_gpu_weights

        # Create GPU experts mask (first kt_num_gpu_experts are on GPU)
        self.gpu_experts_mask = torch.zeros(self.num_experts, dtype=torch.bool)
        self.gpu_experts_mask[:kt_num_gpu_experts] = True
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(device)

        # Create expert ID lists
        self.gpu_expert_ids = list(range(kt_num_gpu_experts))
        self.cpu_expert_ids = list(range(kt_num_gpu_experts, self.num_experts))

        # Create logical to GPU index mapping
        self.logical_to_gpu_index = torch.full(
            (self.num_experts,), -1, dtype=torch.int32, device=device
        )
        for gpu_idx, logical_id in enumerate(self.gpu_expert_ids):
            self.logical_to_gpu_index[logical_id] = gpu_idx

        # Pre-create constant for CUDA graph compatibility (avoid CPU ops during capture)
        self._neg_one = torch.tensor(-1, dtype=torch.int32, device=device)

        # Initialize KTMoEWrapper for CPU experts
        logger.info(f"Initializing KTMoEWrapper: {kt_num_gpu_experts} GPU experts, "
                    f"{self.num_cpu_experts} CPU experts, layer_idx={layer_idx}")
        self.kt_wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            gpu_experts_mask=self.gpu_experts_mask,
            cpuinfer_threads=kt_cpuinfer,
            threadpool_count=kt_threadpool_count,
            weight_path=kt_weight_path,
            chunked_prefill_size=kt_chunked_prefill_size,
            method=kt_method,
            max_deferred_experts_per_token=0,
        )

        # Create mock layer and GPU method (skip if pure CPU mode)
        if not skip_gpu_weights:
            self._create_gpu_method()
        else:
            logger.info("Skipping GPU weight allocation (pure CPU mode)")
            self.mock_layer = None
            self.gpu_method = None

        # Create CPU stream for parallel execution
        self._cpu_stream = torch.cuda.Stream(device=device)
        self._sync_done_event = torch.cuda.Event()

        # Staging buffer cache
        self._staging_buffers: Dict[int, torch.Tensor] = {}

        # Flag to indicate if CPU weights are loaded
        self._cpu_weights_loaded = False

    def _create_gpu_method(self):
        """Create GPU method with optimized kernels."""
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        # Create mock layer
        self.mock_layer = MockMoELayer(
            num_experts=self.num_experts,
            num_gpu_experts=self.num_gpu_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            params_dtype=self.params_dtype,
            device=self.device,
        )

        # Create GPU method (use_triton_kernels=False for StandardTopKOutput compatibility)
        self.gpu_method = UnquantizedFusedMoEMethod(use_triton_kernels=False)

        # Create weights
        self.gpu_method.create_weights(
            layer=self.mock_layer,
            num_experts=self.num_gpu_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            params_dtype=self.params_dtype,
        )

        # Move weights to device and initialize randomly
        self.mock_layer.w13_weight.data = self.mock_layer.w13_weight.data.to(self.device)
        self.mock_layer.w2_weight.data = self.mock_layer.w2_weight.data.to(self.device)
        torch.nn.init.normal_(self.mock_layer.w13_weight.data, mean=0, std=0.02)
        torch.nn.init.normal_(self.mock_layer.w2_weight.data, mean=0, std=0.02)

        # Create MoE runner
        self.gpu_method.create_moe_runner(self.mock_layer, self.mock_layer.moe_runner_config)

        logger.info(f"GPU method created: w13_weight shape = {self.mock_layer.w13_weight.shape}, "
                    f"w2_weight shape = {self.mock_layer.w2_weight.shape}")

    def load_gpu_weights(self, model_path: str, layer_idx: int = 0):
        """Load GPU expert weights from model checkpoint.

        Note: For quantized models (FP8, INT4, etc.), weight shapes may not match
        bf16 layout. In such cases, we skip loading and use random weights instead.
        This is acceptable for benchmarking since weight values don't affect timing.
        """
        logger.info(f"Loading GPU weights from {model_path} for layer {layer_idx}")

        safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not safetensors_files:
            logger.warning("No safetensors files found, using random weights")
            return

        # Get weight shape info
        # For triton kernels: w13_weight is [num_experts, hidden_size, 2*intermediate]
        #                     w2_weight is [num_experts, intermediate, hidden_size]
        w13_weight = self.mock_layer.w13_weight.data
        w2_weight = self.mock_layer.w2_weight.data

        layer_prefix = f"model.layers.{layer_idx}."
        loaded_count = 0
        skipped_count = 0

        for sf_file in safetensors_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if layer_prefix not in key:
                        continue
                    if "experts" not in key:
                        continue
                    # Skip scale tensors for quantized models
                    if "scale" in key:
                        continue

                    # Extract expert ID
                    parts = key.split(".")
                    expert_idx = None
                    for i, part in enumerate(parts):
                        if part == "experts" and i + 1 < len(parts):
                            try:
                                expert_idx = int(parts[i + 1])
                            except ValueError:
                                continue
                            break

                    if expert_idx is None:
                        continue

                    # Skip CPU experts
                    if expert_idx >= self.num_gpu_experts:
                        continue

                    # Load weight
                    weight = f.get_tensor(key)

                    # Check if this is a quantized weight (shape mismatch)
                    # For bf16: w1/w3 should be [intermediate, hidden], w2 should be [hidden, intermediate]
                    expected_w13_shape = (self.intermediate_size, self.hidden_size)
                    expected_w2_shape = (self.hidden_size, self.intermediate_size)

                    try:
                        weight_bf16 = weight.to(self.params_dtype)

                        # For triton_kernels layout: [num_experts, K, N] where input is [M, K]
                        # w13: [num_experts, hidden_size, 2*intermediate]
                        # w2: [num_experts, intermediate, hidden_size]
                        if ("w1" in key or "gate_proj" in key) and "w13" not in key:
                            if weight_bf16.shape == expected_w13_shape:
                                w13_weight[expert_idx, :, :self.intermediate_size].copy_(weight_bf16.T)
                                loaded_count += 1
                            else:
                                skipped_count += 1
                        elif ("w3" in key or "up_proj" in key) and "w13" not in key:
                            if weight_bf16.shape == expected_w13_shape:
                                w13_weight[expert_idx, :, self.intermediate_size:].copy_(weight_bf16.T)
                                loaded_count += 1
                            else:
                                skipped_count += 1
                        elif ("w2" in key or "down_proj" in key) and "w13" not in key:
                            if weight_bf16.shape == expected_w2_shape:
                                w2_weight[expert_idx, :, :].copy_(weight_bf16.T)
                                loaded_count += 1
                            else:
                                skipped_count += 1
                    except Exception as e:
                        skipped_count += 1
                        continue

        if loaded_count > 0:
            logger.info(f"GPU weights loaded: {loaded_count} weight tensors")
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} weights (quantized format, using random weights instead)")

    def load_cpu_weights(self, layer_idx: int = 0):
        """Load CPU expert weights via KTMoEWrapper.

        This follows the same pattern as NativeMoEWrapper.load_weights():
        1. Use the internal loader to load expert weights from safetensors
        2. Build pointer lists
        3. Submit load_weights_task to CPU inference engine
        """
        logger.info("Loading CPU weights via KTMoEWrapper")

        # Create identity mapping for benchmark (physical = logical)
        # Must be int64 to match C++ uint64_t* interpretation
        physical_to_logical = torch.arange(
            self.num_experts, dtype=torch.int64, device="cpu"
        ).contiguous()

        # The KTMoEWrapper already has internal loader initialized during __init__
        # Just call load_weights with the mapping
        self.kt_wrapper.load_weights(physical_to_logical)
        self._cpu_weights_loaded = True
        logger.info("CPU weights loaded successfully")

    def _mask_and_remap_expert_ids(
        self,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mask CPU expert IDs and remap GPU expert IDs.

        CPU experts -> -1 (skipped by GPU kernel)
        GPU experts -> remapped to GPU weight indices
        """
        is_gpu_expert = self.gpu_experts_mask_cuda[topk_ids]
        remapped_ids = torch.where(
            is_gpu_expert,
            self.logical_to_gpu_index[topk_ids],
            self._neg_one  # Use pre-created constant for CUDA graph compatibility
        )
        return remapped_ids

    def _create_dispatch_output(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Create StandardDispatchOutput for GPU method."""
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

        # Create router_logits (not used in computation, just for interface)
        router_logits = torch.zeros(
            hidden_states.shape[0], self.num_experts,
            dtype=torch.float32, device=hidden_states.device
        )

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=router_logits,
        )

        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute hybrid CPU+GPU MoE forward pass.

        Steps:
        1. Submit CPU expert computation (non-blocking) - skipped if CPU weights not loaded
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results
        """
        batch_size = hidden_states.shape[0]

        # Remap expert IDs for GPU computation
        gpu_topk_ids = self._mask_and_remap_expert_ids(topk_ids)

        # Create dispatch output for GPU method
        dispatch_output = self._create_dispatch_output(hidden_states, gpu_topk_ids, topk_weights)

        if self._cpu_weights_loaded:
            # Full hybrid CPU+GPU path
            # Get or create staging buffer
            if batch_size not in self._staging_buffers:
                self._staging_buffers[batch_size] = torch.empty_like(hidden_states)
            staging_buffer = self._staging_buffers[batch_size]

            # Copy to staging buffer (main stream)
            staging_buffer.copy_(hidden_states, non_blocking=True)

            # Fork to CPU stream and submit CPU computation
            self._cpu_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._cpu_stream):
                self.kt_wrapper.submit_forward(
                    staging_buffer,
                    topk_ids,
                    topk_weights,
                    torch.cuda.current_stream(staging_buffer.device).cuda_stream,
                )

            # Execute GPU computation using optimized kernels
            gpu_combine_input = self.gpu_method.apply(self.mock_layer, dispatch_output)
            gpu_output = gpu_combine_input.hidden_states

            # Sync CPU results
            with torch.cuda.stream(self._cpu_stream):
                cpu_output = self.kt_wrapper.sync_forward(
                    staging_buffer,
                    torch.cuda.current_stream(staging_buffer.device).cuda_stream,
                )
                self._sync_done_event.record(self._cpu_stream)

            # Main stream waits for CPU stream
            torch.cuda.current_stream(self.device).wait_event(self._sync_done_event)

            # Merge results
            output = gpu_output + cpu_output
        else:
            # GPU-only path (when CPU weights not loaded)
            gpu_combine_input = self.gpu_method.apply(self.mock_layer, dispatch_output)
            output = gpu_combine_input.hidden_states

        return output

    def apply_gpu_only(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute GPU-only MoE forward pass (no CPU stream, no CPU operations).

        This method only runs GPU expert computation, ignoring CPU experts entirely.
        Useful for benchmarking pure GPU performance without CPU overhead.
        """
        # Remap expert IDs for GPU computation
        gpu_topk_ids = self._mask_and_remap_expert_ids(topk_ids)

        # Create dispatch output for GPU method
        dispatch_output = self._create_dispatch_output(hidden_states, gpu_topk_ids, topk_weights)

        # Execute GPU computation only
        gpu_combine_input = self.gpu_method.apply(self.mock_layer, dispatch_output)
        output = gpu_combine_input.hidden_states

        return output

    def apply_cpu_only(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Execute CPU-only MoE forward pass (no GPU computation).

        This method only runs CPU expert computation, ignoring GPU experts entirely.
        Useful for benchmarking pure CPU performance. Requires CPU weights to be loaded.
        """
        if not self._cpu_weights_loaded:
            raise RuntimeError("CPU weights not loaded. Call load_cpu_weights() first.")

        batch_size = hidden_states.shape[0]

        # Get or create staging buffer
        if batch_size not in self._staging_buffers:
            self._staging_buffers[batch_size] = torch.empty_like(hidden_states)
        staging_buffer = self._staging_buffers[batch_size]

        # Copy to staging buffer
        staging_buffer.copy_(hidden_states, non_blocking=True)

        # Submit and sync CPU computation (synchronous for simplicity)
        self.kt_wrapper.submit_forward(
            staging_buffer,
            topk_ids,
            topk_weights,
            torch.cuda.current_stream(staging_buffer.device).cuda_stream,
        )

        output = self.kt_wrapper.sync_forward(
            staging_buffer,
            torch.cuda.current_stream(staging_buffer.device).cuda_stream,
        )

        return output

    def apply_with_timing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Execute with detailed timing breakdown."""
        timings = {}
        batch_size = hidden_states.shape[0]

        # Remap expert IDs for GPU computation
        gpu_topk_ids = self._mask_and_remap_expert_ids(topk_ids)
        dispatch_output = self._create_dispatch_output(hidden_states, gpu_topk_ids, topk_weights)

        if self._cpu_weights_loaded:
            # Full hybrid CPU+GPU path with timing
            # Get or create staging buffer
            if batch_size not in self._staging_buffers:
                self._staging_buffers[batch_size] = torch.empty_like(hidden_states)
            staging_buffer = self._staging_buffers[batch_size]

            # Copy to staging
            staging_buffer.copy_(hidden_states, non_blocking=True)
            torch.cuda.synchronize()

            # 1. Submit CPU
            t0 = time.perf_counter()
            self.kt_wrapper.submit_forward(
                staging_buffer,
                topk_ids,
                topk_weights,
                torch.cuda.current_stream(staging_buffer.device).cuda_stream,
            )
            # Note: submit is non-blocking, so we don't sync here
            timings["submit_cpu"] = (time.perf_counter() - t0) * 1000

            # 2. GPU compute
            t0 = time.perf_counter()
            gpu_combine_input = self.gpu_method.apply(self.mock_layer, dispatch_output)
            gpu_output = gpu_combine_input.hidden_states
            torch.cuda.synchronize()
            timings["gpu_compute"] = (time.perf_counter() - t0) * 1000

            # 3. Sync CPU
            t0 = time.perf_counter()
            cpu_output = self.kt_wrapper.sync_forward(
                staging_buffer,
                torch.cuda.current_stream(staging_buffer.device).cuda_stream,
            )
            torch.cuda.synchronize()
            timings["sync_cpu"] = (time.perf_counter() - t0) * 1000

            # Merge
            output = gpu_output + cpu_output
        else:
            # GPU-only path with timing
            timings["submit_cpu"] = 0.0
            timings["sync_cpu"] = 0.0

            t0 = time.perf_counter()
            gpu_combine_input = self.gpu_method.apply(self.mock_layer, dispatch_output)
            output = gpu_combine_input.hidden_states
            torch.cuda.synchronize()
            timings["gpu_compute"] = (time.perf_counter() - t0) * 1000

        return output, timings


# =============================================================================
# Benchmark Runner
# =============================================================================

class CUDAGraphBenchmark:
    """CUDA Graph wrapper for hybrid CPU+GPU MoE computation benchmark.

    Note: The KT wrapper's submit_forward/sync_forward use cudaLaunchHostFunc
    internally, which CAN be captured by CUDA graph. So we capture the entire
    hybrid computation flow.
    """

    def __init__(
        self,
        wrapper: BenchmarkKTWrapper,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gpu_only: bool = False,
        cpu_only: bool = False,
    ):
        self.wrapper = wrapper
        self.device = hidden_states.device
        self.graph = None
        self.gpu_only = gpu_only
        self.cpu_only = cpu_only

        # Input buffers (will be captured)
        self.input_hidden_states = hidden_states.clone()
        self.input_topk_ids = topk_ids.clone()
        self.input_topk_weights = topk_weights.clone()

        # Output buffer (will be set during capture)
        self.output = None

    def capture(self):
        """Capture the entire hybrid CPU+GPU computation into CUDA graph."""
        if self.cpu_only:
            mode_str = "CPU-only"
        elif self.gpu_only:
            mode_str = "GPU-only"
        else:
            mode_str = "hybrid CPU+GPU"
        logger.info(f"Capturing CUDA graph for {mode_str} computation...")
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()

        # Capture stream
        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_device(self.device)

        # Select apply function based on mode
        if self.cpu_only:
            apply_fn = self.wrapper.apply_cpu_only
        elif self.gpu_only:
            apply_fn = self.wrapper.apply_gpu_only
        else:
            apply_fn = self.wrapper.apply

        with torch.cuda.graph(self.graph, stream=capture_stream):
            # Capture the apply() - for gpu_only this is pure GPU computation,
            # for hybrid this includes:
            # 1. Staging buffer copy
            # 2. KT submit_forward (uses cudaLaunchHostFunc, capturable)
            # 3. GPU expert computation
            # 4. KT sync_forward (uses cudaLaunchHostFunc, capturable)
            # 5. Output merge
            self.output = apply_fn(
                self.input_hidden_states,
                self.input_topk_ids,
                self.input_topk_weights,
            )
            capture_stream.wait_stream(torch.cuda.current_stream())

        torch.cuda.synchronize()
        logger.info("CUDA graph captured successfully")

    def replay(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """Run hybrid computation using captured graph."""
        # Copy inputs to captured buffers
        self.input_hidden_states.copy_(hidden_states)
        self.input_topk_ids.copy_(topk_ids)
        self.input_topk_weights.copy_(topk_weights)

        # Replay graph
        self.graph.replay()
        torch.cuda.synchronize()

        return self.output


def run_benchmark(
    wrappers: List[BenchmarkKTWrapper],
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    use_cuda_graph: bool,
    gpu_only: bool = False,
    cpu_only: bool = False,
    throughput_mode: bool = False,
) -> Dict[str, float]:
    """Run the benchmark and collect timing statistics.

    Args:
        wrappers: List of BenchmarkKTWrapper instances (one per layer).
                  Multiple wrappers are used to alternate between layers
                  and reduce caching effects.
        gpu_only: If True, only run GPU computation (no CPU).
        cpu_only: If True, only run CPU computation (no GPU).
        throughput_mode: If True, run all iterations without per-iteration sync
                        to measure throughput (better CPU utilization).
    """
    import statistics

    device = hidden_states.device
    num_layers = len(wrappers)

    # Select apply functions based on mode
    if cpu_only:
        apply_fns = [w.apply_cpu_only for w in wrappers]
    elif gpu_only:
        apply_fns = [w.apply_gpu_only for w in wrappers]
    else:
        apply_fns = [w.apply for w in wrappers]

    # Warmup - alternate between layers
    mode_str = "cpu_only" if cpu_only else ("gpu_only" if gpu_only else "hybrid")
    logger.info(f"Running {warmup_iters} warmup iterations across {num_layers} layers... (mode={mode_str})")
    for i in range(warmup_iters):
        layer_idx = i % num_layers
        _ = apply_fns[layer_idx](hidden_states, topk_ids, topk_weights)
    torch.cuda.synchronize()

    if use_cuda_graph:
        # CUDA graph mode - capture graphs for all layers
        if cpu_only:
            mode_str = "CPU-only"
        elif gpu_only:
            mode_str = "GPU-only"
        else:
            mode_str = "full hybrid CPU+GPU"
        logger.info(f"CUDA graph mode: capturing {mode_str} computation for {num_layers} layers")

        # Create and capture CUDA graphs for each layer
        cuda_graph_runners = []
        for i, wrapper in enumerate(wrappers):
            logger.info(f"  Capturing graph for layer {i}...")
            runner = CUDAGraphBenchmark(
                wrapper, hidden_states, topk_ids, topk_weights,
                gpu_only=gpu_only, cpu_only=cpu_only
            )
            runner.capture()
            cuda_graph_runners.append(runner)

        # Warmup graph replay - alternate between layers
        logger.info(f"Warming up CUDA graph replay ({warmup_iters} iterations across {num_layers} layers)...")
        for i in range(warmup_iters):
            layer_idx = i % num_layers
            cuda_graph_runners[layer_idx].replay(hidden_states, topk_ids, topk_weights)

        if throughput_mode:
            # Throughput mode: run all iterations back-to-back, sync only at the end
            logger.info(f"Running {bench_iters} iterations in THROUGHPUT mode (no per-iteration sync)...")
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for i in range(bench_iters):
                layer_idx = i % num_layers
                runner = cuda_graph_runners[layer_idx]

                # Copy inputs to captured buffers
                runner.input_hidden_states.copy_(hidden_states)
                runner.input_topk_ids.copy_(topk_ids)
                runner.input_topk_weights.copy_(topk_weights)

                # Replay without sync
                runner.graph.replay()

            end_event.record()
            torch.cuda.synchronize()

            total_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = total_time_ms / bench_iters
            throughput = bench_iters / (total_time_ms / 1000.0)  # iterations per second

            return {
                "total_ms": avg_time_ms,
                "total_std_ms": 0.0,  # Can't measure std in throughput mode
                "total_min_ms": avg_time_ms,
                "total_max_ms": avg_time_ms,
                "throughput_iters_per_sec": throughput,
                "total_batch_time_ms": total_time_ms,
            }
        else:
            # Latency mode: sync after each iteration for per-iteration timing
            logger.info(f"Running {bench_iters} CUDA graph replay iterations (alternating {num_layers} layers)...")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            all_times = []
            for i in range(bench_iters):
                layer_idx = i % num_layers
                runner = cuda_graph_runners[layer_idx]

                # Copy inputs to captured buffers
                runner.input_hidden_states.copy_(hidden_states)
                runner.input_topk_ids.copy_(topk_ids)
                runner.input_topk_weights.copy_(topk_weights)

                # Measure replay time
                start_event.record()
                runner.graph.replay()
                end_event.record()
                torch.cuda.synchronize()
                all_times.append(start_event.elapsed_time(end_event))

            return {
                "total_ms": statistics.mean(all_times),
                "total_std_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0,
                "total_min_ms": min(all_times),
                "total_max_ms": max(all_times),
            }
    else:
        # Non-CUDA graph mode with timing breakdown - alternate between layers
        logger.info(f"Running {bench_iters} benchmark iterations (alternating {num_layers} layers)...")

        all_timings = {
            "submit_cpu": [],
            "gpu_compute": [],
            "sync_cpu": [],
            "total": [],
        }

        for i in range(bench_iters):
            layer_idx = i % num_layers
            wrapper = wrappers[layer_idx]

            torch.cuda.synchronize()
            total_start = time.perf_counter()
            _, timings = wrapper.apply_with_timing(hidden_states, topk_ids, topk_weights)
            torch.cuda.synchronize()
            total_time = (time.perf_counter() - total_start) * 1000

            all_timings["submit_cpu"].append(timings["submit_cpu"])
            all_timings["gpu_compute"].append(timings["gpu_compute"])
            all_timings["sync_cpu"].append(timings["sync_cpu"])
            all_timings["total"].append(total_time)

        # Calculate statistics
        results = {}
        for key, values in all_timings.items():
            results[f"{key}_mean_ms"] = statistics.mean(values)
            results[f"{key}_std_ms"] = statistics.stdev(values) if len(values) > 1 else 0
            results[f"{key}_min_ms"] = min(values)
            results[f"{key}_max_ms"] = max(values)

        return results


def print_results(
    results: Dict[str, float],
    use_cuda_graph: bool,
    args,
    model_config: MoEModelConfig,
    gpu_only_mode: bool = False,
    cpu_only_mode: bool = False,
    num_layers: int = 1,
    throughput_mode: bool = False,
):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    if throughput_mode:
        print("  (THROUGHPUT MODE - no per-iteration sync, higher CPU utilization)")
    if cpu_only_mode:
        print("  (CPU-ONLY MODE - no GPU weights allocated, pure CPU computation)")
    elif gpu_only_mode:
        if args.gpu_only:
            print("  (GPU-ONLY MODE - no CPU stream or CPU operations)")
        else:
            print("  (GPU-ONLY MODE - CPU weights not loaded)")
    if getattr(args, 'zero_cpu_slots', False):
        print("  (ZERO-CPU-SLOTS - all slots routed to GPU, CPU does no work)")
    if getattr(args, 'zero_gpu_slots', False):
        print("  (ZERO-GPU-SLOTS - all slots routed to CPU, GPU does no work)")
    if num_layers > 1:
        print(f"  (Alternating {num_layers} layers to reduce caching effects)")
    print("=" * 60)

    # Print configuration summary
    total_slots = args.num_tokens * model_config.top_k
    cpu_slots = total_slots - args.gpu_slots
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Tokens: {args.num_tokens}, Top-k: {model_config.top_k}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Intermediate size: {model_config.intermediate_size}")
    print(f"  Total experts: {model_config.num_experts}")
    print(f"  GPU experts: {args.kt_num_gpu_experts} (active: {args.gpu_experts_active})")
    if not gpu_only_mode:
        print(f"  CPU experts: {model_config.num_experts - args.kt_num_gpu_experts} (active: {args.cpu_experts_active})")
        print(f"  GPU slots: {args.gpu_slots} / {total_slots} ({100*args.gpu_slots/total_slots:.1f}%)")
        print(f"  CPU slots: {cpu_slots} / {total_slots} ({100*cpu_slots/total_slots:.1f}%)")
    else:
        mode_reason = "--gpu-only" if args.gpu_only else "--skip-weight-loading"
        print(f"  CPU experts: SKIPPED ({mode_reason})")
        print(f"  GPU slots: {args.gpu_slots} / {total_slots} ({100*args.gpu_slots/total_slots:.1f}%)")
        print(f"  CPU slots: SKIPPED")

    if use_cuda_graph:
        if throughput_mode:
            print(f"\nThroughput Results (CUDA Graph):")
            print(f"  Avg latency:  {results['total_ms']:.3f} ms")
            print(f"  Throughput:   {results['throughput_iters_per_sec']:.1f} iterations/sec")
            print(f"  Total time:   {results['total_batch_time_ms']:.1f} ms for {args.bench_iters} iterations")
        else:
            print(f"\nTotal apply() time (CUDA Graph):")
            print(f"  Mean:  {results['total_ms']:.3f} ms")
            print(f"  Std:   {results['total_std_ms']:.3f} ms")
            print(f"  Min:   {results['total_min_ms']:.3f} ms")
            print(f"  Max:   {results['total_max_ms']:.3f} ms")
    else:
        print(f"\nBreakdown (mean over iterations):")
        print(f"  Submit CPU:     {results['submit_cpu_mean_ms']:.3f} ms "
              f"(std: {results['submit_cpu_std_ms']:.3f})")
        print(f"  GPU compute:    {results['gpu_compute_mean_ms']:.3f} ms "
              f"(std: {results['gpu_compute_std_ms']:.3f})")
        print(f"  Sync CPU:       {results['sync_cpu_mean_ms']:.3f} ms "
              f"(std: {results['sync_cpu_std_ms']:.3f})")
        print(f"  Total apply():  {results['total_mean_ms']:.3f} ms "
              f"(std: {results['total_std_ms']:.3f})")
        print(f"\nMin/Max:")
        print(f"  Total: {results['total_min_ms']:.3f} ms / {results['total_max_ms']:.3f} ms")

    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this benchmark")

    # Initialize global server args (required for MoE computation)
    setup_minimal_server_args(args.model)

    # Load model configuration
    logger.info(f"Loading model config from {args.model}")
    model_config = load_model_config(args.model)

    # Determine effective top_k (can be overridden for zero-slots modes)
    effective_top_k = args.override_top_k if args.override_top_k is not None else model_config.top_k
    total_slots = args.num_tokens * effective_top_k
    num_cpu_experts = model_config.num_experts - args.kt_num_gpu_experts

    # Determine slot distribution based on zero-slots flags
    if args.zero_cpu_slots:
        # All slots go to GPU
        actual_gpu_slots = total_slots
        actual_cpu_slots = 0
        logger.info(f"--zero-cpu-slots mode: all {total_slots} slots go to GPU")
    elif args.zero_gpu_slots:
        # All slots go to CPU
        actual_gpu_slots = 0
        actual_cpu_slots = total_slots
        logger.info(f"--zero-gpu-slots mode: all {total_slots} slots go to CPU")
    else:
        actual_gpu_slots = args.gpu_slots
        actual_cpu_slots = total_slots - args.gpu_slots

    logger.info(f"Workload configuration:")
    logger.info(f"  Tokens: {args.num_tokens}, Top-k: {effective_top_k}" +
                (f" (overridden from {model_config.top_k})" if args.override_top_k else ""))
    logger.info(f"  Total slots: {total_slots} (GPU: {actual_gpu_slots}, CPU: {actual_cpu_slots})")
    logger.info(f"  GPU experts: {args.kt_num_gpu_experts} (active: {args.gpu_experts_active})")
    logger.info(f"  CPU experts: {num_cpu_experts} (active: {args.cpu_experts_active})")

    if actual_gpu_slots > total_slots:
        raise ValueError(f"gpu_slots ({actual_gpu_slots}) > total_slots ({total_slots})")
    if args.gpu_experts_active > args.kt_num_gpu_experts:
        raise ValueError(f"gpu_experts_active ({args.gpu_experts_active}) > "
                         f"kt_num_gpu_experts ({args.kt_num_gpu_experts})")
    if args.cpu_experts_active > num_cpu_experts:
        raise ValueError(f"cpu_experts_active ({args.cpu_experts_active}) > "
                         f"num_cpu_experts ({num_cpu_experts})")
    if args.cpu_only and args.gpu_only:
        raise ValueError("Cannot use both --cpu-only and --gpu-only at the same time")
    if args.zero_cpu_slots and args.zero_gpu_slots:
        raise ValueError("Cannot use both --zero-cpu-slots and --zero-gpu-slots at the same time")

    # Determine starting layer (skip dense layers if necessary)
    start_layer = args.layer_idx
    if start_layer < model_config.first_moe_layer:
        logger.warning(f"Layer {start_layer} is a dense layer (no MoE experts). "
                       f"Adjusting to first MoE layer: {model_config.first_moe_layer}")
        start_layer = model_config.first_moe_layer

    # Create wrappers for each layer
    # Skip GPU weights allocation if --cpu-only to save GPU memory
    skip_gpu_weights = args.cpu_only

    wrappers = []
    for layer_offset in range(args.num_layers):
        layer_idx = start_layer + layer_offset
        logger.info(f"Creating BenchmarkKTWrapper for layer {layer_idx}...")
        wrapper = BenchmarkKTWrapper(
            model_config=model_config,
            kt_weight_path=args.kt_weight_path,
            kt_num_gpu_experts=args.kt_num_gpu_experts,
            kt_cpuinfer=args.kt_cpuinfer,
            kt_threadpool_count=args.kt_threadpool_count,
            kt_method=args.kt_method,
            kt_chunked_prefill_size=args.kt_chunked_prefill_size,
            device=device,
            layer_idx=layer_idx,
            skip_gpu_weights=skip_gpu_weights,
            override_top_k=args.override_top_k,
        )

        # Load weights
        if args.skip_weight_loading:
            if layer_offset == 0:
                logger.warning("Skipping weight loading (--skip-weight-loading), using random weights")
        else:
            # Skip GPU weight loading if --cpu-only
            if not args.cpu_only:
                wrapper.load_gpu_weights(args.model, layer_idx=layer_idx)
            wrapper.load_cpu_weights(layer_idx=layer_idx)

        wrappers.append(wrapper)

    logger.info(f"Created {len(wrappers)} layer wrappers")

    # Generate test inputs (use first wrapper's expert IDs, same for all layers)
    logger.info("Generating test inputs...")
    hidden_states = torch.randn(
        args.num_tokens, model_config.hidden_size,
        dtype=model_config.params_dtype, device=device
    )

    # Generate topk_ids with the effective_top_k (no padding needed)
    topk_ids = generate_workload_topk_ids(
        num_tokens=args.num_tokens,
        top_k=effective_top_k,
        gpu_slots=actual_gpu_slots,
        gpu_experts_active=args.gpu_experts_active,
        cpu_experts_active=args.cpu_experts_active,
        gpu_expert_ids=wrappers[0].gpu_expert_ids,
        cpu_expert_ids=wrappers[0].cpu_expert_ids,
        device=device,
        seed=args.seed,
    )
    topk_weights = generate_topk_weights(args.num_tokens, effective_top_k, device)

    # Verify topk_ids distribution
    gpu_count = (topk_ids < args.kt_num_gpu_experts).sum().item()
    cpu_count = (topk_ids >= args.kt_num_gpu_experts).sum().item()
    logger.info(f"Generated topk_ids: GPU slots = {gpu_count}, CPU slots = {cpu_count}")

    # Run benchmark (alternating between layers)
    results = run_benchmark(
        wrappers=wrappers,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        use_cuda_graph=args.cuda_graph,
        gpu_only=args.gpu_only,
        cpu_only=args.cpu_only,
        throughput_mode=args.throughput_mode,
    )

    # Print results
    # gpu_only_mode is True if either --gpu-only or --skip-weight-loading is set
    gpu_only_mode = args.gpu_only or args.skip_weight_loading
    print_results(results, args.cuda_graph, args, model_config,
                  gpu_only_mode=gpu_only_mode, cpu_only_mode=args.cpu_only,
                  num_layers=args.num_layers, throughput_mode=args.throughput_mode)


if __name__ == "__main__":
    main()
