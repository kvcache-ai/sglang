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
    parser.add_argument("--kt-cpuinfer", type=int, default=32,
                        help="Number of CPU inference threads")
    parser.add_argument("--kt-threadpool-count", type=int, default=1,
                        help="Number of thread pools for CPU computation")
    parser.add_argument("--kt-method", type=str, default="int4",
                        choices=["int4", "fp8", "bf16"],
                        help="CPU computation method")
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


def load_model_config(model_path: str) -> MoEModelConfig:
    """Load MoE configuration from model."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Extract hidden_size
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Model config missing 'hidden_size'")

    # Extract intermediate_size (try multiple attribute names)
    intermediate_size = getattr(config, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = getattr(config, "moe_intermediate_size", None)
    if intermediate_size is None:
        raise ValueError("Model config missing 'intermediate_size'")

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

    logger.info(f"Model config: hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
                f"num_experts={num_experts}, top_k={top_k}, num_layers={num_layers}, "
                f"params_dtype={params_dtype}")

    return MoEModelConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_layers=num_layers,
        params_dtype=params_dtype,
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
    ):
        from kt_kernel import KTMoEWrapper

        self.model_config = model_config
        self.num_experts = model_config.num_experts
        self.num_gpu_experts = kt_num_gpu_experts
        self.num_cpu_experts = self.num_experts - kt_num_gpu_experts
        self.hidden_size = model_config.hidden_size
        self.intermediate_size = model_config.intermediate_size
        self.top_k = model_config.top_k
        self.params_dtype = model_config.params_dtype
        self.device = device

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

        # Initialize KTMoEWrapper for CPU experts
        logger.info(f"Initializing KTMoEWrapper: {kt_num_gpu_experts} GPU experts, "
                    f"{self.num_cpu_experts} CPU experts")
        self.kt_wrapper = KTMoEWrapper(
            layer_idx=0,
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

        # Create mock layer and GPU method
        self._create_gpu_method()

        # Create CPU stream for parallel execution
        self._cpu_stream = torch.cuda.Stream(device=device)
        self._sync_done_event = torch.cuda.Event()

        # Staging buffer cache
        self._staging_buffers: Dict[int, torch.Tensor] = {}

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

        # Create GPU method (use triton kernels for better performance)
        self.gpu_method = UnquantizedFusedMoEMethod(use_triton_kernels=True)

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
        """Load GPU expert weights from model checkpoint."""
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

        for sf_file in safetensors_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if layer_prefix not in key:
                        continue
                    if "experts" not in key:
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
                    weight = f.get_tensor(key).to(self.params_dtype)

                    # For triton_kernels layout: [num_experts, K, N] where input is [M, K]
                    # w13: [num_experts, hidden_size, 2*intermediate]
                    # w2: [num_experts, intermediate, hidden_size]
                    if "w1" in key or "gate_proj" in key:
                        # Gate projection: [intermediate, hidden] -> transpose to [hidden, intermediate]
                        # Goes to first half of w13's last dim
                        w13_weight[expert_idx, :, :self.intermediate_size].copy_(weight.T)
                        loaded_count += 1
                    elif "w3" in key or "up_proj" in key:
                        # Up projection: [intermediate, hidden] -> transpose to [hidden, intermediate]
                        # Goes to second half of w13's last dim
                        w13_weight[expert_idx, :, self.intermediate_size:].copy_(weight.T)
                        loaded_count += 1
                    elif "w2" in key or "down_proj" in key:
                        # Down projection: [hidden, intermediate] -> transpose to [intermediate, hidden]
                        w2_weight[expert_idx, :, :].copy_(weight.T)
                        loaded_count += 1

        logger.info(f"GPU weights loaded: {loaded_count} weight tensors")

    def load_cpu_weights(self):
        """Load CPU expert weights via KTMoEWrapper."""
        logger.info("Loading CPU weights via KTMoEWrapper")
        # Create identity mapping for benchmark (physical = logical)
        physical_to_logical = torch.arange(self.num_experts, dtype=torch.int32)
        self.kt_wrapper.load_weights(physical_to_logical)
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
            torch.tensor(-1, dtype=torch.int32, device=topk_ids.device)
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
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results
        """
        batch_size = hidden_states.shape[0]

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

        # Remap expert IDs for GPU computation
        gpu_topk_ids = self._mask_and_remap_expert_ids(topk_ids)

        # Create dispatch output for GPU method
        dispatch_output = self._create_dispatch_output(hidden_states, gpu_topk_ids, topk_weights)

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
        gpu_topk_ids = self._mask_and_remap_expert_ids(topk_ids)
        dispatch_output = self._create_dispatch_output(hidden_states, gpu_topk_ids, topk_weights)

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
    ):
        self.wrapper = wrapper
        self.device = hidden_states.device
        self.graph = None

        # Input buffers (will be captured)
        self.input_hidden_states = hidden_states.clone()
        self.input_topk_ids = topk_ids.clone()
        self.input_topk_weights = topk_weights.clone()

        # Output buffer (will be set during capture)
        self.output = None

    def capture(self):
        """Capture the entire hybrid CPU+GPU computation into CUDA graph."""
        logger.info("Capturing CUDA graph for hybrid CPU+GPU computation...")
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()

        # Capture stream
        capture_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_device(self.device)

        with torch.cuda.graph(self.graph, stream=capture_stream):
            # Capture the full apply() - this includes:
            # 1. Staging buffer copy
            # 2. KT submit_forward (uses cudaLaunchHostFunc, capturable)
            # 3. GPU expert computation
            # 4. KT sync_forward (uses cudaLaunchHostFunc, capturable)
            # 5. Output merge
            self.output = self.wrapper.apply(
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
    wrapper: BenchmarkKTWrapper,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    warmup_iters: int,
    bench_iters: int,
    use_cuda_graph: bool,
) -> Dict[str, float]:
    """Run the benchmark and collect timing statistics."""
    import statistics

    device = hidden_states.device

    # Warmup
    logger.info(f"Running {warmup_iters} warmup iterations...")
    for _ in range(warmup_iters):
        _ = wrapper.apply(hidden_states, topk_ids, topk_weights)
    torch.cuda.synchronize()

    if use_cuda_graph:
        # CUDA graph mode - capture the entire hybrid CPU+GPU computation
        # KT wrapper's submit_forward/sync_forward use cudaLaunchHostFunc,
        # which CAN be captured by CUDA graph
        logger.info("CUDA graph mode: capturing full hybrid CPU+GPU computation")

        # Create and capture CUDA graph
        cuda_graph_runner = CUDAGraphBenchmark(
            wrapper, hidden_states, topk_ids, topk_weights
        )
        cuda_graph_runner.capture()

        # Warmup graph replay
        logger.info(f"Warming up CUDA graph replay ({warmup_iters} iterations)...")
        for _ in range(warmup_iters):
            cuda_graph_runner.replay(hidden_states, topk_ids, topk_weights)

        # Benchmark graph replay
        logger.info(f"Running {bench_iters} CUDA graph replay iterations...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        all_times = []
        for _ in range(bench_iters):
            # Copy inputs to captured buffers
            cuda_graph_runner.input_hidden_states.copy_(hidden_states)
            cuda_graph_runner.input_topk_ids.copy_(topk_ids)
            cuda_graph_runner.input_topk_weights.copy_(topk_weights)

            # Measure replay time
            start_event.record()
            cuda_graph_runner.graph.replay()
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
        # Non-CUDA graph mode with timing breakdown
        logger.info(f"Running {bench_iters} benchmark iterations...")

        all_timings = {
            "submit_cpu": [],
            "gpu_compute": [],
            "sync_cpu": [],
            "total": [],
        }

        for _ in range(bench_iters):
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
):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
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
    print(f"  CPU experts: {model_config.num_experts - args.kt_num_gpu_experts} (active: {args.cpu_experts_active})")
    print(f"  GPU slots: {args.gpu_slots} / {total_slots} ({100*args.gpu_slots/total_slots:.1f}%)")
    print(f"  CPU slots: {cpu_slots} / {total_slots} ({100*cpu_slots/total_slots:.1f}%)")

    if use_cuda_graph:
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

    # Load model configuration
    logger.info(f"Loading model config from {args.model}")
    model_config = load_model_config(args.model)

    # Validate arguments
    total_slots = args.num_tokens * model_config.top_k
    cpu_slots = total_slots - args.gpu_slots
    num_cpu_experts = model_config.num_experts - args.kt_num_gpu_experts

    logger.info(f"Workload configuration:")
    logger.info(f"  Tokens: {args.num_tokens}, Top-k: {model_config.top_k}")
    logger.info(f"  Total slots: {total_slots} (GPU: {args.gpu_slots}, CPU: {cpu_slots})")
    logger.info(f"  GPU experts: {args.kt_num_gpu_experts} (active: {args.gpu_experts_active})")
    logger.info(f"  CPU experts: {num_cpu_experts} (active: {args.cpu_experts_active})")

    if args.gpu_slots > total_slots:
        raise ValueError(f"gpu_slots ({args.gpu_slots}) > total_slots ({total_slots})")
    if args.gpu_experts_active > args.kt_num_gpu_experts:
        raise ValueError(f"gpu_experts_active ({args.gpu_experts_active}) > "
                         f"kt_num_gpu_experts ({args.kt_num_gpu_experts})")
    if args.cpu_experts_active > num_cpu_experts:
        raise ValueError(f"cpu_experts_active ({args.cpu_experts_active}) > "
                         f"num_cpu_experts ({num_cpu_experts})")

    # Create wrapper
    logger.info("Creating BenchmarkKTWrapper...")
    wrapper = BenchmarkKTWrapper(
        model_config=model_config,
        kt_weight_path=args.kt_weight_path,
        kt_num_gpu_experts=args.kt_num_gpu_experts,
        kt_cpuinfer=args.kt_cpuinfer,
        kt_threadpool_count=args.kt_threadpool_count,
        kt_method=args.kt_method,
        kt_chunked_prefill_size=args.kt_chunked_prefill_size,
        device=device,
    )

    # Load weights
    wrapper.load_gpu_weights(args.model, layer_idx=0)
    wrapper.load_cpu_weights()

    # Generate test inputs
    logger.info("Generating test inputs...")
    hidden_states = torch.randn(
        args.num_tokens, model_config.hidden_size,
        dtype=model_config.params_dtype, device=device
    )
    topk_ids = generate_workload_topk_ids(
        num_tokens=args.num_tokens,
        top_k=model_config.top_k,
        gpu_slots=args.gpu_slots,
        gpu_experts_active=args.gpu_experts_active,
        cpu_experts_active=args.cpu_experts_active,
        gpu_expert_ids=wrapper.gpu_expert_ids,
        cpu_expert_ids=wrapper.cpu_expert_ids,
        device=device,
        seed=args.seed,
    )
    topk_weights = generate_topk_weights(args.num_tokens, model_config.top_k, device)

    # Verify topk_ids distribution
    gpu_count = (topk_ids < args.kt_num_gpu_experts).sum().item()
    cpu_count = (topk_ids >= args.kt_num_gpu_experts).sum().item()
    logger.info(f"Generated topk_ids: GPU slots = {gpu_count}, CPU slots = {cpu_count}")

    # Run benchmark
    results = run_benchmark(
        wrapper=wrapper,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
        use_cuda_graph=args.cuda_graph,
    )

    # Print results
    print_results(results, args.cuda_graph, args, model_config)


if __name__ == "__main__":
    main()
