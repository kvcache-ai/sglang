# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import ctypes
import logging
import time
from dataclasses import dataclass, replace
from multiprocessing import shared_memory
from types import MethodType
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter as TorchParameter

from sglang.srt.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


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
        prefill_token_threshold: token threshold for enabling full GPU fallback
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
    prefill_token_threshold: Optional[int] = None


@dataclass
class _FullMoEContext:
    layer: torch.nn.Module
    method: FusedMoEMethodBase
    runner_config: "MoeRunnerConfig"


_SHARED_FULL_CONTEXT = None


class SharedFullContext:
    def __init__(
        self,
        layer: torch.nn.Module,
        method_factory: Callable,
        init_args: tuple,
        extra_weight_attrs: dict,
        global_num_experts: int,
        moe_runner_config: "MoeRunnerConfig",
    ):
        self.gpu_layer = None
        self.gpu_method = None
        self.runner_config = None
        self.original_params = {}
        self.original_buffers = {}
        self.cpu_buffers = {}  # Store CPU buffers for weight loading

        self._build_layers(
            layer,
            method_factory,
            init_args,
            extra_weight_attrs,
            global_num_experts,
            moe_runner_config,
        )

        # Capture original tensors to support restoration before loading
        for name, param in self.gpu_layer.named_parameters():
            self.original_params[name] = param
        for name, buf in self.gpu_layer.named_buffers():
            self.original_buffers[name] = buf

        # Create CPU buffers once for weight loading (shared across layers)
        self._create_cpu_buffers()

    def _build_layers(
        self,
        layer,
        method_factory,
        init_args,
        extra_weight_attrs,
        global_num_experts,
        moe_runner_config,
    ):
        hidden_size, intermediate_size_per_partition, params_dtype = init_args

        # 1. Create GPU layer
        layer_cls = layer.__class__
        self.gpu_layer = layer_cls.__new__(layer_cls)
        torch.nn.Module.__init__(self.gpu_layer)

        # Copy attributes
        copy_attrs = (
            "top_k",
            "layer_id",
            "hidden_size",
            "intermediate_size_per_partition",
            "moe_tp_size",
            "moe_tp_rank",
            "moe_ep_size",
            "moe_ep_rank",
            "num_fused_shared_experts",
            "quant_config",
            "use_triton_kernels",
            "use_presharded_weights",
            "reduce_results",
            "weight_prefix",
            "moe_runner_config",
            "should_fuse_routed_scaling_factor_in_topk",
        )
        for attr in copy_attrs:
            if hasattr(layer, attr):
                setattr(self.gpu_layer, attr, getattr(layer, attr))

        self.gpu_layer.num_experts = getattr(layer, "num_experts", global_num_experts)
        self.gpu_layer.num_local_experts = global_num_experts
        self.gpu_layer.intermediate_size_per_partition = intermediate_size_per_partition
        self.gpu_layer.params_dtype = params_dtype
        self.gpu_layer.training = layer.training
        self.gpu_layer.num_gpu_experts = global_num_experts

        target_device = next(layer.parameters()).device
        self.gpu_layer.to(target_device)

        # Handle extra weight attrs
        local_extra_weight_attrs = dict(extra_weight_attrs)
        if "weight_loader" in local_extra_weight_attrs and hasattr(
            self.gpu_layer, "weight_loader"
        ):
            local_extra_weight_attrs["weight_loader"] = self.gpu_layer.weight_loader
        if "weight_loader_fused" in local_extra_weight_attrs and hasattr(
            self.gpu_layer, "weight_loader_fused"
        ):
            local_extra_weight_attrs[
                "weight_loader_fused"
            ] = self.gpu_layer.weight_loader_fused

        # Create method
        self.gpu_method = method_factory(self.gpu_layer)
        self.gpu_layer.quant_method = self.gpu_method

        # Hack register_parameter for device placement
        original_register_parameter = self.gpu_layer.register_parameter
        original_register_buffer = self.gpu_layer.register_buffer
        parameter_cls = TorchParameter
        original_parameter_new = parameter_cls.__new__

        def _register_parameter_on_device(obj, name, param):
            if param is not None and param.device != target_device:
                with torch.no_grad():
                    param.data = param.data.to(target_device)
            return original_register_parameter(name, param)

        def _register_buffer_on_device(obj, name, tensor, persistent=True):
            if tensor is not None and tensor.device != target_device:
                tensor = tensor.to(target_device)
            return original_register_buffer(name, tensor, persistent=persistent)

        def _parameter_new(cls, data=None, requires_grad=True):
            if data is not None and data.device != target_device:
                with torch.no_grad():
                    data = data.to(target_device)
            return original_parameter_new(cls, data, requires_grad)

        self.gpu_layer.register_parameter = MethodType(
            _register_parameter_on_device, self.gpu_layer
        )
        self.gpu_layer.register_buffer = MethodType(
            _register_buffer_on_device, self.gpu_layer
        )
        parameter_cls.__new__ = _parameter_new

        try:
            self.gpu_method.create_weights(
                layer=self.gpu_layer,
                num_experts=global_num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                **local_extra_weight_attrs,
            )
        finally:
            self.gpu_layer.register_parameter = original_register_parameter
            self.gpu_layer.register_buffer = original_register_buffer
            parameter_cls.__new__ = original_parameter_new

        self.gpu_layer._kt_force_global_loading = True

        # Ensure parameters on device
        for param in self.gpu_layer.parameters():
            if param.device != target_device:
                param.data = param.data.to(target_device)
            setattr(param, "_sglang_require_global_experts", True)

        for name, buffer in list(self.gpu_layer.named_buffers()):
            if buffer.device != target_device:
                self.gpu_layer._buffers[name] = buffer.to(target_device)

        # Create runner config
        self.runner_config = replace(
            moe_runner_config, num_local_experts=global_num_experts
        )
        self.gpu_layer.moe_runner_config = self.runner_config
        self.gpu_method.create_moe_runner(self.gpu_layer, self.runner_config)

    def _create_cpu_buffers(self):
        """Create CPU buffers in POSIX shared memory and register as pinned memory.

        Uses multiprocessing.shared_memory to create named shared memory regions
        that can be accessed by all ranks. Then registers the shared memory with
        CUDA using cudaHostRegister to enable fast DMA transfers to GPU.

        This approach gives us both:
        - Cross-process sharing (via POSIX shared memory)
        - Fast GPU transfers (via CUDA pinned memory registration)

        Data flow: CPU wrapper -> shared memory (pinned) -> GPU (DMA)
        """
        self.cpu_buffers = {}  # Shared memory buffers (pinned for fast GPU access)
        self.shm_handles: Dict[str, shared_memory.SharedMemory] = {}

        # Create CPU buffers for each weight tensor
        weight_mappings = {
            "w13_weight_packed": self.gpu_layer.w13_weight_packed,
            "w13_weight_scale": self.gpu_layer.w13_weight_scale,
            "w2_weight_packed": self.gpu_layer.w2_weight_packed,
            "w2_weight_scale": self.gpu_layer.w2_weight_scale,
        }

        tp_rank = get_tensor_model_parallel_rank()

        # Synchronize before creating shared memory to avoid race conditions
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        for name, gpu_tensor in weight_mappings.items():
            # Calculate buffer size in bytes
            nbytes = gpu_tensor.numel() * gpu_tensor.element_size()

            # Create unique shared memory name for this rank and buffer
            shm_name = f"kt_buf_{name}_r{tp_rank}"

            # Try to unlink existing shared memory with the same name (cleanup from previous runs)
            try:
                old_shm = shared_memory.SharedMemory(name=shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass

            # Create new shared memory region
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
            self.shm_handles[name] = shm

            # Create torch tensor with correct dtype by viewing the raw bytes
            # torch.frombuffer creates a tensor sharing memory with the buffer
            cpu_buffer = torch.frombuffer(shm.buf, dtype=gpu_tensor.dtype).reshape(gpu_tensor.shape)

            # Register the shared memory with CUDA as pinned memory for fast DMA
            # cudaHostRegister allows pinning externally allocated memory
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaHostRegister(
                    cpu_buffer.data_ptr(), nbytes, 0  # 0 = cudaHostRegisterDefault
                )

            self.cpu_buffers[name] = cpu_buffer

            logger.info(
                "Rank %d: Created shared memory '%s' (pinned), shape=%s, dtype=%s, size=%d bytes",
                tp_rank, shm_name, gpu_tensor.shape, gpu_tensor.dtype, nbytes
            )

        # Synchronize after all ranks have created their shared memory
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        # Now collect buffer pointers from all ranks
        # Rank 0 will open all other ranks' shared memory regions
        self.all_rank_buffer_ptrs = self._collect_all_rank_buffer_pointers()

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks.

        Rank 0 opens all other ranks' shared memory regions and gets their
        memory addresses. Other ranks only need their own pointers.

        Returns:
            Dictionary mapping buffer names to lists of pointers from all ranks.
        """
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()

        buffer_names = ["w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale"]

        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}

        # Store references to opened shared memory to prevent garbage collection
        if not hasattr(self, '_opened_shm_refs'):
            self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}

        for rank in range(tp_world_size):
            for name in buffer_names:
                if rank == tp_rank:
                    # Use our own buffer pointer
                    ptr = self.cpu_buffers[name].data_ptr()
                else:
                    # Rank 0 (or any rank that needs it) opens other ranks' shared memory
                    if tp_rank == 0:
                        shm_name = f"kt_buf_{name}_r{rank}"
                        try:
                            # Open existing shared memory created by other rank
                            shm = shared_memory.SharedMemory(name=shm_name)
                            self._opened_shm_refs[f"{name}_r{rank}"] = shm
                            # Get the buffer address using ctypes
                            ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                        except FileNotFoundError:
                            logger.error("Rank %d: Failed to open shared memory '%s'", tp_rank, shm_name)
                            ptr = 0
                    else:
                        # Other ranks don't need pointers to other ranks' buffers
                        ptr = 0

                all_rank_ptrs[name].append(ptr)

        logger.info(
            "Rank %d: Collected buffer pointers from %d ranks",
            tp_rank, tp_world_size
        )

        return all_rank_ptrs

    def cleanup_shared_memory(self):
        """Clean up shared memory resources."""
        tp_rank = get_tensor_model_parallel_rank()

        # Unregister pinned memory first
        if torch.cuda.is_available():
            for name, cpu_buffer in self.cpu_buffers.items():
                try:
                    torch.cuda.cudart().cudaHostUnregister(cpu_buffer.data_ptr())
                except Exception as e:
                    logger.warning("Rank %d: Failed to unregister pinned memory '%s': %s", tp_rank, name, e)

        # Close opened references
        if hasattr(self, '_opened_shm_refs'):
            for key, shm in self._opened_shm_refs.items():
                try:
                    shm.close()
                except Exception as e:
                    logger.warning("Rank %d: Failed to close shared memory ref '%s': %s", tp_rank, key, e)
            self._opened_shm_refs.clear()

        # Close and unlink our own shared memory
        for name, shm in self.shm_handles.items():
            try:
                shm.close()
                shm.unlink()
                logger.info("Rank %d: Unlinked shared memory for '%s'", tp_rank, name)
            except Exception as e:
                logger.warning("Rank %d: Failed to unlink shared memory '%s': %s", tp_rank, name, e)

    def _copy_cpu_to_gpu_buffers(self):
        """Copy data from shared memory (pinned) directly to GPU.

        Since the shared memory is registered with CUDA via cudaHostRegister,
        it can be transferred to GPU using fast DMA, no intermediate copy needed.
        """
        weight_mappings = {
            "w13_weight_packed": self.gpu_layer.w13_weight_packed,
            "w13_weight_scale": self.gpu_layer.w13_weight_scale,
            "w2_weight_packed": self.gpu_layer.w2_weight_packed,
            "w2_weight_scale": self.gpu_layer.w2_weight_scale,
        }

        for name, gpu_tensor in weight_mappings.items():
            cpu_buffer = self.cpu_buffers[name]  # Shared memory buffer (pinned)

            # Direct DMA transfer from pinned shared memory to GPU
            gpu_tensor.copy_(cpu_buffer, non_blocking=True)
            tmp = gpu_tensor.reshape(gpu_tensor.size(0), gpu_tensor.size(2), gpu_tensor.size(1)).transpose(1,2).contiguous()
            gpu_tensor.copy_(tmp, non_blocking=True)

    def load(self, layer_idx, wrapper, original_layer):
        # Restore original tensors to ensure correct shape/layout for loading
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        # Ensure previous CUDA work is finished before starting the timer so
        # we measure the actual GPU work performed by the loader.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        gpu_tp_count = get_tensor_model_parallel_world_size()
        gpu_experts_num = self.gpu_layer.num_experts

        t0 = time.perf_counter()

        # Only rank 0 submits the write task (it has the wrapper)
        # Use pre-collected pointers from all ranks (collected once in _create_cpu_buffers)
        # Rank 0 passes pointers to ALL ranks' shared memory buffers
        if tp_rank == 0 and wrapper is not None:
            wrapper.submit_write_weight_scale_to_buffer(
                gpu_tp_count,
                gpu_experts_num,
                self.all_rank_buffer_ptrs["w13_weight_packed"],
                self.all_rank_buffer_ptrs["w13_weight_scale"],
                self.all_rank_buffer_ptrs["w2_weight_packed"],
                self.all_rank_buffer_ptrs["w2_weight_scale"],
            )
            wrapper.sync_write_weight_scale_to_buffer()

        # Barrier to ensure all ranks wait for rank 0 to finish writing
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        write_time = (time.perf_counter() - t0) * 1000.0

        # Each rank copies its own CPU buffer to GPU
        t1 = time.perf_counter()
        self._copy_cpu_to_gpu_buffers()

        # Wait for async CPU-to-GPU copy to complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        copy_time = (time.perf_counter() - t1) * 1000.0

        # Same for processing weights: synchronize before and after to get
        # an accurate measure of GPU work time.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = time.perf_counter()
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(self.gpu_layer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        process_time = (time.perf_counter() - t2) * 1000.0

        logger.info(
            "KT fallback: layer %d rank %d write=%.2f ms copy=%.2f ms process=%.2f ms",
            layer_idx,
            tp_rank,
            write_time,
            copy_time,
            process_time,
        )

        return _FullMoEContext(
            layer=self.gpu_layer,
            method=self.gpu_method,
            runner_config=self.runner_config,
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

    # Try to get num_layers from model config
    hf_config = server_args.get_hf_config()
    num_layers = getattr(hf_config, "num_hidden_layers", None)

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
        prefill_token_threshold=server_args.kt_prefill_token_threshold,
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
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids

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
        method_factory: Optional[Callable[[torch.nn.Module], FusedMoEMethodBase]] = None,
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
        self._method_factory = method_factory
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_tensor_model_parallel_rank()

        # Prefill token threshold for dynamic full-layer loading
        self.prefill_token_threshold = kt_config.prefill_token_threshold or 0

        # Full expert execution context (lazy initialisation)
        self._full_init_args = None
        self._full_extra_weight_attrs = {}

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

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
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self._full_init_args = (
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
        )
        self._full_extra_weight_attrs = dict(extra_weight_attrs)

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

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
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

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        num_tokens = int(x.shape[0]) if x.dim() > 0 else 0

        # Check for full GPU fallback
        if self.prefill_token_threshold > 0 and num_tokens >= self.prefill_token_threshold:
            context = self._build_full_context(layer)
            
            layer_idx = self.kt_config.layer_idx
            # Synchronize before starting the timer so the measured interval
            # only includes work for this forward call.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            forward_start = time.perf_counter()
            try:
                result = context.method.apply(context.layer, dispatch_output)
                return result
            finally:
                # Ensure all CUDA work for the forward has completed before
                # computing elapsed time.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                forward_elapsed = (time.perf_counter() - forward_start) * 1000.0
                logger.info(
                    "KT profile: layer %d fallback forward=%.2f ms",
                    layer_idx,
                    forward_elapsed,
                )
                teardown_start = time.perf_counter()
                self._teardown_context(context)
                logger.info(
                    "KT profile: layer %d fallback teardown=%.2f ms",
                    layer_idx,
                    (time.perf_counter() - teardown_start) * 1000.0,
                )

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
            cpu_output = self.sync(x)
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

    def _build_full_context(
        self, layer: torch.nn.Module
    ) -> _FullMoEContext:
        global _SHARED_FULL_CONTEXT

        if _SHARED_FULL_CONTEXT is None:
            _SHARED_FULL_CONTEXT = SharedFullContext(
                layer=layer,
                method_factory=self._method_factory,
                init_args=self._full_init_args,
                extra_weight_attrs=self._full_extra_weight_attrs,
                global_num_experts=self.global_num_experts,
                moe_runner_config=self.moe_runner_config,
            )

        return _SHARED_FULL_CONTEXT.load(
            layer_idx=self.kt_config.layer_idx,
            wrapper=self.wrapper,
            original_layer=layer,
        )

    def _teardown_context(self, context: _FullMoEContext) -> None:
        """Release the full-GPU context so cached experts free their VRAM."""

        layer_idx = self.kt_config.layer_idx
        teardown_total_start = time.perf_counter()

        # With SharedFullContext, we do not destroy the layer or method.
        # We just clear the reference in the context object.
        context.layer = None
        context.method = None
        context.runner_config = None

        del context

        logger.info(
            "KT profile: layer %d teardown DONE total=%.2f ms",
            layer_idx,
            (time.perf_counter() - teardown_total_start) * 1000.0,
        )
