# SPDX-License-Identifier: Apache-2.0
"""
KT Full Context management for GPU fallback.

This module provides the SharedFullContext class that manages a full GPU layer
for prefill token threshold fallback in the KT expert parallelism wrapper.
"""

import copy
import ctypes
import logging
import os
import time
import uuid
from dataclasses import replace
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel import gptq_marlin_repack

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)

# Global shared instances
_SHARED_FULL_CONTEXT = None
_SHARED_STAGING_BUFFER = None  # Global shared staging buffer for all MoE layers


class SharedStagingBuffer:
    """Global shared staging buffer for CPU expert input across all MoE layers.

    This avoids allocating a separate staging buffer per layer, which would
    consume significant GPU memory (chunked_prefill_size * hidden_size * N_layers).
    Instead, all layers share a single buffer since MoE layers are processed
    sequentially, not in parallel.
    """

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.buffer = torch.empty(
            (max_tokens, hidden_size),
            dtype=dtype,
            device=device,
        )
        buffer_size_mb = self.buffer.numel() * self.buffer.element_size() / 1024**2
        logger.info(
            f"[KT] Created shared staging buffer: {buffer_size_mb:.1f} MiB "
            f"(shape={self.buffer.shape}, dtype={dtype})"
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        """Get a slice of the buffer for the given number of tokens."""
        assert num_tokens <= self.max_tokens, (
            f"Batch size {num_tokens} exceeds staging buffer max size {self.max_tokens}"
        )
        return self.buffer[:num_tokens]


def get_or_create_shared_staging_buffer(
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    """Get or create the global shared staging buffer."""
    global _SHARED_STAGING_BUFFER
    if _SHARED_STAGING_BUFFER is None:
        _SHARED_STAGING_BUFFER = SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
    return _SHARED_STAGING_BUFFER


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
        # Set routed_scaling_factor=None to avoid double scaling:
        # - moe_sum_reduce would apply routed_scaling_factor internally
        # - deepseek_v2.py forward_normal also applies routed_scaling_factor for KTEPWrapperMethod
        # By setting it to None here, we ensure it's only applied once in forward_normal
        runner_config = replace(
            moe_runner_config,
            num_experts=global_num_experts,
            num_local_experts=global_num_experts,
            routed_scaling_factor=None,
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

        # Setup CUDA IPC handles for cross-rank GPU access (V2 batch API)
        self._setup_gpu_ipc_handles()

        # Mark batch load setup as pending (will be done on first use)

        self._batch_load_stream = None

    def _collect_all_rank_buffer_pointers(self) -> Dict[str, List[int]]:
        """Collect CPU buffer pointers from all ranks.

        On rank 0, also registers remote shared memory buffers with cudaHostRegister
        so that cudaMemcpyAsync from these buffers is truly asynchronous.
        """
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        buffer_names = list(self.cpu_buffers.keys())
        all_rank_ptrs: Dict[str, List[int]] = {name: [] for name in buffer_names}
        self._opened_shm_refs: Dict[str, shared_memory.SharedMemory] = {}
        self._registered_remote_bufs: List[tuple] = []  # (ptr, size) for cudaHostUnregister

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
                        # Pin remote buffer in rank 0's CUDA context for async DMA
                        buf_nbytes = self.cpu_buffers[name].numel() * self.cpu_buffers[name].element_size()
                        if torch.cuda.is_available():
                            torch.cuda.cudart().cudaHostRegister(
                                ptr, buf_nbytes, 0
                            )
                            self._registered_remote_bufs.append((ptr, buf_nbytes))
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

    def _setup_gpu_ipc_handles(self):
        """Set up CUDA IPC handles to allow Rank 0 to access all ranks' GPU buffers.

        Uses kt_kernel C++ bindings for cudaIpcGetMemHandle/Open/Close.
        Falls back to PyTorch's _share_cuda_() if direct IPC fails (e.g. when
        expandable_segments is enabled and memory is not from cudaMalloc).
        """
        import kt_kernel
        kt_ext = kt_kernel.kt_kernel_ext

        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        weight_names = self.weight_names
        local_device = torch.cuda.current_device()

        # Step 1: Each rank exports IPC handles via _share_cuda_()
        # Must use _share_cuda_() because PyTorch's caching allocator sub-allocates
        # from large cudaMalloc blocks. cudaIpcGetMemHandle on a sub-pointer returns
        # the containing block's handle, causing multiple tensors to share the same
        # IPC base address. _share_cuda_() correctly provides the offset within the block.
        local_info = {}
        for name in weight_names:
            tensor = getattr(self.gpu_layer, name)
            share_data = tensor.untyped_storage()._share_cuda_()
            local_info[name] = {
                "handle": share_data[1],
                "offset": share_data[3],
            }
        local_info["__device_id__"] = local_device

        # Step 2: Gather all handles to all ranks
        all_info_list = [None] * tp_world_size
        if dist.is_initialized():
            dist.all_gather_object(all_info_list, local_info, group=get_tp_group().cpu_group)
        else:
            all_info_list[0] = local_info

        # Step 3: Rank 0 opens all other ranks' handles
        # Note: multiple tensors may share the same cudaMalloc block (same handle).
        # We must only open each unique handle once per rank to avoid CUDA errors.
        self.ipc_gpu_ptrs: Dict[str, List[int]] = {name: [] for name in weight_names}
        self._ipc_opened_ptrs: List[int] = []
        # Cache: (rank, handle_bytes) -> opened base ptr
        _opened_cache: Dict[tuple, int] = {}

        for name in weight_names:
            ptrs = []
            for rank in range(tp_world_size):
                if rank == tp_rank:
                    ptrs.append(getattr(self.gpu_layer, name).data_ptr())
                elif tp_rank == 0:
                    info = all_info_list[rank].get(name)
                    device_id = all_info_list[rank]["__device_id__"]
                    if info is not None:
                        raw_handle = info["handle"]
                        offset = info["offset"]
                        # _share_cuda_() returns 66 bytes: 2-byte header + 64-byte cudaIpcMemHandle_t
                        handle_bytes = raw_handle[2:66]

                        cache_key = (rank, handle_bytes)
                        if cache_key in _opened_cache:
                            base_ptr = _opened_cache[cache_key]
                        else:
                            base_ptr = kt_ext.cuda_ipc_open_mem_handle(handle_bytes, device_id)
                            _opened_cache[cache_key] = base_ptr
                            self._ipc_opened_ptrs.append(base_ptr)

                        actual_ptr = base_ptr + offset
                        ptrs.append(actual_ptr)
                        logger.info(f"Rank 0: IPC opened {name} rank {rank}: "
                                    f"base=0x{base_ptr:x}, offset={offset}, ptr=0x{actual_ptr:x}")
                    else:
                        ptrs.append(0)
                else:
                    ptrs.append(0)
            self.ipc_gpu_ptrs[name] = ptrs

        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        if tp_rank == 0:
            logger.info(f"[KT] GPU IPC handles setup complete for {len(weight_names)} weights, {tp_world_size} ranks")

    def _cleanup_gpu_ipc_handles(self):
        """Close opened CUDA IPC handles and unregister pinned remote buffers."""
        # Unregister remote shared memory buffers
        if torch.cuda.is_available():
            for ptr, _ in getattr(self, '_registered_remote_bufs', []):
                try:
                    torch.cuda.cudart().cudaHostUnregister(ptr)
                except Exception:
                    pass
        self._registered_remote_bufs = []

        # Close IPC handles
        try:
            import kt_kernel
            kt_ext = kt_kernel.kt_kernel_ext
            for ptr in getattr(self, '_ipc_opened_ptrs', []):
                if ptr != 0:
                    kt_ext.cuda_ipc_close_mem_handle(ptr)
        except Exception:
            pass
        self._ipc_opened_ptrs = []

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

    def _prepare_weight_fp8(self, wrapper, original_layer=None,
                            gpu_expert_ids_t=None, cpu_expert_ids_t=None, gpu_indices_t=None):
        """Prepare FP8 block quant weights by writing from KT and copying to GPU.

        Uses V2 Batch API: Rank 0 handles all CPU expert transfers via CUDA IPC,
        other ranks wait at barrier.

        For GPU experts, all ranks copy independently (fast GPU-to-GPU).

        Args:
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts
            gpu_expert_ids_t: Precomputed GPU expert IDs tensor [N_gpu], device=cuda
            cpu_expert_ids_t: Precomputed CPU expert IDs tensor [N_cpu], device=cuda
            gpu_indices_t: Precomputed GPU indices tensor [N_gpu], maps to src indices
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
            weight_infos.append((name, cpu_buf, gpu_t))

        # Use precomputed tensors or fallback to all-CPU mode
        if gpu_expert_ids_t is not None and original_layer is not None:
            has_gpu_experts = len(gpu_expert_ids_t) > 0
            has_cpu_experts = cpu_expert_ids_t is not None and len(cpu_expert_ids_t) > 0
        else:
            has_gpu_experts = False
            has_cpu_experts = True
            cpu_expert_ids_t = torch.arange(num_experts, device=device, dtype=torch.long)

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU, batch operation) ---
        if has_gpu_experts:
            for name, _, dst in weight_infos:
                src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                # Batch copy: dst[gpu_expert_ids] = src[gpu_indices]
                dst[gpu_expert_ids_t] = src[gpu_indices_t]

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not has_cpu_experts:
            # All experts are on GPU, nothing more to do
            return

        # Convert tensor to list for Python iteration (only done once)
        cpu_expert_ids = cpu_expert_ids_t.tolist()

        tp_world_size = get_tensor_model_parallel_world_size()

        # --- Batch API: Rank 0 handles all CPU expert transfers via CUDA IPC ---
        if tp_rank == 0 and wrapper is not None:
            self._prepare_weight_fp8_batch_api(
                wrapper, weight_infos, cpu_expert_ids, device, tp_world_size
            )
        # All ranks wait for Rank 0 to finish
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().cpu_group)

    def _setup_batch_load_buffers(self, wrapper, device, tp_world_size):
        """V2 API: One-time setup of batch load buffer pointers.

        Registers all fixed buffer pointers (pinned CPU buffers, GPU IPC pointers,
        CUDA stream) in the C++ MoE instance. This is called once per layer, and
        subsequent calls to submit_batch_load_cpu_experts_to_gpu() only need cpu_expert_ids.

        Args:
            wrapper: KT wrapper with setup_batch_load_buffers API
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        # Calculate per-expert byte sizes for buffer pointer computation
        w13_weight_buf = self.cpu_buffers["w13_weight"]
        w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
        w2_weight_buf = self.cpu_buffers["w2_weight"]
        w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

        w13_weight_expert_nbytes = w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
        w13_scale_expert_nbytes = w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
        w2_weight_expert_nbytes = w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
        w2_scale_expert_nbytes = w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()

        # Collect pinned buffer pointers for double buffering (2 slots × tp_world_size ranks)
        # Each slot needs tp_world_size pointers because write_weight_scale_to_buffer
        # splits the expert across TP shards and writes each shard to a different buffer.
        # Layout: [slot0_rank0, slot0_rank1, ..., slot1_rank0, slot1_rank1, ...]
        pinned_w13_weight_ptrs = []
        pinned_w13_scale_ptrs = []
        pinned_w2_weight_ptrs = []
        pinned_w2_scale_ptrs = []
        for slot in range(2):
            for ptr in self.all_rank_buffer_ptrs["w13_weight"]:
                pinned_w13_weight_ptrs.append(ptr + slot * w13_weight_expert_nbytes)
            for ptr in self.all_rank_buffer_ptrs["w13_weight_scale_inv"]:
                pinned_w13_scale_ptrs.append(ptr + slot * w13_scale_expert_nbytes)
            for ptr in self.all_rank_buffer_ptrs["w2_weight"]:
                pinned_w2_weight_ptrs.append(ptr + slot * w2_weight_expert_nbytes)
            for ptr in self.all_rank_buffer_ptrs["w2_weight_scale_inv"]:
                pinned_w2_scale_ptrs.append(ptr + slot * w2_scale_expert_nbytes)

        # Use IPC GPU pointers for cross-rank access (V2 API)
        # Rank 0 can directly write to all ranks' GPUs via IPC
        gpu_w13_weight_ptrs_per_rank = self.ipc_gpu_ptrs["w13_weight"]
        gpu_w13_scale_ptrs_per_rank = self.ipc_gpu_ptrs["w13_weight_scale_inv"]
        gpu_w2_weight_ptrs_per_rank = self.ipc_gpu_ptrs["w2_weight"]
        gpu_w2_scale_ptrs_per_rank = self.ipc_gpu_ptrs["w2_weight_scale_inv"]

        # Create per-rank CUDA streams for parallel H2D transfers
        # Each rank's memcpy goes on a separate stream so transfers to different GPUs overlap
        self._batch_load_streams = [torch.cuda.Stream(device=device) for _ in range(tp_world_size)]
        # Keep first stream as the "primary" for backward compat
        self._batch_load_stream = self._batch_load_streams[0]

        # Register all buffer pointers in the C++ MoE instance
        wrapper.setup_batch_load_buffers(
            gpu_tp_count=tp_world_size,
            pinned_w13_weight_ptrs=pinned_w13_weight_ptrs,
            pinned_w13_scale_ptrs=pinned_w13_scale_ptrs,
            pinned_w2_weight_ptrs=pinned_w2_weight_ptrs,
            pinned_w2_scale_ptrs=pinned_w2_scale_ptrs,
            gpu_w13_weight_ptrs_per_rank=gpu_w13_weight_ptrs_per_rank,
            gpu_w13_scale_ptrs_per_rank=gpu_w13_scale_ptrs_per_rank,
            gpu_w2_weight_ptrs_per_rank=gpu_w2_weight_ptrs_per_rank,
            gpu_w2_scale_ptrs_per_rank=gpu_w2_scale_ptrs_per_rank,
            cuda_streams=[s.cuda_stream for s in self._batch_load_streams],
            w13_weight_expert_nbytes=w13_weight_expert_nbytes,
            w13_scale_expert_nbytes=w13_scale_expert_nbytes,
            w2_weight_expert_nbytes=w2_weight_expert_nbytes,
            w2_scale_expert_nbytes=w2_scale_expert_nbytes,
        )

        tp_rank = get_tensor_model_parallel_rank()
        if tp_rank == 0:
            logger.info("[KT] V2 batch load buffers setup complete")

    def _prepare_weight_fp8_batch_api(self, wrapper, weight_infos, cpu_expert_ids, device, tp_world_size):
        """V2 Batch API implementation for FP8 CPU expert weight loading.

        Offloads the entire Phase 2 (CPU expert transfer) to cpuinfer process:
        - Eliminates Python loop overhead
        - Eliminates per-expert dist.barrier() calls
        - cpuinfer handles write(e+1) || copy(e) pipeline internally
        - Uses CUDA IPC to write directly to all ranks' GPUs from Rank 0

        Args:
            wrapper: KT wrapper with batch_load_cpu_experts_to_gpu API
            weight_infos: List of (name, cpu_buf, gpu_t) tuples
            cpu_expert_ids: List of CPU expert IDs to load
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        # V2 API: Setup buffers for this wrapper's C++ MoE instance
        # Setup batch load buffers once per C++ moe instance (different layers have different wrappers)
        if not hasattr(self, '_batch_load_configured_wrappers'):
            self._batch_load_configured_wrappers = set()
        wrapper_id = id(wrapper.moe)
        if wrapper_id not in self._batch_load_configured_wrappers:
            self._setup_batch_load_buffers(wrapper, device, tp_world_size)
            self._batch_load_configured_wrappers.add(wrapper_id)

        # V2 API: Submit task with only cpu_expert_ids (all buffers pre-registered)
        wrapper.submit_batch_load_cpu_experts_to_gpu(cpu_expert_ids)

        # Wait for completion
        wrapper.sync_batch_load_cpu_experts_to_gpu()

        # Synchronize all per-rank batch streams with the default stream
        default_stream = torch.cuda.current_stream(device)
        for s in self._batch_load_streams:
            default_stream.wait_stream(s)

    # ==================== V3 Polling-Based Batch Load API ====================

    def _create_polling_sync_slots(self):
        """Create polling sync slots in shared memory for each rank.

        Each rank creates a 64-byte sync slot in POSIX shared memory.
        These slots are used for CPU-GPU synchronization in polling-based batch load.
        """
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()

        # Create sync slot in shared memory (64 bytes, cache-line aligned)
        sync_slot_size = 64
        shm_name = f"kt_sync_slot_r{tp_rank}_{self.shm_unique_id}"
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sync_slot_size)
        self._polling_sync_slot_shm = shm

        # Initialize sync slot to IDLE state
        sync_slot_buf = ctypes.c_char.from_buffer(shm.buf)
        sync_slot_ptr = ctypes.addressof(sync_slot_buf)
        # Zero-initialize the slot (signal=0=IDLE)
        ctypes.memset(sync_slot_ptr, 0, sync_slot_size)

        # Register as pinned memory for GPU polling
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaHostRegister(sync_slot_ptr, sync_slot_size, 0)

        self._polling_local_sync_slot_ptr = sync_slot_ptr

        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)

        # Collect all ranks' sync slot pointers (rank 0 opens remote slots)
        self._polling_all_sync_slot_ptrs = self._collect_polling_sync_slot_pointers()

        # Unlink shared memory (remains accessible via mmap)
        if dist.is_initialized():
            dist.barrier(group=get_tp_group().device_group)
        shm.unlink()

    def _collect_polling_sync_slot_pointers(self) -> List[int]:
        """Collect sync slot pointers from all ranks.

        On rank 0, also opens and registers remote sync slots.
        """
        tp_rank = get_tensor_model_parallel_rank()
        tp_world_size = get_tensor_model_parallel_world_size()
        sync_slot_size = 64

        all_ptrs = []
        self._polling_opened_sync_slot_refs = []
        self._polling_registered_sync_slot_ptrs = []

        for rank in range(tp_world_size):
            if rank == tp_rank:
                all_ptrs.append(self._polling_local_sync_slot_ptr)
            elif tp_rank == 0:
                shm_name = f"kt_sync_slot_r{rank}_{self.shm_unique_id}"
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    self._polling_opened_sync_slot_refs.append(shm)
                    ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
                    # Register remote sync slot as pinned for GPU visibility
                    if torch.cuda.is_available():
                        torch.cuda.cudart().cudaHostRegister(ptr, sync_slot_size, 0)
                        self._polling_registered_sync_slot_ptrs.append(ptr)
                    all_ptrs.append(ptr)
                except FileNotFoundError:
                    logger.error(f"Rank 0: Failed to open sync slot for rank {rank}")
                    all_ptrs.append(0)
            else:
                all_ptrs.append(0)

        return all_ptrs

    def _setup_polling_batch_load(self, wrapper, device, tp_world_size):
        """V3 API: Setup polling-based batch load with persistent kernels.

        Each rank launches a persistent polling kernel that:
        1. Polls sync slot for DATA_READY signal
        2. Copies from pinned CPU buffer to local GPU memory
        3. Signals GPU_DONE

        Args:
            wrapper: KT wrapper with polling batch load API
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        tp_rank = get_tensor_model_parallel_rank()

        # Calculate per-expert byte sizes
        w13_weight_buf = self.cpu_buffers["w13_weight"]
        w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
        w2_weight_buf = self.cpu_buffers["w2_weight"]
        w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

        w13_weight_expert_nbytes = w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
        w13_scale_expert_nbytes = w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
        w2_weight_expert_nbytes = w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
        w2_scale_expert_nbytes = w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()

        # Build source buffer pointers - each rank only uses its own local buffers
        # Format: [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
        # Note: For the local rank's kernel, we use local buffer pointers
        # For rank 0's batch_load function (which signals), it uses all_rank_buffer_ptrs

        # Build source buffer pointers for all ranks
        # Rank 0 has valid pointers for all ranks (via shared memory)
        # Other ranks only have valid pointers for themselves
        src_buffer_ptrs_per_rank = []
        for rank in range(tp_world_size):
            if rank == tp_rank:
                # Use local buffer pointers
                w13_w_base = self.cpu_buffers["w13_weight"].data_ptr()
                w13_s_base = self.cpu_buffers["w13_weight_scale_inv"].data_ptr()
                w2_w_base = self.cpu_buffers["w2_weight"].data_ptr()
                w2_s_base = self.cpu_buffers["w2_weight_scale_inv"].data_ptr()
            elif tp_rank == 0:
                # Rank 0 has access to all ranks' shared memory
                w13_w_base = self.all_rank_buffer_ptrs["w13_weight"][rank]
                w13_s_base = self.all_rank_buffer_ptrs["w13_weight_scale_inv"][rank]
                w2_w_base = self.all_rank_buffer_ptrs["w2_weight"][rank]
                w2_s_base = self.all_rank_buffer_ptrs["w2_weight_scale_inv"][rank]
            else:
                # Non-rank-0 ranks don't have access to other ranks' buffers
                w13_w_base = 0
                w13_s_base = 0
                w2_w_base = 0
                w2_s_base = 0

            if w13_w_base != 0:
                rank_buffers = [
                    w13_w_base,  # slot 0
                    w13_w_base + w13_weight_expert_nbytes,  # slot 1
                    w13_s_base,  # slot 0
                    w13_s_base + w13_scale_expert_nbytes,  # slot 1
                    w2_w_base,  # slot 0
                    w2_w_base + w2_weight_expert_nbytes,  # slot 1
                    w2_s_base,  # slot 0
                    w2_s_base + w2_scale_expert_nbytes,  # slot 1
                ]
            else:
                rank_buffers = [0] * 8
            src_buffer_ptrs_per_rank.append(rank_buffers)

        # Get GPU destination pointers - each rank only has its own
        dst_w13_weight_per_rank = []
        dst_w13_scale_per_rank = []
        dst_w2_weight_per_rank = []
        dst_w2_scale_per_rank = []
        for rank in range(tp_world_size):
            if rank == tp_rank:
                dst_w13_weight_per_rank.append(getattr(self.gpu_layer, "w13_weight").data_ptr())
                dst_w13_scale_per_rank.append(getattr(self.gpu_layer, "w13_weight_scale_inv").data_ptr())
                dst_w2_weight_per_rank.append(getattr(self.gpu_layer, "w2_weight").data_ptr())
                dst_w2_scale_per_rank.append(getattr(self.gpu_layer, "w2_weight_scale_inv").data_ptr())
            else:
                # Other ranks' GPU pointers are not accessible without IPC
                dst_w13_weight_per_rank.append(0)
                dst_w13_scale_per_rank.append(0)
                dst_w2_weight_per_rank.append(0)
                dst_w2_scale_per_rank.append(0)

        # Create CUDA stream for this rank's persistent kernel
        self._polling_stream = torch.cuda.Stream(device=device)
        # Build stream list (only this rank's stream is valid)
        stream_ptrs = [0] * tp_world_size
        stream_ptrs[tp_rank] = self._polling_stream.cuda_stream

        # Setup polling batch load in C++
        wrapper.setup_polling_batch_load(
            num_ranks=tp_world_size,
            sync_slot_ptrs=self._polling_all_sync_slot_ptrs,
            src_buffer_ptrs_per_rank=src_buffer_ptrs_per_rank,
            dst_w13_weight_per_rank=dst_w13_weight_per_rank,
            dst_w13_scale_per_rank=dst_w13_scale_per_rank,
            dst_w2_weight_per_rank=dst_w2_weight_per_rank,
            dst_w2_scale_per_rank=dst_w2_scale_per_rank,
            stream_ptrs=stream_ptrs,
            w13_weight_size=w13_weight_expert_nbytes,
            w13_scale_size=w13_scale_expert_nbytes,
            w2_weight_size=w2_weight_expert_nbytes,
            w2_scale_size=w2_scale_expert_nbytes,
        )

        # Launch persistent polling kernel for this rank
        wrapper.launch_polling_kernel(local_rank=tp_rank, total_experts=-1)

        logger.info(f"[KT] Rank {tp_rank}: V3 polling batch load setup complete")

    def _prepare_weight_fp8_polling_api(self, wrapper, weight_infos, cpu_expert_ids, device, tp_world_size):
        """V3 Polling API implementation for FP8 CPU expert weight loading.

        Uses persistent polling kernels launched during setup. Each rank's GPU
        independently polls a shared CPU memory flag and copies data when signaled.

        Args:
            wrapper: KT wrapper with polling batch load API
            weight_infos: List of (name, cpu_buf, gpu_t) tuples
            cpu_expert_ids: List of CPU expert IDs to load
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        tp_rank = get_tensor_model_parallel_rank()

        # Setup polling batch load once per C++ moe instance
        if not hasattr(self, '_polling_configured_wrappers'):
            self._polling_configured_wrappers = set()
        wrapper_id = id(wrapper.moe)
        if wrapper_id not in self._polling_configured_wrappers:
            # Create sync slots if not yet created
            if not hasattr(self, '_polling_all_sync_slot_ptrs'):
                self._create_polling_sync_slots()
            self._setup_polling_batch_load(wrapper, device, tp_world_size)
            self._polling_configured_wrappers.add(wrapper_id)

        # Only rank 0 submits the batch load task (writes to all ranks' buffers and signals)
        if tp_rank == 0:
            wrapper.submit_batch_load_cpu_experts_polling(cpu_expert_ids)
            wrapper.sync_batch_load_cpu_experts_polling()

        # Wait for local polling stream to complete
        default_stream = torch.cuda.current_stream(device)
        default_stream.wait_stream(self._polling_stream)

    def _cleanup_polling_sync_slots(self):
        """Clean up polling sync slots and unregister pinned memory."""
        # Unregister remote sync slots
        if torch.cuda.is_available():
            for ptr in getattr(self, '_polling_registered_sync_slot_ptrs', []):
                try:
                    torch.cuda.cudart().cudaHostUnregister(ptr)
                except Exception:
                    pass

        # Unregister local sync slot
        local_ptr = getattr(self, '_polling_local_sync_slot_ptr', None)
        if local_ptr and torch.cuda.is_available():
            try:
                torch.cuda.cudart().cudaHostUnregister(local_ptr)
            except Exception:
                pass

        # Close opened shared memory refs
        for shm in getattr(self, '_polling_opened_sync_slot_refs', []):
            try:
                shm.close()
            except Exception:
                pass

        # Close local sync slot shm
        shm = getattr(self, '_polling_sync_slot_shm', None)
        if shm:
            try:
                shm.close()
            except Exception:
                pass

    # ==================== V4 Polling Memcpy Worker API ====================

    def _get_worker_cpu_core(self, tp_rank: int, tp_world_size: int) -> int:
        """Get CPU core for polling memcpy worker, avoiding cpuinfer cores.

        Strategy: Use the last N cores on the system, where N = tp_world_size.
        This avoids conflict with cpuinfer threads which typically bind to
        earlier cores on NUMA node 0.

        Args:
            tp_rank: Tensor parallel rank
            tp_world_size: Total number of TP ranks

        Returns:
            CPU core ID to bind the worker to
        """
        num_cpus = os.cpu_count()
        # Use last tp_world_size cores, one per rank
        # Rank 0 gets the last core, Rank 1 gets second to last, etc.
        return num_cpus - 1 - tp_rank

    def _setup_polling_memcpy_worker(self, wrapper, device, tp_world_size):
        """V4 API: Setup polling memcpy worker for the local rank.

        Each rank creates a dedicated C++ worker thread that:
        1. Polls sync slot for DATA_READY signal
        2. Calls cudaMemcpyAsync (DMA, doesn't use GPU SM)
        3. Waits for completion and signals GPU_DONE

        Args:
            wrapper: KT wrapper (used for write_weight_scale_to_buffer)
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        from kt_kernel.utils.amx import (
            create_polling_memcpy_worker,
            start_polling_memcpy_worker,
        )

        tp_rank = get_tensor_model_parallel_rank()

        # Calculate per-expert byte sizes
        w13_weight_buf = self.cpu_buffers["w13_weight"]
        w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
        w2_weight_buf = self.cpu_buffers["w2_weight"]
        w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

        w13_weight_expert_nbytes = w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
        w13_scale_expert_nbytes = w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
        w2_weight_expert_nbytes = w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
        w2_scale_expert_nbytes = w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()

        # Build source buffer pointers for this rank
        # Format: [w13_w_s0, w13_w_s1, w13_s_s0, w13_s_s1, w2_w_s0, w2_w_s1, w2_s_s0, w2_s_s1]
        w13_w_base = self.cpu_buffers["w13_weight"].data_ptr()
        w13_s_base = self.cpu_buffers["w13_weight_scale_inv"].data_ptr()
        w2_w_base = self.cpu_buffers["w2_weight"].data_ptr()
        w2_s_base = self.cpu_buffers["w2_weight_scale_inv"].data_ptr()

        src_buffer_ptrs = [
            w13_w_base,  # slot 0
            w13_w_base + w13_weight_expert_nbytes,  # slot 1
            w13_s_base,  # slot 0
            w13_s_base + w13_scale_expert_nbytes,  # slot 1
            w2_w_base,  # slot 0
            w2_w_base + w2_weight_expert_nbytes,  # slot 1
            w2_s_base,  # slot 0
            w2_s_base + w2_scale_expert_nbytes,  # slot 1
        ]

        # Get GPU destination pointers for this rank
        dst_w13_weight = getattr(self.gpu_layer, "w13_weight").data_ptr()
        dst_w13_scale = getattr(self.gpu_layer, "w13_weight_scale_inv").data_ptr()
        dst_w2_weight = getattr(self.gpu_layer, "w2_weight").data_ptr()
        dst_w2_scale = getattr(self.gpu_layer, "w2_weight_scale_inv").data_ptr()

        # Get CPU core for this rank's worker
        cpu_core = self._get_worker_cpu_core(tp_rank, tp_world_size)
        cuda_device = torch.cuda.current_device()

        # Create and start the worker
        create_polling_memcpy_worker(
            rank=tp_rank,
            cuda_device=cuda_device,
            cpu_core=cpu_core,
            sync_slot_ptr=self._polling_local_sync_slot_ptr,
            src_buffer_ptrs=src_buffer_ptrs,
            dst_w13_weight=dst_w13_weight,
            dst_w13_scale=dst_w13_scale,
            dst_w2_weight=dst_w2_weight,
            dst_w2_scale=dst_w2_scale,
            w13_weight_size=w13_weight_expert_nbytes,
            w13_scale_size=w13_scale_expert_nbytes,
            w2_weight_size=w2_weight_expert_nbytes,
            w2_scale_size=w2_scale_expert_nbytes,
        )
        start_polling_memcpy_worker()

        logger.info(f"[KT] Rank {tp_rank}: V4 polling memcpy worker started on core {cpu_core}")

    def _prepare_weight_fp8_worker_api(self, wrapper, weight_infos, cpu_expert_ids, device, tp_world_size):
        """V4 Worker API implementation for FP8 CPU expert weight loading.

        Uses dedicated polling memcpy workers launched during setup. Each rank's
        worker thread independently polls a shared CPU memory flag and performs
        cudaMemcpyAsync (DMA) when signaled.

        Pipeline: write(e) -> signal(e) -> [GPU copy overlaps] -> write(e+1) -> wait(e) -> ...

        Args:
            wrapper: KT wrapper with write_weight_scale_to_buffer API
            weight_infos: List of (name, cpu_buf, gpu_t) tuples
            cpu_expert_ids: List of CPU expert IDs to load
            device: Target CUDA device
            tp_world_size: Number of tensor parallel ranks
        """
        from kt_kernel.utils.amx import stop_polling_memcpy_worker

        tp_rank = get_tensor_model_parallel_rank()

        # Setup worker once per SharedFullContext (not per C++ moe instance)
        if not hasattr(self, '_polling_worker_started'):
            self._polling_worker_started = False

        if not self._polling_worker_started:
            # Create sync slots if not yet created
            if not hasattr(self, '_polling_all_sync_slot_ptrs'):
                self._create_polling_sync_slots()
            self._setup_polling_memcpy_worker(wrapper, device, tp_world_size)
            self._polling_worker_started = True

        # Calculate per-expert byte sizes for buffer addressing
        w13_weight_buf = self.cpu_buffers["w13_weight"]
        w13_scale_buf = self.cpu_buffers["w13_weight_scale_inv"]
        w2_weight_buf = self.cpu_buffers["w2_weight"]
        w2_scale_buf = self.cpu_buffers["w2_weight_scale_inv"]

        w13_weight_expert_nbytes = w13_weight_buf.numel() // 2 * w13_weight_buf.element_size()
        w13_scale_expert_nbytes = w13_scale_buf.numel() // 2 * w13_scale_buf.element_size()
        w2_weight_expert_nbytes = w2_weight_buf.numel() // 2 * w2_weight_buf.element_size()
        w2_scale_expert_nbytes = w2_scale_buf.numel() // 2 * w2_scale_buf.element_size()

        def submit_write_expert(expert_id, slot):
            """Submit write task for an expert to the given double buffer slot."""
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

        def cpu_signal_data_ready(slot_ptr, expert_id, slot_idx):
            """Signal that data is ready for GPU to copy (matches C++ protocol)."""
            import ctypes
            # BatchSyncSlot: signal(4B), expert_id(4B), slot_idx(4B), ...
            slot_array = (ctypes.c_int32 * 3).from_address(slot_ptr)
            slot_array[1] = expert_id  # expert_id
            slot_array[2] = slot_idx   # slot_idx
            ctypes.memmove(slot_ptr, ctypes.addressof(slot_array), 12)
            # Memory fence then set signal
            slot_array[0] = 1  # SIGNAL_DATA_READY

        def cpu_wait_gpu_done(slot_ptr) -> bool:
            """Wait for GPU to complete copy (matches C++ protocol)."""
            import ctypes
            import time
            slot_array = (ctypes.c_int32 * 1).from_address(slot_ptr)
            while True:
                sig = slot_array[0]
                if sig == 2:  # SIGNAL_GPU_DONE
                    slot_array[0] = 0  # Reset to SIGNAL_IDLE
                    return True
                # Busy-wait (could add sleep for between-prefill mode)
                time.sleep(0)  # Yield

        num_experts = len(cpu_expert_ids)
        do_write = tp_rank == 0 and wrapper is not None

        for idx in range(num_experts):
            expert_id = cpu_expert_ids[idx]
            slot = idx % 2

            # 1. Write current expert to buffer (Rank 0 writes to all ranks' buffers)
            if do_write:
                submit_write_expert(expert_id, slot)
                wrapper.sync_write_weight_scale_to_buffer()

            # Barrier: ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            # 2. Wait for PREVIOUS expert's GPU copy to complete (if not first expert)
            #    This ensures the slot we're about to write to (next iteration) is free
            if idx > 0:
                cpu_wait_gpu_done(self._polling_local_sync_slot_ptr)

            # 3. Signal current expert ready for this rank
            cpu_signal_data_ready(self._polling_local_sync_slot_ptr, expert_id, slot)

            # Pipeline: Now GPU worker starts copying expert[idx] while we loop back

        # 4. Wait for the last expert's GPU copy to complete
        if num_experts > 0:
            cpu_wait_gpu_done(self._polling_local_sync_slot_ptr)

    def _stop_polling_memcpy_worker(self):
        """Stop the polling memcpy worker for the local rank."""
        if not getattr(self, '_polling_worker_started', False):
            return

        try:
            from kt_kernel.utils.amx import stop_polling_memcpy_worker
            stop_polling_memcpy_worker()
            self._polling_worker_started = False
            logger.info("[KT] Polling memcpy worker stopped")
        except Exception as e:
            logger.warning(f"[KT] Failed to stop polling memcpy worker: {e}")

    # NOTE: DeepGemm ue8m0 conversion is not used in KT fallback path.
    # The conversion is handled separately in the normal weight loading path.

    def _prepare_weight_fp8_channel(self, wrapper, original_layer=None,
                                     gpu_expert_ids_t=None, cpu_expert_ids_t=None, gpu_indices_t=None):
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

        Optimization: Uses precomputed index tensors for batch GPU-to-GPU copy.
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
            weight_infos.append((name, cpu_buf, gpu_t))

        # Use precomputed tensors or fallback to all-CPU mode
        if gpu_expert_ids_t is not None and original_layer is not None:
            has_gpu_experts = len(gpu_expert_ids_t) > 0
            has_cpu_experts = cpu_expert_ids_t is not None and len(cpu_expert_ids_t) > 0
        else:
            has_gpu_experts = False
            has_cpu_experts = True
            cpu_expert_ids_t = torch.arange(num_experts, device=device, dtype=torch.long)

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU, batch operation) ---
        if has_gpu_experts:
            for name, _, dst in weight_infos:
                src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                # Batch copy: dst[gpu_expert_ids] = src[gpu_indices]
                dst[gpu_expert_ids_t] = src[gpu_indices_t]

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not has_cpu_experts:
            # All experts are on GPU, nothing more to do
            return

        # Convert tensor to list for Python iteration (only done once)
        cpu_expert_ids = cpu_expert_ids_t.tolist()

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
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

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
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

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def _prepare_weight_bf16(self, wrapper, original_layer=None,
                             gpu_expert_ids_t=None, cpu_expert_ids_t=None, gpu_indices_t=None):
        """Prepare BF16/unquantized weights by writing from KT and copying to GPU.

        Pipeline: write(e+1) || copy(e) || postprocess(e-1)

        BF16/unquantized is similar to FP8 block quant:
        - No transpose needed (weight layout is already correct)
        - No marlin_repack needed (only INT4 Marlin needs this)
        - No permute_scales needed (only Marlin format needs this)
        - No scales at all (unlike FP8 which has scale_inv)

        The postprocess stage is a no-op for BF16 but provides pipeline synchronization
        to ensure copy(e-2) completes before write(e) overwrites the same slot.

        Optimization: Uses precomputed index tensors for batch GPU-to-GPU copy.
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
            weight_infos.append((name, cpu_buf, gpu_t))

        # Use precomputed tensors or fallback to all-CPU mode
        if gpu_expert_ids_t is not None and original_layer is not None:
            has_gpu_experts = len(gpu_expert_ids_t) > 0
            has_cpu_experts = cpu_expert_ids_t is not None and len(cpu_expert_ids_t) > 0
        else:
            has_gpu_experts = False
            has_cpu_experts = True
            cpu_expert_ids_t = torch.arange(num_experts, device=device, dtype=torch.long)

        # --- Phase 1: Copy GPU experts directly (fast GPU-to-GPU, batch operation) ---
        if has_gpu_experts:
            for name, _, dst in weight_infos:
                src = getattr(original_layer, name)  # [num_gpu_experts, ...]
                # Batch copy: dst[gpu_expert_ids] = src[gpu_indices]
                dst[gpu_expert_ids_t] = src[gpu_indices_t]

        # --- Phase 2: Transfer CPU experts via KT pipeline ---
        if not has_cpu_experts:
            # All experts are on GPU, nothing more to do
            return

        # Convert tensor to list for Python iteration (only done once)
        cpu_expert_ids = cpu_expert_ids_t.tolist()

        # Pipeline: write(e+1) || copy(e) || postprocess(e-1)
        copy_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        # Events indexed by position in cpu_expert_ids
        events = [torch.cuda.Event() for _ in range(len(cpu_expert_ids))]

        def postprocess_expert(idx):
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

            def submit_write_expert(expert_id, slot):
                # Use provided slot for double buffering
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

            # Submit first CPU expert ahead of time
            submit_write_expert(cpu_expert_ids[0], 0)

        for idx, e in enumerate(cpu_expert_ids):
            slot = idx % 2  # Double buffering based on iteration index

            # Sync write for expert e, submit write for next CPU expert
            if do_write:
                wrapper.sync_write_weight_scale_to_buffer()
                if idx + 1 < len(cpu_expert_ids):
                    next_slot = (idx + 1) % 2
                    # Before writing to next_slot, ensure copy from that slot is complete.
                    if idx > 0:
                        events[idx - 1].synchronize()
                    submit_write_expert(cpu_expert_ids[idx + 1], next_slot)

            # Barrier to ensure all ranks see the written data
            if dist.is_initialized():
                dist.barrier(group=get_tp_group().device_group)

            with torch.cuda.stream(copy_stream):
                for _, cpu_buf, gpu_t in weight_infos:
                    gpu_t[e].copy_(cpu_buf[slot], non_blocking=True)
                events[idx].record(copy_stream)

            # Postprocess expert idx-1: provides pipeline structure for future extensions
            if idx > 0:
                with torch.cuda.stream(post_stream):
                    post_stream.wait_event(events[idx - 1])
                    postprocess_expert(idx - 1)

        # Process last CPU expert
        if cpu_expert_ids:
            with torch.cuda.stream(post_stream):
                post_stream.wait_event(events[-1])
                postprocess_expert(len(cpu_expert_ids) - 1)

        torch.cuda.current_stream(device).wait_stream(post_stream)

    def load(self, layer_idx, wrapper, original_layer=None,
             gpu_expert_ids_t=None, cpu_expert_ids_t=None, gpu_indices_t=None):
        """Load weights from disk to GPU via shared memory.

        Args:
            layer_idx: Layer index in the model
            wrapper: KT wrapper for CPU expert weight loading
            original_layer: Original MoE layer with GPU experts (optional)
            gpu_expert_ids_t: Precomputed GPU expert IDs tensor [N_gpu], device=cuda
            cpu_expert_ids_t: Precomputed CPU expert IDs tensor [N_cpu], device=cuda
            gpu_indices_t: Precomputed GPU indices tensor [N_gpu], maps to src indices
        """
        for name, param in self.original_params.items():
            setattr(self.gpu_layer, name, param)
        for name, buf in self.original_buffers.items():
            self.gpu_layer.register_buffer(name, buf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tp_rank = get_tensor_model_parallel_rank()
        t0 = time.perf_counter()

        # Select appropriate prepare_weight method based on quantization type
        # FP8/BF16 methods support GPU expert optimization; INT4 uses full CPU pipeline
        if self.is_fp8_quant:
            self._prepare_weight_fp8(wrapper, original_layer,
                                     gpu_expert_ids_t, cpu_expert_ids_t, gpu_indices_t)
        elif self.is_fp8_channel_quant:
            self._prepare_weight_fp8_channel(wrapper, original_layer,
                                             gpu_expert_ids_t, cpu_expert_ids_t, gpu_indices_t)
        elif self.is_bf16_quant:
            self._prepare_weight_bf16(wrapper, original_layer,
                                      gpu_expert_ids_t, cpu_expert_ids_t, gpu_indices_t)
        else:
            # INT4 Marlin format: write(e+1) || copy(e) || postprocess(e-1)
            self._prepare_weight_int4(wrapper)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = (time.perf_counter() - t0) * 1000.0

        if tp_rank == 0:
            logger.info(
                "KT fallback: layer %d weight load time = %.2f ms",
                layer_idx,
                load_time,
            )
