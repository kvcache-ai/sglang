# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
MoE LoRA Memory Pool

This module manages GPU memory buffers for MoE (Mixture of Experts) LoRA weights.
Unlike regular LoRA which has per-layer buffers, MoE LoRA requires per-expert buffers.

Buffer Layout (gate and up are stored separately because they have independent lora_A):
- gate_A_buffer: (max_loras, num_layers, num_experts, max_rank, hidden_size)
- gate_B_buffer: (max_loras, num_layers, num_experts, intermediate_size, max_rank)
- up_A_buffer: (max_loras, num_layers, num_experts, max_rank, hidden_size)
- up_B_buffer: (max_loras, num_layers, num_experts, intermediate_size, max_rank)
- down_A_buffer: (max_loras, num_layers, num_experts, max_rank, intermediate_size)
- down_B_buffer: (max_loras, num_layers, num_experts, hidden_size, max_rank)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

from sglang.srt.distributed import divide
from sglang.srt.lora.lora import LoRAAdapter, MoELoRALayer
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


@dataclass
class MoELoRABatchInfo:
    """
    Batch information for MoE LoRA computation.

    This extends the regular LoRABatchInfo with expert routing information.
    """
    # Basic batch info
    bs: int  # Batch size (number of sequences)
    num_tokens: int  # Total number of tokens

    # Adapter info per sequence
    weight_indices: torch.Tensor  # (bs,) adapter index for each sequence
    lora_ranks: torch.Tensor  # (max_loras,) rank of each adapter
    scalings: torch.Tensor  # (max_loras,) alpha/r scaling for each adapter

    # Expert routing info (set during forward pass)
    # These are populated by the TopK router
    topk_ids: Optional[torch.Tensor] = None  # (num_tokens, top_k)
    topk_weights: Optional[torch.Tensor] = None  # (num_tokens, top_k)


class MoELoRAMemoryPool:
    """
    Memory pool for MoE LoRA weights.

    Manages GPU buffers for per-expert LoRA weights across multiple layers
    and multiple adapters.
    """

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_lora_rank: int,
        device: torch.device,
    ):
        self.base_hf_config = base_hf_config
        self.max_loras_per_batch = max_loras_per_batch
        self.dtype = dtype
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.max_lora_rank = max_lora_rank
        self.device = device

        # Get MoE config from base model config
        self.num_layers = base_hf_config.num_hidden_layers
        self.hidden_size = base_hf_config.hidden_size

        # MoE-specific config
        self.num_experts = getattr(base_hf_config, "n_routed_experts", 0)
        self.moe_intermediate_size = getattr(
            base_hf_config, "moe_intermediate_size",
            getattr(base_hf_config, "intermediate_size", 0)
        )

        # Determine which layers are MoE layers
        # For DeepSeek-V2: layers after first_k_dense_replace are MoE
        self.first_k_dense_replace = getattr(base_hf_config, "first_k_dense_replace", 0)
        self.moe_layer_ids = set(range(self.first_k_dense_replace, self.num_layers))

        # Apply TP sharding to intermediate size
        # MoE expert weights are sharded along intermediate dimension
        self.intermediate_size_per_partition = self.moe_intermediate_size
        if self.tp_size > 1:
            self.intermediate_size_per_partition = divide(
                self.moe_intermediate_size, self.tp_size
            )

        # Buffers will be initialized lazily when needed
        self._buffers_initialized = False
        # gate and up are stored separately because they have independent lora_A
        self.gate_A_buffer: Optional[torch.Tensor] = None
        self.gate_B_buffer: Optional[torch.Tensor] = None
        self.up_A_buffer: Optional[torch.Tensor] = None
        self.up_B_buffer: Optional[torch.Tensor] = None
        self.down_A_buffer: Optional[torch.Tensor] = None
        self.down_B_buffer: Optional[torch.Tensor] = None

        # Track which adapters are loaded
        # adapter_uid -> buffer_slot_id
        self.uid_to_buffer_id: Dict[str, int] = {}
        self.buffer_id_to_uid: List[Optional[str]] = [None] * max_loras_per_batch

        logger.info(
            f"MoELoRAMemoryPool initialized: "
            f"num_experts={self.num_experts}, "
            f"moe_layers={len(self.moe_layer_ids)}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}"
        )

    def init_buffers(self):
        """
        Initialize GPU buffers for MoE LoRA weights.

        Called lazily when first MoE LoRA adapter is loaded.
        """
        if self._buffers_initialized:
            return

        if self.num_experts == 0:
            logger.warning("MoE not configured in model, skipping buffer initialization")
            return

        num_moe_layers = len(self.moe_layer_ids)
        if num_moe_layers == 0:
            logger.warning("No MoE layers found, skipping buffer initialization")
            return

        logger.info(
            f"Initializing MoE LoRA buffers: "
            f"max_loras={self.max_loras_per_batch}, "
            f"num_moe_layers={num_moe_layers}, "
            f"num_experts={self.num_experts}, "
            f"max_rank={self.max_lora_rank}"
        )

        # gate_A: (max_loras, num_moe_layers, num_experts, max_rank, hidden_size)
        # gate_proj LoRA A projects from hidden_size to rank
        self.gate_A_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.max_lora_rank,
                self.hidden_size,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # gate_B: (max_loras, num_moe_layers, num_experts, intermediate_size, max_rank)
        # gate_proj LoRA B projects from rank to intermediate_size
        self.gate_B_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.intermediate_size_per_partition,
                self.max_lora_rank,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # up_A: (max_loras, num_moe_layers, num_experts, max_rank, hidden_size)
        # up_proj LoRA A projects from hidden_size to rank
        self.up_A_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.max_lora_rank,
                self.hidden_size,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # up_B: (max_loras, num_moe_layers, num_experts, intermediate_size, max_rank)
        # up_proj LoRA B projects from rank to intermediate_size
        self.up_B_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.intermediate_size_per_partition,
                self.max_lora_rank,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # down_A: (max_loras, num_moe_layers, num_experts, max_rank, intermediate_size)
        # down_proj LoRA A projects from intermediate_size to rank
        self.down_A_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.max_lora_rank,
                self.intermediate_size_per_partition,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        # down_B: (max_loras, num_moe_layers, num_experts, hidden_size, max_rank)
        # down_proj LoRA B projects from rank to hidden_size
        self.down_B_buffer = torch.zeros(
            (
                self.max_loras_per_batch,
                num_moe_layers,
                self.num_experts,
                self.hidden_size,
                self.max_lora_rank,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        self._buffers_initialized = True

        # Calculate memory usage
        total_bytes = (
            self.gate_A_buffer.numel() +
            self.gate_B_buffer.numel() +
            self.up_A_buffer.numel() +
            self.up_B_buffer.numel() +
            self.down_A_buffer.numel() +
            self.down_B_buffer.numel()
        ) * self.gate_A_buffer.element_size()

        logger.info(
            f"MoE LoRA buffers allocated: {total_bytes / 1024 / 1024:.2f} MB"
        )

    def _get_moe_layer_index(self, layer_id: int) -> int:
        """Convert global layer_id to MoE layer index (0-based within MoE layers)."""
        return layer_id - self.first_k_dense_replace

    def load_adapter_weights(
        self,
        adapter: LoRAAdapter,
        buffer_id: int,
    ):
        """
        Load MoE LoRA weights from an adapter to GPU buffer.

        Args:
            adapter: LoRAAdapter with moe_layers populated
            buffer_id: Slot in the buffer to load to (0 to max_loras_per_batch-1)
        """
        if not adapter.has_moe_lora:
            return

        # Initialize buffers if not done yet
        self.init_buffers()

        if not self._buffers_initialized:
            logger.warning("MoE LoRA buffers not initialized, cannot load weights")
            return

        lora_rank = adapter.config.r

        debug_count = 0
        for layer_id, moe_layer in adapter.moe_layers.items():
            if layer_id not in self.moe_layer_ids:
                logger.warning(
                    f"Layer {layer_id} is not a MoE layer, skipping MoE LoRA weights"
                )
                continue

            moe_layer_idx = self._get_moe_layer_index(layer_id)

            # Debug logging for first layer
            if debug_count == 0:
                debug_count += 1
                logger.info(f"[MoE Weight Load DEBUG] Loading layer_id={layer_id} -> moe_layer_idx={moe_layer_idx}")
                logger.info(f"  first_k_dense_replace={self.first_k_dense_replace}")
                logger.info(f"  moe_layer.expert_ids: {sorted(moe_layer.expert_ids)[:5]}...")
                logger.info(f"  gate_lora_a keys: {sorted(moe_layer.gate_lora_a.keys())[:5]}...")
                if moe_layer.gate_lora_a:
                    first_expert = sorted(moe_layer.gate_lora_a.keys())[0]
                    weight = moe_layer.gate_lora_a[first_expert]
                    logger.info(f"  gate_lora_a[{first_expert}] shape: {weight.shape}, norm: {weight.norm():.6f}")

            for expert_id in moe_layer.expert_ids:
                # Load gate_proj LoRA A
                if expert_id in moe_layer.gate_lora_a:
                    weight = moe_layer.gate_lora_a[expert_id]
                    # gate_A input is hidden_size, no TP slicing needed
                    self._copy_weight_to_buffer(
                        self.gate_A_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=0,  # rank dimension
                    )

                # Load gate_proj LoRA B
                if expert_id in moe_layer.gate_lora_b:
                    weight = moe_layer.gate_lora_b[expert_id]
                    # Apply TP slicing for output dimension
                    if self.tp_size > 1:
                        weight = self._slice_intermediate_weight(weight)
                    self._copy_weight_to_buffer(
                        self.gate_B_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=1,  # rank is last dimension
                    )

                # Load up_proj LoRA A
                if expert_id in moe_layer.up_lora_a:
                    weight = moe_layer.up_lora_a[expert_id]
                    # up_A input is hidden_size, no TP slicing needed
                    self._copy_weight_to_buffer(
                        self.up_A_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=0,  # rank dimension
                    )

                # Load up_proj LoRA B
                if expert_id in moe_layer.up_lora_b:
                    weight = moe_layer.up_lora_b[expert_id]
                    # Apply TP slicing for output dimension
                    if self.tp_size > 1:
                        weight = self._slice_intermediate_weight(weight)
                    self._copy_weight_to_buffer(
                        self.up_B_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=1,  # rank is last dimension
                    )

                # Load down_proj LoRA A
                if expert_id in moe_layer.down_lora_a:
                    weight = moe_layer.down_lora_a[expert_id]
                    # Apply TP slicing for input dimension
                    if self.tp_size > 1:
                        weight = self._slice_down_a_weight(weight)
                    self._copy_weight_to_buffer(
                        self.down_A_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=0,  # rank dimension
                    )

                # Load down_proj LoRA B
                if expert_id in moe_layer.down_lora_b:
                    weight = moe_layer.down_lora_b[expert_id]
                    # down_B output is hidden_size, no slicing
                    self._copy_weight_to_buffer(
                        self.down_B_buffer[buffer_id, moe_layer_idx, expert_id],
                        weight,
                        lora_rank,
                        dim=1,  # rank is last dimension
                    )

        # Debug: verify loaded weights
        if adapter.moe_layers:
            first_layer_id = sorted(adapter.moe_layers.keys())[0]
            moe_layer_idx = self._get_moe_layer_index(first_layer_id)
            moe_layer = adapter.moe_layers[first_layer_id]
            if moe_layer.expert_ids:
                first_expert = sorted(moe_layer.expert_ids)[0]
                logger.info(f"[MoE Weight Load DEBUG] Verification after loading:")
                logger.info(f"  buffer_id={buffer_id}, moe_layer_idx={moe_layer_idx}, expert={first_expert}")
                gate_a_loaded = self.gate_A_buffer[buffer_id, moe_layer_idx, first_expert, :lora_rank, :]
                logger.info(f"  gate_A_buffer loaded: shape={gate_a_loaded.shape}, norm={gate_a_loaded.norm():.6f}")
                gate_b_loaded = self.gate_B_buffer[buffer_id, moe_layer_idx, first_expert, :, :lora_rank]
                logger.info(f"  gate_B_buffer loaded: shape={gate_b_loaded.shape}, norm={gate_b_loaded.norm():.6f}")

    def _copy_weight_to_buffer(
        self,
        buffer: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        dim: int,
    ):
        """
        Copy weight to buffer, handling rank dimension.

        Args:
            buffer: Target buffer tensor
            weight: Source weight tensor
            rank: Actual LoRA rank (may be smaller than max_rank)
            dim: Which dimension is the rank dimension (0 or 1)
        """
        if dim == 0:
            # Rank is first dimension: (rank, other_dim)
            buffer[:rank, :].copy_(weight.to(self.device))
        else:
            # Rank is last dimension: (other_dim, rank)
            buffer[:, :rank].copy_(weight.to(self.device))

    def _slice_intermediate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Slice LoRA B weight along intermediate dimension for TP.

        weight has shape (intermediate_size, rank).
        """
        shard_size = weight.shape[0] // self.tp_size
        start = self.tp_rank * shard_size
        end = start + shard_size
        return weight[start:end, :]

    def _slice_down_a_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Slice down LoRA A weight for TP.

        down_A has shape (rank, intermediate_size).
        We need to slice along intermediate_size dimension.
        """
        shard_size = weight.shape[1] // self.tp_size
        start = self.tp_rank * shard_size
        end = start + shard_size
        return weight[:, start:end]

    def prepare_batch(
        self,
        cur_uids: Set[str],
        lora_adapters: Dict[str, LoRAAdapter],
    ) -> Optional[Dict[str, int]]:
        """
        Prepare MoE LoRA buffers for a batch.

        Ensures all required adapters are loaded to GPU buffers.

        Args:
            cur_uids: Set of adapter UIDs needed for this batch
            lora_adapters: Dictionary of available adapters

        Returns:
            Dictionary mapping uid -> buffer_id, or None if no MoE LoRA
        """
        # Check if any adapter has MoE LoRA
        has_moe_lora = any(
            uid is not None and
            uid in lora_adapters and
            lora_adapters[uid].has_moe_lora
            for uid in cur_uids
        )

        if not has_moe_lora:
            return None

        # Ensure buffers are initialized
        self.init_buffers()

        # Load missing adapters
        for uid in cur_uids:
            if uid is None:
                continue

            if uid not in self.uid_to_buffer_id:
                # Find empty slot
                buffer_id = self._get_available_slot(cur_uids)
                if buffer_id is None:
                    raise RuntimeError(
                        f"No available MoE LoRA buffer slot for adapter {uid}"
                    )

                # Load weights
                adapter = lora_adapters.get(uid)
                if adapter and adapter.has_moe_lora:
                    self.load_adapter_weights(adapter, buffer_id)

                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

        return {uid: self.uid_to_buffer_id[uid] for uid in cur_uids if uid is not None}

    def _get_available_slot(self, cur_uids: Set[str]) -> Optional[int]:
        """Find an available buffer slot, evicting if necessary."""
        # First, look for empty slots
        for i, uid in enumerate(self.buffer_id_to_uid):
            if uid is None:
                return i

        # Need to evict - find slot not in current batch
        for i, uid in enumerate(self.buffer_id_to_uid):
            if uid not in cur_uids:
                # Evict this slot
                old_uid = self.buffer_id_to_uid[i]
                if old_uid is not None:
                    del self.uid_to_buffer_id[old_uid]
                self.buffer_id_to_uid[i] = None
                return i

        return None

    def get_buffer_id(self, uid: str) -> Optional[int]:
        """Get buffer slot ID for an adapter."""
        return self.uid_to_buffer_id.get(uid)

    def get_buffers(
        self,
        layer_id: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Get MoE LoRA buffers for a specific layer.

        Args:
            layer_id: Global layer ID

        Returns:
            Tuple of (gate_A, gate_B, up_A, up_B, down_A, down_B) buffers for this layer,
            or (None, None, None, None, None, None) if not a MoE layer or buffers not initialized.
        """
        if not self._buffers_initialized:
            return None, None, None, None, None, None

        if layer_id not in self.moe_layer_ids:
            return None, None, None, None, None, None

        moe_layer_idx = self._get_moe_layer_index(layer_id)

        return (
            self.gate_A_buffer[:, moe_layer_idx, :, :, :],
            self.gate_B_buffer[:, moe_layer_idx, :, :, :],
            self.up_A_buffer[:, moe_layer_idx, :, :, :],
            self.up_B_buffer[:, moe_layer_idx, :, :, :],
            self.down_A_buffer[:, moe_layer_idx, :, :, :],
            self.down_B_buffer[:, moe_layer_idx, :, :, :],
        )

    def has_moe_lora_for_layer(self, layer_id: int) -> bool:
        """Check if there's MoE LoRA data for a specific layer."""
        return (
            self._buffers_initialized and
            layer_id in self.moe_layer_ids and
            len(self.uid_to_buffer_id) > 0
        )
