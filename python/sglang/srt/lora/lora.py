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

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.lora_registry import LORA_SUPPORTED_BACKENDS
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


# Pattern to match MoE expert LoRA weights
# Examples:
#   model.layers.1.mlp.experts.0.gate_proj.lora_A.weight
#   model.layers.1.mlp.experts.0.up_proj.lora_B.weight
#   model.layers.1.mlp.experts.0.down_proj.lora_A.weight
MOE_EXPERT_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_([AB])\.weight"
)


def get_moe_expert_info(weight_name: str) -> Optional[Tuple[int, int, str, str]]:
    """
    Parse MoE expert LoRA weight name.

    Args:
        weight_name: Weight name like "model.layers.1.mlp.experts.0.gate_proj.lora_A.weight"

    Returns:
        Tuple of (layer_id, expert_id, proj_type, lora_type) or None if not a MoE expert weight
        - layer_id: int, the layer index
        - expert_id: int, the expert index
        - proj_type: str, one of "gate_proj", "up_proj", "down_proj"
        - lora_type: str, "A" or "B"
    """
    match = MOE_EXPERT_PATTERN.search(weight_name)
    if match:
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        proj_type = match.group(3)
        lora_type = match.group(4)
        return (layer_id, expert_id, proj_type, lora_type)
    return None


@dataclass
class MoELoRALayer:
    """
    Stores per-expert LoRA weights for a single MoE layer.

    For each expert, we store SEPARATE gate and up LoRA weights:
    - gate_lora_a: (rank, hidden_size) - gate_proj LoRA A
    - gate_lora_b: (intermediate_size, rank) - gate_proj LoRA B
    - up_lora_a: (rank, hidden_size) - up_proj LoRA A
    - up_lora_b: (intermediate_size, rank) - up_proj LoRA B
    - down_lora_a: (rank, intermediate_size) - down LoRA A
    - down_lora_b: (hidden_size, rank) - down LoRA B

    NOTE: gate and up are stored separately because in PEFT LoRA,
    each linear layer has independent lora_A and lora_B matrices.
    """
    # Per-expert gate_proj LoRA weights
    gate_lora_a: Dict[int, torch.Tensor] = field(default_factory=dict)
    gate_lora_b: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-expert up_proj LoRA weights
    up_lora_a: Dict[int, torch.Tensor] = field(default_factory=dict)
    up_lora_b: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-expert down_proj LoRA weights
    down_lora_a: Dict[int, torch.Tensor] = field(default_factory=dict)
    down_lora_b: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Metadata
    expert_ids: Set[int] = field(default_factory=set)
    rank: int = 0
    scaling: float = 1.0

    def add_expert_weight(
        self,
        expert_id: int,
        proj_type: str,
        lora_type: str,
        weight: torch.Tensor,
    ):
        """Add a single expert weight."""
        self.expert_ids.add(expert_id)

        if proj_type == "gate_proj":
            if lora_type == "A":
                self.gate_lora_a[expert_id] = weight
            else:
                self.gate_lora_b[expert_id] = weight
        elif proj_type == "up_proj":
            if lora_type == "A":
                self.up_lora_a[expert_id] = weight
            else:
                self.up_lora_b[expert_id] = weight
        elif proj_type == "down_proj":
            if lora_type == "A":
                self.down_lora_a[expert_id] = weight
            else:
                self.down_lora_b[expert_id] = weight

        # Update rank from weight shape
        if lora_type == "A" and self.rank == 0:
            self.rank = weight.shape[0]

    def finalize(self):
        """
        Finalize the MoE LoRA layer after all weights are loaded.

        Previously this merged gate+up, but now we keep them separate
        because gate_lora_a and up_lora_a are different matrices in PEFT.
        """
        # No merging needed - gate and up are stored separately
        pass

    @property
    def num_experts(self) -> int:
        return len(self.expert_ids)

    def has_expert(self, expert_id: int) -> bool:
        return expert_id in self.expert_ids


class LoRALayer(nn.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # lora weights in cpu. The weights are loaded from checkpoint.
        self.weights: Dict[str, torch.Tensor] = {}


class LoRAAdapter(nn.Module):

    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: BaseLoRABackend = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: List[LoRALayer] = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for _ in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.added_tokens_embeddings: Dict[str, torch.Tensor] = {}

        # MoE LoRA layers: layer_id -> MoELoRALayer
        self.moe_layers: Dict[int, MoELoRALayer] = {}
        self.has_moe_lora: bool = False

    # initialize the LoRA weights to cpu
    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        # Get normalized target modules for filtering
        from sglang.srt.lora.utils import get_normalized_target_modules

        normalized_target_modules = get_normalized_target_modules(
            self.config.target_modules
        )

        # Track MoE expert weights separately
        moe_expert_count = 0

        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            # Check if this is a MoE expert weight
            moe_info = get_moe_expert_info(name)
            if moe_info is not None:
                layer_id, expert_id, proj_type, lora_type = moe_info

                # Create MoELoRALayer if not exists
                if layer_id not in self.moe_layers:
                    self.moe_layers[layer_id] = MoELoRALayer(scaling=self.scaling)

                # Add weight to MoE layer
                self.moe_layers[layer_id].add_expert_weight(
                    expert_id, proj_type, lora_type, loaded_weight.cpu()
                )
                moe_expert_count += 1
                continue

            layer_id = get_layer_id(name)
            if layer_id is not None:
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            elif "embed_tokens" in name or "lm_head" in name:
                # Check if this module is declared in target_modules before loading
                module_name = "embed_tokens" if "embed_tokens" in name else "lm_head"
                if module_name in normalized_target_modules:
                    self.embedding_layers[name] = loaded_weight.cpu()
                else:
                    logger.debug(
                        f"Skipping {name} as '{module_name}' is not in adapter's target_modules: {self.config.target_modules}"
                    )
            elif "input_embeddings" in name or "output_embeddings" in name:
                # added/extra token emb
                self.added_tokens_embeddings[name] = loaded_weight.cpu()
                assert loaded_weight.shape[0] == self.config.lora_added_tokens_size, (
                    f"LoRA adapter {self.uid} has extra_vocab_size {self.config.extra_vocab_size} specified in the config, "
                    f"but the loaded weight has {loaded_weight.shape[0]} extra vocab size"
                )

        # normalize kv_proj and gate_up_proj
        for layer in self.layers:
            weight_names = list(layer.weights.keys())
            self.normalize_qkv_proj(weight_names, layer.weights)
            self.normalize_gate_up_proj(weight_names, layer.weights)

        # Finalize MoE LoRA layers (merge gate + up)
        if self.moe_layers:
            self.has_moe_lora = True
            for layer_id, moe_layer in self.moe_layers.items():
                moe_layer.finalize()
            logger.info(
                f"LoRA adapter '{self.uid}' loaded {moe_expert_count} MoE expert weights "
                f"across {len(self.moe_layers)} layers, "
                f"covering {sum(l.num_experts for l in self.moe_layers.values())} unique experts"
            )

    def normalize_qkv_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        # Collect target q/k/v modules. This process is necessary since there might be no lora attached to k_proj
        target_module = set()
        for weight_name in weight_names:
            if "k_proj" in weight_name:
                target_module.add("k_proj")
            if "q_proj" in weight_name:
                target_module.add("q_proj")
            if "v_proj" in weight_name:
                target_module.add("v_proj")
            if "qkv_proj" in weight_name:
                target_module.add("qkv_proj")
            # Check for DeepSeek-V2 MLA architecture modules
            if "kv_a_proj_with_mqa" in weight_name or "kv_b_proj" in weight_name:
                target_module.add("deepseek_v2_mla")
        if len(target_module) == 0:
            return

        # Check if this is DeepSeek-V2 or V3 with MLA architecture
        # These models use q_proj + kv_a_proj_with_mqa + kv_b_proj instead of traditional q/k/v
        is_deepseek_mla = "deepseek_v2_mla" in target_module or (
            hasattr(self.base_hf_config, "model_type")
            and self.base_hf_config.model_type in ["deepseek_v2", "deepseek_v3"]
        )

        for weight_name in weight_names:
            # We assume every lora adaptor should contain lora modules for q_proj
            if "q_proj" in weight_name:
                q_name = weight_name
                k_name = weight_name.replace("q_proj", "k_proj")
                v_name = weight_name.replace("q_proj", "v_proj")
                qkv_name = weight_name.replace("q_proj", "qkv_proj")

                # For DeepSeek-V2/V3 MLA architecture, q_proj is standalone (ColumnParallelLinear)
                # Do NOT rename or merge - keep q_proj as is
                # The MLA architecture uses separate kv_a_proj_with_mqa and kv_b_proj for K/V
                if is_deepseek_mla:
                    # Keep q_proj unchanged
                    continue

                # Traditional architecture: merge q/k/v
                # If k_proj doesn't have lora, initialize it to zero
                k_proj_weight = (
                    weights[k_name]
                    if "k_proj" in target_module
                    else torch.zeros_like(weights[v_name])
                )
                weights[qkv_name] = torch.cat(
                    (
                        weights[q_name],
                        k_proj_weight,
                        weights[v_name],
                    ),
                    0,
                )
                weights.pop(q_name)
                if "k_proj" in target_module:
                    weights.pop(k_name)
                weights.pop(v_name)
            elif "qkv_proj" in weight_name:
                # If qkv_proj is already stacked, we normalize it following the SGL convention.
                qkv_name = weight_name
                q_name = weight_name.replace("qkv_proj", "q_proj")
                k_name = weight_name.replace("qkv_proj", "k_proj")
                v_name = weight_name.replace("qkv_proj", "v_proj")
                if "lora_A" in weight_name:
                    weights[qkv_name] = weights[qkv_name].repeat(3, 1)
                # else: no-op as LoRA B weight is already stacked.

    def normalize_gate_up_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        for weight_name in weight_names:
            if "gate_proj" in weight_name:
                up_name = weight_name.replace("gate_proj", "up_proj")
                gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")
                if up_name not in weights:
                    weights[up_name] = torch.zeros_like(weights[weight_name])
                    assert self.lora_backend.name in LORA_SUPPORTED_BACKENDS, (
                        f"LoRA weight initialization currently only supported for LoRA backends: {', '.join(b for b in LORA_SUPPORTED_BACKENDS)}"
                        f"Received backend: {self.lora_backend.name}. Please verify your backend configuration "
                        f"or consider implementing custom initialization logic for other backends."
                    )
                weights[gate_up_name] = torch.cat(
                    (weights[weight_name], weights[up_name]), 0
                )
                weights.pop(weight_name)
                if up_name in weights:
                    weights.pop(up_name)
            elif "gate_up_proj" in weight_name:
                # If gate_up_proj is already stacked, we normalize it following the SGL convention
                gate_up_name = weight_name
                if "lora_A" in weight_name:
                    weights[gate_up_name] = weights[gate_up_name].repeat(2, 1)
                # else: no-op as LoRA B weight is already stacked.
