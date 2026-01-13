from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Set, Tuple

import torch

from sglang.srt.utils.hf_transformers_utils import AutoConfig


@dataclass
class LoRABatchInfo:
    # The forward mode is using CUDA Graph.
    use_cuda_graph: bool

    # Batch size
    bs: int

    # Number of segments. For triton backend, it is equal to batch size.
    num_segments: int

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    weight_indices: torch.Tensor

    # ranks of each lora adapter, in shape (lora_num,)
    lora_ranks: torch.Tensor

    # scaling of each lora adapter, in shape (lora_num,)
    scalings: torch.Tensor

    # Maximum segment length of current batch
    max_len: Optional[int]

    # Lengths of each segments in shape (num_segments,)
    seg_lens: Optional[torch.Tensor]

    # The logical (re)ordering of input rows (tokens), in shape (num_tokens,)
    permutation: Optional[torch.Tensor]


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_hidden_dim(
    module_name: str,
    config: AutoConfig,
    base_model: torch.nn.Module,
    layer_idx: int,
    lora_added_vocab_size: int = 0,
) -> Tuple[int]:
    """
    Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
    """

    if hasattr(base_model, "get_hidden_dim"):
        return base_model.get_hidden_dim(module_name, layer_idx)
    else:
        """
        WARNING: get_hidden_dim() is not defined,
        which is used to get the hidden dim for different lora modules
        Use the default one, but please check if it is correct for your model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if module_name == "q_proj":
            # For DeepSeek-V2 MLA and similar architectures where q_proj is not merged
            return config.hidden_size, head_dim * config.num_attention_heads
        elif module_name == "qkv_proj":
            return config.hidden_size, head_dim * (
                config.num_attention_heads + config.num_key_value_heads * 2
            )
        elif module_name == "o_proj":
            return (
                head_dim * config.num_attention_heads,
                config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            return config.hidden_size, config.intermediate_size * 2
        elif module_name == "down_proj":
            return config.intermediate_size, config.hidden_size
        elif module_name == "embed_tokens":
            # For embedding: input is vocab_size (as embedding lookup), output is hidden_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.vocab_size + lora_added_vocab_size, config.hidden_size
        elif module_name == "lm_head":
            # For lm_head: input is hidden_size, output is vocab_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.hidden_size, config.vocab_size + lora_added_vocab_size
        elif module_name == "gate":
            # MoE router gate: input is hidden_size, output is num_experts
            return config.hidden_size, config.num_experts
        else:
            raise NotImplementedError()


def get_normalized_target_modules(
    target_modules: Iterable[str],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized LoRA weights.
    Handles both base module names (e.g., "gate_proj") and prefixed module names (e.g., "feed_forward.gate_proj").

    For DeepSeek-V2/V3 MLA architecture, q_proj is kept separate (not merged into qkv_proj).
    """
    # Check if this is DeepSeek-V2/V3 MLA architecture
    target_modules_list = list(target_modules)
    is_deepseek_mla = any(
        "kv_a_proj_with_mqa" in name or "kv_b_proj" in name
        for name in target_modules_list
    )

    params_mapping = {
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "embed_tokens": "embed_tokens",
        "vocab_emb": "embed_tokens",
        "embeddings": "embed_tokens",
        "word_embeddings": "embed_tokens",
        "lm_head": "lm_head",
        "output": "lm_head",
    }

    # For non-MLA architectures, q_proj should also be mapped to qkv_proj
    if not is_deepseek_mla:
        params_mapping["q_proj"] = "qkv_proj"

    result = set()
    for name in target_modules_list:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result


def get_stacked_multiply(module_name: str) -> int:
    """
    Mapping a lora module name to its magnification at output dimension
    """
    stacked_rank = {
        "qkv_proj": 3,
        "gate_up_proj": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name
    Else raise ValueError.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


EMBEDDING_NAMES = ["embed_tokens", "lm_head"]
ROW_PARALLELISM_LINEAR_LORA_NAMES = ["o_proj", "down_proj"]
# ReplicatedLinear modules: weights are replicated across all TP ranks, not sharded
REPLICATED_LINEAR_LORA_NAMES = ["kv_a_proj_with_mqa"]
