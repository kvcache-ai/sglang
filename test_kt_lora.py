#!/usr/bin/env python3
"""
使用 KT-kernel MoE + LoRA 的对比脚本（CPU-only）
独立脚本，不依赖 LLaMA-Factory，也不依赖旧版 ktransformers
"""

import argparse
import os
import sys

# 设置环境变量禁用 flash_attn（在导入任何库之前）
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'

# 屏蔽旧版 ktransformers 和 flash_attn 的导入
# 必须在导入 torch/transformers 之前完成
import types

def create_fake_module(name):
    """创建一个假的模块，避免导入错误"""
    import importlib.machinery
    module = types.ModuleType(name)
    module.__file__ = f"<fake {name}>"
    module.__path__ = []
    # 创建一个假的 ModuleSpec
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    # 添加一些常用的假函数/类
    def fake_func(*args, **kwargs):
        return None
    module.__dict__.update({
        'flash_attn_func': fake_func,
        'flash_attn_varlen_func': fake_func,
        'flash_attn_with_kvcache': fake_func,
        'index_first_axis': fake_func,
        'pad_input': fake_func,
        'unpad_input': fake_func,
    })
    return module

# 屏蔽所有可能的 flash_attn 和 ktransformers 导入
sys.modules['flash_attn'] = create_fake_module('flash_attn')
sys.modules['flash_attn.flash_attn_interface'] = create_fake_module('flash_attn.flash_attn_interface')
sys.modules['flash_attn.bert_padding'] = create_fake_module('flash_attn.bert_padding')
sys.modules['flash_attn_2_cuda'] = create_fake_module('flash_attn_2_cuda')
sys.modules['ktransformers'] = create_fake_module('ktransformers')
sys.modules['ktransformers.models'] = create_fake_module('ktransformers.models')
sys.modules['ktransformers.util'] = create_fake_module('ktransformers.util')
sys.modules['ktransformers.util.grad_wrapper'] = create_fake_module('ktransformers.util.grad_wrapper')
# 添加 maybe_no_grad 函数
sys.modules['ktransformers.util.grad_wrapper'].maybe_no_grad = lambda fn: fn

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig

# 检查 kt_kernel 是否可用
try:
    from kt_kernel import kt_kernel_ext
    KT_KERNEL_AVAILABLE = True
except ImportError:
    KT_KERNEL_AVAILABLE = False
    kt_kernel_ext = None


# =============================================================================
# MoE Architecture Configuration
# =============================================================================

@dataclass
class MOEArchConfig:
    """MoE architecture configuration for different model types."""
    moe_layer_attr: str  # Attribute name for MoE layer in transformer block
    router_attr: str  # Attribute name for router in MoE layer
    experts_attr: str  # Attribute name for experts list in MoE layer
    weight_names: tuple  # (gate_proj, up_proj, down_proj) names
    expert_num: int  # Total number of experts
    intermediate_size: int  # MLP intermediate dimension
    num_experts_per_tok: int  # Number of experts per token (top-k)
    has_shared_experts: bool = False  # Whether model has shared experts
    router_type: str = "linear"  # Router type


def get_moe_arch_config(config: PretrainedConfig) -> MOEArchConfig:
    """Get MoE architecture configuration based on model type."""
    arch = config.architectures[0] if config.architectures else ""

    if "DeepseekV2" in arch or "DeepseekV3" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",
        )
    elif "Qwen2Moe" in arch or "Qwen3Moe" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "shared_expert_intermediate_size", 0) > 0,
        )
    elif "Mixtral" in arch:
        return MOEArchConfig(
            moe_layer_attr="block_sparse_moe",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("w1", "w3", "w2"),
            expert_num=config.num_local_experts,
            intermediate_size=config.intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=False,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")


def get_moe_module(layer: nn.Module, moe_config: MOEArchConfig) -> Optional[nn.Module]:
    """Get MoE module from transformer layer."""
    moe_module = getattr(layer, moe_config.moe_layer_attr, None)
    if moe_module is None:
        return None
    if not hasattr(moe_module, moe_config.experts_attr):
        return None
    return moe_module


def extract_moe_weights(
    moe_module: nn.Module, moe_config: MOEArchConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract MoE expert weights from the module."""
    experts = getattr(moe_module, moe_config.experts_attr)
    gate_name, up_name, down_name = moe_config.weight_names

    gate_weights = []
    up_weights = []
    down_weights = []

    for expert in experts:
        gate_weights.append(getattr(expert, gate_name).weight.data)
        up_weights.append(getattr(expert, up_name).weight.data)
        down_weights.append(getattr(expert, down_name).weight.data)

    # Stack to [expert_num, out_features, in_features]
    gate_proj = torch.stack(gate_weights, dim=0)
    up_proj = torch.stack(up_weights, dim=0)
    down_proj = torch.stack(down_weights, dim=0)

    return gate_proj, up_proj, down_proj


def create_lora_params(
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    lora_rank: int,
    lora_alpha: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, nn.Parameter]:
    """Create LoRA parameters for MoE layer."""
    # Gate projection LoRA
    gate_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    # Up projection LoRA
    up_lora_a = torch.zeros(expert_num, lora_rank, hidden_size, dtype=dtype, device=device)
    up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=dtype, device=device)

    # Down projection LoRA
    down_lora_a = torch.zeros(expert_num, lora_rank, intermediate_size, dtype=dtype, device=device)
    down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=dtype, device=device)

    # Initialize A matrices with kaiming_uniform
    for tensor in [gate_lora_a, up_lora_a, down_lora_a]:
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    return {
        "gate_lora_a": nn.Parameter(gate_lora_a),
        "gate_lora_b": nn.Parameter(gate_lora_b),
        "up_lora_a": nn.Parameter(up_lora_a),
        "up_lora_b": nn.Parameter(up_lora_b),
        "down_lora_a": nn.Parameter(down_lora_a),
        "down_lora_b": nn.Parameter(down_lora_b),
    }


# =============================================================================
# LoRA Linear Wrapper for Shared Experts
# =============================================================================

class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for Linear layer (inference only).

    Uses the same computation as PEFT: compute LoRA in the same dtype as weights (bf16).
    """
    def __init__(
        self,
        base: nn.Module,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        lora_alpha: float
    ):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(lora_a, requires_grad=False)
        self.lora_b = nn.Parameter(lora_b, requires_grad=False)
        self.scaling = lora_alpha / float(self.lora_a.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # LoRA path: x @ lora_a.T @ lora_b.T
        # Use same dtype as PEFT (bf16) to minimize numerical differences
        x_lora = x.to(self.lora_a.dtype)
        lora_out = (x_lora @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + lora_out * self.scaling


# =============================================================================
# MOE AMX Function (Custom Autograd)
# =============================================================================

class MOEAMXFunction(torch.autograd.Function):
    """Custom autograd function for AMX MOE forward/backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_amx: Any,
        cpu_infer: Any,
        lora_params: Dict[str, nn.Parameter],
        hidden_size: int,
        num_experts_per_tok: int,
        training: bool = False,
    ) -> torch.Tensor:
        """Forward pass using AMX operator."""
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape

        # Flatten for AMX
        qlen = batch_size * seq_len
        expert_ids = topk_ids.view(qlen, num_experts_per_tok).to(torch.int64).cpu().contiguous()
        weights = topk_weights.view(qlen, num_experts_per_tok).to(torch.float32).cpu().contiguous()

        # Prepare input
        input_data = hidden_states.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()
        output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()

        # Batch size tensor
        bsz_tensor = torch.tensor([qlen], device="cpu")

        # Call AMX forward
        cpu_infer.submit(
            moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_data.data_ptr(),
                output.data_ptr(),
                training,
            )
        )
        cpu_infer.sync()

        # Save for backward
        ctx.moe_amx = moe_amx
        ctx.cpu_infer = cpu_infer
        ctx.lora_params = lora_params
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype

        # Reshape and return
        output = output.view(batch_size, seq_len, hidden_size)
        return output.to(device=original_device, dtype=original_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using AMX operator."""
        qlen = ctx.qlen
        hidden_size = ctx.hidden_size

        grad_output_flat = grad_output.view(qlen, hidden_size).to(torch.bfloat16).cpu().contiguous()

        # Allocate gradient buffers
        grad_input = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        grad_gate_lora_a = torch.zeros_like(ctx.lora_params["gate_lora_a"].data, device="cpu")
        grad_gate_lora_b = torch.zeros_like(ctx.lora_params["gate_lora_b"].data, device="cpu")
        grad_up_lora_a = torch.zeros_like(ctx.lora_params["up_lora_a"].data, device="cpu")
        grad_up_lora_b = torch.zeros_like(ctx.lora_params["up_lora_b"].data, device="cpu")
        grad_down_lora_a = torch.zeros_like(ctx.lora_params["down_lora_a"].data, device="cpu")
        grad_down_lora_b = torch.zeros_like(ctx.lora_params["down_lora_b"].data, device="cpu")

        # Call AMX backward
        ctx.cpu_infer.submit(
            ctx.moe_amx.backward_task(
                grad_output_flat.data_ptr(),
                grad_input.data_ptr(),
                grad_gate_lora_a.data_ptr(),
                grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(),
                grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(),
                grad_down_lora_b.data_ptr(),
            )
        )
        ctx.cpu_infer.sync()

        # Reshape grad_input
        grad_input = grad_input.view(ctx.batch_size, ctx.seq_len, hidden_size)
        grad_input = grad_input.to(device=ctx.original_device, dtype=ctx.original_dtype)

        # Return gradients (None for non-Tensor inputs)
        return grad_input, None, None, None, None, None, None, None, None


# =============================================================================
# MOE Layer Wrapper
# =============================================================================

class MOELayerWrapper(nn.Module):
    """Wrapper for MoE layer with AMX acceleration."""

    def __init__(
        self,
        original_moe: nn.Module,
        moe_amx: Any,
        cpu_infer: Any,
        lora_params: Dict[str, nn.Parameter],
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
    ):
        super().__init__()
        self.moe_amx = moe_amx
        self.cpu_infer = cpu_infer
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type

        # Store LoRA params as module parameters
        self.lora_params = nn.ParameterDict(lora_params)

        # Get router from original MoE
        self.router = getattr(original_moe, moe_config.router_attr)

        # Store shared experts if present
        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using AMX acceleration."""
        # Update LoRA pointers before forward
        self.update_lora_pointers()

        batch_size, seq_len, _ = hidden_states.shape

        # Get topk_ids and topk_weights based on router type
        if self.router_type == "deepseek_gate":
            router_output = self.router(hidden_states)
            if len(router_output) == 2:
                topk_ids, topk_weights = router_output
            else:
                topk_ids, topk_weights, _ = router_output
        else:
            # Qwen/Mixtral router
            router_logits = self.router(hidden_states.view(-1, self.hidden_size))
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(
                routing_weights, self.moe_config.num_experts_per_tok, dim=-1
            )
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Apply AMX forward
        moe_output = MOEAMXFunction.apply(
            hidden_states,
            topk_ids,
            topk_weights,
            self.moe_amx,
            self.cpu_infer,
            dict(self.lora_params),
            self.hidden_size,
            self.moe_config.num_experts_per_tok,
            self.training,
        )

        # Handle shared experts if present
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            moe_output = moe_output + shared_output

        return moe_output

    def update_lora_pointers(self):
        """Update AMX operator with current LoRA weight pointers."""
        self.cpu_infer.submit(
            self.moe_amx.update_lora_weights_task(
                self.lora_params["gate_lora_a"].data.data_ptr(),
                self.lora_params["gate_lora_b"].data.data_ptr(),
                self.lora_params["up_lora_a"].data.data_ptr(),
                self.lora_params["up_lora_b"].data.data_ptr(),
                self.lora_params["down_lora_a"].data.data_ptr(),
                self.lora_params["down_lora_b"].data.data_ptr(),
            )
        )
        self.cpu_infer.sync()


# =============================================================================
# KT Backend Functions
# =============================================================================

def init_kt_backend(num_threads: int = 32):
    """初始化 KT backend"""
    if not KT_KERNEL_AVAILABLE:
        raise RuntimeError("kt_kernel not available. Please install kt-kernel.")

    print(f"[KT] Creating CPUInfer with {num_threads} threads")
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    cpu_infer = kt_kernel_ext.CPUInfer(pool_config)

    return cpu_infer


def wrap_moe_layers_with_kt(
    model: Any,
    config: PretrainedConfig,
    cpu_infer: Any,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    backend: str = "AMXBF16",
):
    """用 KT AMX 包装 MoE 层"""
    moe_config = get_moe_arch_config(config)
    hidden_size = config.hidden_size

    # Determine AMX backend class
    if backend == "AMXBF16":
        AMX_MOE_CLASS = kt_kernel_ext.moe.AMXBF16_SFT_MOE
    else:
        AMX_MOE_CLASS = kt_kernel_ext.moe.AMXInt8_SFT_MOE

    wrappers = []
    moe_layer_count = 0

    # Iterate through transformer layers
    for layer_idx, layer in enumerate(model.model.layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        print(f"[KT] Wrapping MoE layer {layer_idx}")

        # 1. Extract MoE weights
        gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
        gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
        up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
        down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()

        # 2. Create LoRA parameters
        lora_params = create_lora_params(
            expert_num=moe_config.expert_num,
            hidden_size=hidden_size,
            intermediate_size=moe_config.intermediate_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # 3. Create MOESFTConfig
        kt_config = kt_kernel_ext.moe.MOESFTConfig()
        kt_config.expert_num = moe_config.expert_num
        kt_config.num_experts_per_tok = moe_config.num_experts_per_tok
        kt_config.hidden_size = hidden_size
        kt_config.intermediate_size = moe_config.intermediate_size
        kt_config.lora_rank = lora_rank
        kt_config.lora_alpha = lora_alpha
        kt_config.max_cache_depth = 1
        kt_config.max_len = 4096
        kt_config.layer_idx = layer_idx

        # Set base weight pointers
        kt_config.gate_proj = gate_proj.data_ptr()
        kt_config.up_proj = up_proj.data_ptr()
        kt_config.down_proj = down_proj.data_ptr()

        # Set LoRA weight pointers
        kt_config.gate_lora_a = lora_params["gate_lora_a"].data.data_ptr()
        kt_config.gate_lora_b = lora_params["gate_lora_b"].data.data_ptr()
        kt_config.up_lora_a = lora_params["up_lora_a"].data.data_ptr()
        kt_config.up_lora_b = lora_params["up_lora_b"].data.data_ptr()
        kt_config.down_lora_a = lora_params["down_lora_a"].data.data_ptr()
        kt_config.down_lora_b = lora_params["down_lora_b"].data.data_ptr()

        # Set thread pool
        kt_config.pool = cpu_infer.backend_

        # 4. Create AMX MOE instance
        moe_amx = AMX_MOE_CLASS(kt_config)

        # 5. Load base weights
        cpu_infer.submit(moe_amx.load_weights_task())
        cpu_infer.sync()

        # 6. Warm up
        cpu_infer.submit(moe_amx.warm_up_task())
        cpu_infer.sync()

        # 7. Create wrapper
        wrapper = MOELayerWrapper(
            original_moe=moe_module,
            moe_amx=moe_amx,
            cpu_infer=cpu_infer,
            lora_params=lora_params,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
        )

        # 8. Replace MoE module in layer
        setattr(layer, moe_config.moe_layer_attr, wrapper)

        # Store base weights reference
        wrapper._base_weights = (gate_proj, up_proj, down_proj)

        wrappers.append(wrapper)
        moe_layer_count += 1

    print(f"[KT] Wrapped {moe_layer_count} MoE layers")
    return wrappers


def load_moe_lora_weights(wrappers: list, lora_path: str, lora_alpha: float = 16.0):
    """从 LoRA adapter 加载 MoE LoRA 权重（包括 shared_experts）"""
    from safetensors import safe_open

    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        print(f"[Warning] No adapter file found at {adapter_file}")
        return

    print(f"[KT] Loading MoE LoRA from {adapter_file}")

    # Build layer_idx -> wrapper mapping
    wrapper_map = {w.layer_idx: w for w in wrappers}

    # MoE LoRA key patterns
    moe_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(original_moe\.)?experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )
    # Shared experts LoRA pattern
    shared_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    # Group weights by layer
    layer_weights = {}
    shared_weights = {}

    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            # Try MoE expert pattern first
            match = moe_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(3))
                proj_name = match.group(4)
                ab = match.group(5)

                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                if expert_idx not in layer_weights[layer_idx]:
                    layer_weights[layer_idx][expert_idx] = {}
                if proj_name not in layer_weights[layer_idx][expert_idx]:
                    layer_weights[layer_idx][expert_idx][proj_name] = {}

                tensor = f.get_tensor(key)
                layer_weights[layer_idx][expert_idx][proj_name][ab] = tensor
                continue

            # Try shared experts pattern
            match = shared_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                proj_name = match.group(2)
                ab = match.group(3)

                if layer_idx not in shared_weights:
                    shared_weights[layer_idx] = {}
                if proj_name not in shared_weights[layer_idx]:
                    shared_weights[layer_idx][proj_name] = {}

                tensor = f.get_tensor(key)
                shared_weights[layer_idx][proj_name][ab] = tensor

    # Convert and load into KT wrappers
    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            continue

        wrapper = wrapper_map[layer_idx]
        num_experts = wrapper.moe_config.expert_num
        lora_rank = wrapper.lora_params["gate_lora_a"].shape[1]
        hidden_size = wrapper.hidden_size
        intermediate_size = wrapper.moe_config.intermediate_size

        # Initialize tensors
        gate_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        gate_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        up_lora_a = torch.zeros(num_experts, lora_rank, hidden_size, dtype=torch.bfloat16)
        up_lora_b = torch.zeros(num_experts, intermediate_size, lora_rank, dtype=torch.bfloat16)
        down_lora_a = torch.zeros(num_experts, lora_rank, intermediate_size, dtype=torch.bfloat16)
        down_lora_b = torch.zeros(num_experts, hidden_size, lora_rank, dtype=torch.bfloat16)

        # Fill in from adapter weights
        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= num_experts:
                continue

            for proj_name, ab_dict in proj_dict.items():
                if "A" in ab_dict:
                    a_tensor = ab_dict["A"].to(torch.bfloat16)
                    if proj_name == "gate_proj":
                        gate_lora_a[expert_idx] = a_tensor
                    elif proj_name == "up_proj":
                        up_lora_a[expert_idx] = a_tensor
                    elif proj_name == "down_proj":
                        down_lora_a[expert_idx] = a_tensor

                if "B" in ab_dict:
                    b_tensor = ab_dict["B"].to(torch.bfloat16)
                    if proj_name == "gate_proj":
                        gate_lora_b[expert_idx] = b_tensor
                    elif proj_name == "up_proj":
                        up_lora_b[expert_idx] = b_tensor
                    elif proj_name == "down_proj":
                        down_lora_b[expert_idx] = b_tensor

        # Copy to wrapper's lora_params
        device = wrapper.lora_params["gate_lora_a"].device
        wrapper.lora_params["gate_lora_a"].data.copy_(gate_lora_a.to(device))
        wrapper.lora_params["gate_lora_b"].data.copy_(gate_lora_b.to(device))
        wrapper.lora_params["up_lora_a"].data.copy_(up_lora_a.to(device))
        wrapper.lora_params["up_lora_b"].data.copy_(up_lora_b.to(device))
        wrapper.lora_params["down_lora_a"].data.copy_(down_lora_a.to(device))
        wrapper.lora_params["down_lora_b"].data.copy_(down_lora_b.to(device))

        # Update AMX operator pointers
        wrapper.update_lora_pointers()

        loaded_count += 1
        print(f"[KT] Loaded MoE LoRA for layer {layer_idx} ({len(experts_dict)} experts)")

    # Load shared_experts LoRA
    shared_loaded_count = 0
    for layer_idx, proj_dict in shared_weights.items():
        if layer_idx not in wrapper_map:
            continue

        wrapper = wrapper_map[layer_idx]
        if wrapper.shared_experts is None:
            continue

        # Apply LoRA to each projection
        for proj_name, ab_dict in proj_dict.items():
            if "A" not in ab_dict or "B" not in ab_dict:
                print(f"[Warning] Incomplete LoRA weights for layer {layer_idx} shared_experts.{proj_name}")
                continue

            lora_a = ab_dict["A"].to(torch.bfloat16)
            lora_b = ab_dict["B"].to(torch.bfloat16)

            # Get base layer
            base_layer = getattr(wrapper.shared_experts, proj_name, None)
            if base_layer is None:
                print(f"[Warning] Layer {layer_idx} shared_experts.{proj_name} not found")
                continue

            # Wrap with LoRALinear
            lora_layer = LoRALinear(base_layer, lora_a, lora_b, lora_alpha)
            setattr(wrapper.shared_experts, proj_name, lora_layer)
            shared_loaded_count += 1
            print(f"[KT] Loaded shared_experts LoRA for layer {layer_idx}.{proj_name} (lora_a: {lora_a.shape}, lora_b: {lora_b.shape})")

    print(f"[KT] Loaded MoE LoRA into {loaded_count} wrappers")
    print(f"[KT] Loaded shared_experts LoRA: {shared_loaded_count} projections")


def load_attention_lora_weights(model, lora_path: str, lora_alpha: float = 16.0):
    """Load attention layer LoRA weights (for layers that use standard attention)"""
    from safetensors import safe_open

    safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        print(f"[KT] Warning: LoRA adapter file not found: {safetensors_file}")
        return

    print(f"[KT] Loading attention LoRA from {safetensors_file}")

    # Pattern for attention LoRA keys
    attn_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.lora_(A|B)\.weight"
    )

    # Pattern for layer 0 MLP LoRA (non-MoE layer)
    mlp0_pattern = re.compile(
        r"base_model\.model\.model\.layers\.0\.mlp\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    attn_lora_weights = {}
    mlp0_lora_weights = {}

    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            match = attn_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                proj_name = match.group(2)
                lora_type = match.group(3).lower()

                if layer_idx not in attn_lora_weights:
                    attn_lora_weights[layer_idx] = {}
                if proj_name not in attn_lora_weights[layer_idx]:
                    attn_lora_weights[layer_idx][proj_name] = {}

                attn_lora_weights[layer_idx][proj_name][lora_type] = f.get_tensor(key).to(torch.bfloat16)
                continue

            match = mlp0_pattern.match(key)
            if match:
                proj_name = match.group(1)
                lora_type = match.group(2).lower()

                if proj_name not in mlp0_lora_weights:
                    mlp0_lora_weights[proj_name] = {}
                mlp0_lora_weights[proj_name][lora_type] = f.get_tensor(key).to(torch.bfloat16)

    # Apply attention LoRA
    loaded_attn = 0
    for layer_idx, projs in attn_lora_weights.items():
        layer = model.model.layers[layer_idx]
        for proj_name, lora_pair in projs.items():
            if "a" not in lora_pair or "b" not in lora_pair:
                continue
            base_layer = getattr(layer.self_attn, proj_name, None)
            if base_layer is None:
                continue
            setattr(
                layer.self_attn,
                proj_name,
                LoRALinear(base_layer, lora_pair["a"], lora_pair["b"], lora_alpha),
            )
            loaded_attn += 1

    # Apply layer 0 MLP LoRA
    loaded_mlp0 = 0
    layer0_mlp = model.model.layers[0].mlp
    for proj_name, lora_pair in mlp0_lora_weights.items():
        if "a" not in lora_pair or "b" not in lora_pair:
            continue
        base_layer = getattr(layer0_mlp, proj_name, None)
        if base_layer is None:
            continue
        setattr(
            layer0_mlp,
            proj_name,
            LoRALinear(base_layer, lora_pair["a"], lora_pair["b"], lora_alpha),
        )
        loaded_mlp0 += 1

    print(f"[KT] Loaded attention LoRA: {loaded_attn} proj tensors")
    print(f"[KT] Loaded layer0 MLP LoRA: {loaded_mlp0} proj tensors")


def generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0):
    """生成回复"""
    messages = [{"role": "user", "content": prompt}]

    # 应用 chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # 禁用 cache 避免兼容性问题
        )

    # 解码
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KT-kernel MoE + LoRA Chat Test")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/data/models/DeepSeek-V2-Lite-Chat",
        help="Path to base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/mnt/data/lpl/test_adapter_new/Kllama2_deepseekV2_WEST_ALL",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你是谁",
        help="User prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of CPU threads for KT",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("KT-kernel MoE + LoRA Chat Test")
    print("=" * 80)

    # 1. 初始化 KT backend
    print(f"\n[1/5] Initializing KT backend...")
    cpu_infer = init_kt_backend(args.num_threads)

    # 2. 加载 tokenizer
    print(f"\n[2/5] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # 3. 加载 model
    print(f"[3/5] Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",  # 使用标准 attention，不用 flash_attn
    )

    # 4. 用 KT 包装 MoE 层
    print(f"[4/5] Wrapping MoE layers with KT...")
    wrappers = wrap_moe_layers_with_kt(
        model,
        model.config,
        cpu_infer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # 5. 加载 LoRA 权重
    print(f"[5/6] Loading MoE LoRA weights from {args.lora_path}...")
    load_moe_lora_weights(wrappers, args.lora_path, lora_alpha=args.lora_alpha)

    # 6. 加载 attention LoRA 权重
    print(f"[6/6] Loading attention LoRA weights...")
    load_attention_lora_weights(model, args.lora_path, lora_alpha=args.lora_alpha)

    # 打印 LoRA 统计
    if wrappers:
        wrapper = wrappers[0]
        print(f"\n[LoRA Stats] Layer {wrapper.layer_idx}:")
        for name, param in wrapper.lora_params.items():
            print(f"  {name}: shape={param.shape}, mean={param.abs().mean():.6e}, max={param.abs().max():.6e}")

    # 设置为 eval 模式（推理模式，不需要 backward 缓存）
    model.eval()
    print("\n[Model] Set to eval mode (inference only)")

    # 生成
    print(f"\n[Generate] Prompt: {args.prompt}")
    print(f"           Max tokens: {args.max_tokens}")
    print()

    response = generate_response(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
    )

    print("=" * 80)
    print("OUTPUT:")
    print("-" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
