#!/usr/bin/env python3
"""
Dump每层输出 - KT-kernel MoE 版本
"""

import argparse
import os
import sys
import pickle
import numpy as np

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


class LayerOutputRecorder:
    """记录每层的输出"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.layer_outputs = {}

    def make_hook(self, layer_name):
        """创建一个hook函数"""
        def hook(module, input, output):
            # 保存输出
            if isinstance(output, tuple):
                # 通常第一个元素是hidden_states
                tensor = output[0]
            else:
                tensor = output

            # 转换为numpy并保存
            if isinstance(tensor, torch.Tensor):
                self.layer_outputs[layer_name] = tensor.detach().cpu().float().numpy()
                print(f"[DUMP] {layer_name}: shape={tensor.shape}, mean={tensor.float().mean():.6e}, std={tensor.float().std():.6e}")

        return hook

    def save(self, filename):
        """保存所有层的输出"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.layer_outputs, f)
        print(f"\n[SAVE] Saved {len(self.layer_outputs)} layer outputs to {filepath}")

        # 打印统计信息
        print("\n[STATS] Layer output statistics:")
        for layer_name in sorted(self.layer_outputs.keys()):
            arr = self.layer_outputs[layer_name]
            print(f"  {layer_name}: shape={arr.shape}, mean={arr.mean():.6e}, std={arr.std():.6e}, min={arr.min():.6e}, max={arr.max():.6e}")


# 导入test_kt_lora.py中的所有必要代码
# 这里我直接复制关键部分

@dataclass
class MOEArchConfig:
    """MoE architecture configuration for different model types."""
    moe_layer_attr: str
    router_attr: str
    experts_attr: str
    weight_names: tuple
    expert_num: int
    num_experts_per_tok: int
    router_type: str
    has_shared_experts: bool = False


def get_moe_arch_config(config: PretrainedConfig) -> MOEArchConfig:
    """获取MoE架构配置"""
    model_type = getattr(config, "model_type", "").lower()

    if "deepseek" in model_type:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=getattr(config, "n_routed_experts", 64),
            num_experts_per_tok=getattr(config, "num_experts_per_tok", 6),
            router_type="deepseek_gate",
            has_shared_experts=True,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def extract_moe_weights(original_moe: nn.Module, moe_config: MOEArchConfig) -> Dict[str, torch.Tensor]:
    """从原始MoE层提取权重"""
    experts = getattr(original_moe, moe_config.experts_attr)
    gate_name, up_name, down_name = moe_config.weight_names

    num_experts = moe_config.expert_num
    expert_0 = experts[0]

    gate_weight = getattr(expert_0, gate_name).weight
    up_weight = getattr(expert_0, up_name).weight
    down_weight = getattr(expert_0, down_name).weight

    intermediate_size, hidden_size = gate_weight.shape
    _, intermediate_size_down = down_weight.shape

    gate_weights = torch.zeros((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up_weights = torch.zeros((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down_weights = torch.zeros((num_experts, hidden_size, intermediate_size_down), dtype=torch.bfloat16)

    for i in range(num_experts):
        expert = experts[i]
        gate_weights[i] = getattr(expert, gate_name).weight.data.to(torch.bfloat16)
        up_weights[i] = getattr(expert, up_name).weight.data.to(torch.bfloat16)
        down_weights[i] = getattr(expert, down_name).weight.data.to(torch.bfloat16)

    return {
        "gate_weight": gate_weights,
        "up_weight": up_weights,
        "down_weight": down_weights,
    }


def create_lora_params(num_experts: int, lora_rank: int, intermediate_size: int, hidden_size: int) -> Dict[str, nn.Parameter]:
    """创建LoRA参数"""
    lora_params = {
        "gate_lora_a": nn.Parameter(torch.zeros((num_experts, lora_rank, hidden_size), dtype=torch.bfloat16)),
        "gate_lora_b": nn.Parameter(torch.zeros((num_experts, intermediate_size, lora_rank), dtype=torch.bfloat16)),
        "up_lora_a": nn.Parameter(torch.zeros((num_experts, lora_rank, hidden_size), dtype=torch.bfloat16)),
        "up_lora_b": nn.Parameter(torch.zeros((num_experts, intermediate_size, lora_rank), dtype=torch.bfloat16)),
        "down_lora_a": nn.Parameter(torch.zeros((num_experts, lora_rank, intermediate_size), dtype=torch.bfloat16)),
        "down_lora_b": nn.Parameter(torch.zeros((num_experts, hidden_size, lora_rank), dtype=torch.bfloat16)),
    }
    return lora_params


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for Linear (inference only).

    Uses the same computation as PEFT: compute LoRA in the same dtype as weights (bf16).
    """

    def __init__(self, base: nn.Module, lora_a: torch.Tensor, lora_b: torch.Tensor, lora_alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(lora_a, requires_grad=False)
        self.lora_b = nn.Parameter(lora_b, requires_grad=False)
        self.scaling = lora_alpha / float(self.lora_a.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Use same dtype as PEFT (bf16) to minimize numerical differences
        x_lora = x.to(self.lora_a.dtype)
        lora_out = (x_lora @ self.lora_a.t()) @ self.lora_b.t()
        return base_out + lora_out * self.scaling


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
                training,  # save_for_backward
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
    def backward(ctx, grad_output):
        """Backward pass (not implemented for inference)."""
        raise NotImplementedError("Backward pass not implemented for KT MoE")


class MOELayerWrapper(nn.Module):
    """Wrapper that replaces original MoE layer with KT AMX implementation."""

    def __init__(
        self,
        original_moe: nn.Module,
        moe_amx: Any,
        cpu_infer: Any,
        moe_config: MOEArchConfig,
        lora_params: Dict[str, nn.Parameter],
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
    wrappers = []

    for layer_idx, layer in enumerate(model.model.layers):
        original_moe = getattr(layer, moe_config.moe_layer_attr)

        if not hasattr(original_moe, moe_config.experts_attr):
            continue

        # 提取权重
        weights = extract_moe_weights(original_moe, moe_config)
        gate_weight = weights["gate_weight"]
        up_weight = weights["up_weight"]
        down_weight = weights["down_weight"]

        num_experts, intermediate_size, hidden_size = gate_weight.shape

        # 创建 LoRA 参数
        lora_params = create_lora_params(num_experts, lora_rank, intermediate_size, hidden_size)

        # 创建 MOE SFT Config
        kt_config = kt_kernel_ext.moe.MOESFTConfig()
        kt_config.expert_num = num_experts
        kt_config.num_experts_per_tok = moe_config.num_experts_per_tok
        kt_config.hidden_size = hidden_size
        kt_config.intermediate_size = intermediate_size
        kt_config.lora_rank = lora_rank
        kt_config.lora_alpha = lora_alpha
        kt_config.max_cache_depth = 1
        kt_config.max_len = 4096
        kt_config.layer_idx = layer_idx

        # Set base weight pointers
        kt_config.gate_proj = gate_weight.data_ptr()
        kt_config.up_proj = up_weight.data_ptr()
        kt_config.down_proj = down_weight.data_ptr()

        # Set LoRA weight pointers
        kt_config.gate_lora_a = lora_params["gate_lora_a"].data.data_ptr()
        kt_config.gate_lora_b = lora_params["gate_lora_b"].data.data_ptr()
        kt_config.up_lora_a = lora_params["up_lora_a"].data.data_ptr()
        kt_config.up_lora_b = lora_params["up_lora_b"].data.data_ptr()
        kt_config.down_lora_a = lora_params["down_lora_a"].data.data_ptr()
        kt_config.down_lora_b = lora_params["down_lora_b"].data.data_ptr()

        # Set thread pool
        kt_config.pool = cpu_infer.backend_

        # 创建 AMX operator (use AMXBF16 backend)
        if backend == "AMXBF16":
            moe_amx = kt_kernel_ext.moe.AMXBF16_SFT_MOE(kt_config)
        else:
            moe_amx = kt_kernel_ext.moe.AMXInt8_SFT_MOE(kt_config)

        # Load base weights
        cpu_infer.submit(moe_amx.load_weights_task())
        cpu_infer.sync()

        # Warm up
        cpu_infer.submit(moe_amx.warm_up_task())
        cpu_infer.sync()

        # 创建 wrapper
        wrapper = MOELayerWrapper(
            original_moe,
            moe_amx,
            cpu_infer,
            moe_config,
            lora_params,
            hidden_size,
            layer_idx,
        )

        # 替换原始 MoE 层
        setattr(layer, moe_config.moe_layer_attr, wrapper)
        wrappers.append(wrapper)

        print(f"[KT] Wrapped MoE layer {layer_idx}")

    return wrappers


def load_moe_lora_weights(wrappers: list, lora_path: str, lora_alpha: float):
    """从 LoRA adapter 加载 MoE LoRA 权重"""
    from safetensors import safe_open

    safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(f"LoRA adapter file not found: {safetensors_file}")

    print(f"[KT] Loading MoE LoRA from {safetensors_file}")

    moe_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(original_moe\.)?experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )
    shared_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    layer_lora_weights = {}
    shared_lora_weights = {}

    total_keys = 0
    matched_keys = 0
    matched_moe_keys = 0
    matched_shared_keys = 0
    unmatched_keys = []
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            total_keys += 1
            match = moe_pattern.match(key)
            if match:
                matched_keys += 1
                matched_moe_keys += 1

                layer_idx = int(match.group(1))
                expert_idx = int(match.group(3))
                proj_name = match.group(4)
                lora_type = match.group(5)

                if layer_idx not in layer_lora_weights:
                    layer_lora_weights[layer_idx] = {}

                param_key = f"{proj_name}_lora_{lora_type.lower()}"
                if param_key not in layer_lora_weights[layer_idx]:
                    layer_lora_weights[layer_idx][param_key] = {}

                layer_lora_weights[layer_idx][param_key][expert_idx] = f.get_tensor(key).to(torch.bfloat16)
                continue

            match = shared_pattern.match(key)
            if match:
                matched_keys += 1
                matched_shared_keys += 1

                layer_idx = int(match.group(1))
                proj_name = match.group(2)
                lora_type = match.group(3).lower()

                if layer_idx not in shared_lora_weights:
                    shared_lora_weights[layer_idx] = {}
                if proj_name not in shared_lora_weights[layer_idx]:
                    shared_lora_weights[layer_idx][proj_name] = {}

                shared_lora_weights[layer_idx][proj_name][lora_type] = f.get_tensor(key).to(torch.bfloat16)
                continue

            unmatched_keys.append(key)

    # 将权重批量复制到每个wrapper
    loaded_layers = 0
    loaded_params = 0
    loaded_shared = 0
    for wrapper in wrappers:
        layer_idx = wrapper.layer_idx
        if layer_idx not in layer_lora_weights:
            continue

        lora_data = layer_lora_weights[layer_idx]
        num_experts = wrapper.moe_config.expert_num

        for param_name, expert_weights in lora_data.items():
            # 获取第一个expert的shape来初始化batch tensor
            first_weight = expert_weights[0]
            if param_name.endswith("_a"):
                # lora_A: [lora_rank, input_dim] -> batch: [num_experts, lora_rank, input_dim]
                lora_rank, input_dim = first_weight.shape
                batch_tensor = torch.zeros((num_experts, lora_rank, input_dim), dtype=torch.bfloat16)
            else:
                # lora_B: [output_dim, lora_rank] -> batch: [num_experts, output_dim, lora_rank]
                output_dim, lora_rank = first_weight.shape
                batch_tensor = torch.zeros((num_experts, output_dim, lora_rank), dtype=torch.bfloat16)

            # 填充所有expert的权重
            missing_experts = 0
            for expert_idx in range(num_experts):
                if expert_idx in expert_weights:
                    batch_tensor[expert_idx] = expert_weights[expert_idx]
                else:
                    missing_experts += 1

            # 复制到wrapper的parameter
            # 转换参数名: gate_proj_lora_a -> gate_lora_a
            wrapper_param_name = param_name.replace("_proj", "")
            wrapper.lora_params[wrapper_param_name].data.copy_(batch_tensor)
            loaded_params += 1
            if missing_experts > 0:
                print(
                    f"[KT][WARN] Layer {layer_idx} {param_name}: missing {missing_experts}/{num_experts} experts"
                )

        print(f"[KT] Loaded MoE LoRA for layer {layer_idx} ({num_experts} experts)")
        loaded_layers += 1

        if wrapper.shared_experts is not None and layer_idx in shared_lora_weights:
            for proj_name, lora_pair in shared_lora_weights[layer_idx].items():
                if "a" not in lora_pair or "b" not in lora_pair:
                    print(
                        f"[KT][WARN] Layer {layer_idx} shared_experts.{proj_name}: missing A/B weights"
                    )
                    continue
                base_layer = getattr(wrapper.shared_experts, proj_name, None)
                if base_layer is None:
                    print(
                        f"[KT][WARN] Layer {layer_idx} shared_experts.{proj_name}: base layer not found"
                    )
                    continue
                setattr(
                    wrapper.shared_experts,
                    proj_name,
                    LoRALinear(base_layer, lora_pair["a"], lora_pair["b"], lora_alpha),
                )
                loaded_shared += 1

    print(
        f"[KT] LoRA key match: {matched_keys}/{total_keys} keys "
        f"({0.0 if total_keys == 0 else matched_keys / total_keys:.2%})"
    )
    print(f"[KT] Matched MoE keys: {matched_moe_keys}, shared_experts keys: {matched_shared_keys}")
    if unmatched_keys:
        print("[KT][WARN] Unmatched LoRA keys:")
        for key in unmatched_keys:
            print(f"  {key}")
    print(f"[KT] Loaded MoE LoRA into {loaded_layers}/{len(wrappers)} wrappers")
    print(f"[KT] Loaded LoRA params: {loaded_params} param tensors")
    print(f"[KT] Loaded shared_experts LoRA: {loaded_shared} proj tensors")


def load_attention_lora_weights(model, lora_path: str, lora_alpha: float):
    """从 LoRA adapter 加载 attention 层的 LoRA 权重"""
    from safetensors import safe_open

    safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(f"LoRA adapter file not found: {safetensors_file}")

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


def register_hooks(model, recorder, moe_config):
    """注册hooks到所有层"""
    hooks = []

    # 注册到每个transformer层
    for i, layer in enumerate(model.model.layers):
        # 注册到整个层的输出
        hook = layer.register_forward_hook(recorder.make_hook(f"layer_{i}"))
        hooks.append(hook)

        # 注册到MoE层
        moe_layer = getattr(layer, moe_config.moe_layer_attr)
        hook = moe_layer.register_forward_hook(recorder.make_hook(f"layer_{i}_moe"))
        hooks.append(hook)

    # 注册到最后的norm和lm_head
    if hasattr(model.model, 'norm'):
        hook = model.model.norm.register_forward_hook(recorder.make_hook("final_norm"))
        hooks.append(hook)

    if hasattr(model, 'lm_head'):
        hook = model.lm_head.register_forward_hook(recorder.make_hook("lm_head"))
        hooks.append(hook)

    return hooks


def generate_with_dump(model, tokenizer, prompt, recorder, max_new_tokens=32):
    """生成文本并dump中间层输出"""
    messages = [{"role": "user", "content": prompt}]

    # 应用 chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt")

    # 生成
    print(f"\n[GENERATE] Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=None,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # 解码（跳过输入部分）
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


def main():
    parser = argparse.ArgumentParser(description="Dump KT layer outputs")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/data/models/DeepSeek-V2-Lite-Chat",
        help="Path to base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
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
        default=32,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/kt_dumps",
        help="Directory to save dumps",
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Number of CPU threads",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("KT Layer Output Dump")
    print("=" * 80)

    # 1. 初始化 KT backend
    print(f"\n[1/6] Initializing KT backend...")
    cpu_infer = init_kt_backend(args.num_threads)

    # 2. 加载 tokenizer
    print(f"\n[2/6] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # 3. 加载 model
    print(f"[3/6] Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # 4. 用 KT 包装 MoE 层
    print(f"[4/6] Wrapping MoE layers with KT...")
    wrappers = wrap_moe_layers_with_kt(
        model,
        model.config,
        cpu_infer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # 5. 加载 LoRA 权重
    print(f"[5/7] Loading MoE LoRA weights from {args.lora_path}...")
    load_moe_lora_weights(wrappers, args.lora_path, args.lora_alpha)

    # 5.5 加载 attention LoRA 权重
    print(f"[6/7] Loading attention LoRA weights...")
    load_attention_lora_weights(model, args.lora_path, args.lora_alpha)

    # 设置为 eval 模式
    model.eval()
    print("\n[Model] Set to eval mode (inference only)")

    # 7. 注册 hooks
    print(f"[7/7] Registering hooks...")
    moe_config = get_moe_arch_config(model.config)
    recorder = LayerOutputRecorder(args.output_dir)
    hooks = register_hooks(model, recorder, moe_config)
    print(f"       Registered {len(hooks)} hooks")

    # 7. 生成并记录
    output_filename = "kt_lora_outputs.pkl"
    print(f"\n[RUN] Running inference with prompt: '{args.prompt}'")
    print(f"      Output will be saved to: {args.output_dir}/{output_filename}")
    print()

    response = generate_with_dump(
        model,
        tokenizer,
        args.prompt,
        recorder,
        max_new_tokens=args.max_tokens
    )

    print("=" * 80)
    print("OUTPUT:")
    print("-" * 80)
    print(response)
    print("=" * 80)

    # 8. 保存dumps
    recorder.save(output_filename)

    # 9. 清理hooks
    for hook in hooks:
        hook.remove()

    print(f"\n[DONE] Dump complete!")


if __name__ == "__main__":
    main()
