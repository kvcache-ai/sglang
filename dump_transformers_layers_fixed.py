#!/usr/bin/env python3
"""
Dump每层输出 - 纯 Transformers + PEFT LoRA 版本
修复版本：手动加载 original_moe 格式的 LoRA 权重
"""

import os
import sys
import pickle
import numpy as np

# 设置环境变量禁用 flash_attn（在导入任何库之前）
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'

# 屏蔽可能导致问题的模块
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

sys.modules['flash_attn'] = create_fake_module('flash_attn')
sys.modules['flash_attn.flash_attn_interface'] = create_fake_module('flash_attn.flash_attn_interface')
sys.modules['flash_attn.bert_padding'] = create_fake_module('flash_attn.bert_padding')
sys.modules['flash_attn_2_cuda'] = create_fake_module('flash_attn_2_cuda')
sys.modules['ktransformers'] = create_fake_module('ktransformers')
sys.modules['ktransformers.util'] = create_fake_module('ktransformers.util')
sys.modules['ktransformers.util.grad_wrapper'] = create_fake_module('ktransformers.util.grad_wrapper')
sys.modules['ktransformers.util.grad_wrapper'].maybe_no_grad = lambda fn: fn

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors import safe_open
import argparse


def fix_load_moe_lora(peft_model, lora_path):
    """手动修复加载 original_moe 格式的 LoRA 权重"""
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")

    # 匹配 original_moe 格式的键名
    moe_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.original_moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.lora_(A|B)\.weight"
    )

    loaded_count = 0
    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            match = moe_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)  # gate_proj, up_proj, down_proj
                lora_type = match.group(4)  # A or B

                # 获取 PEFT 模型中对应的层
                peft_base = peft_model.base_model.model
                layer_mlp = peft_base.model.layers[layer_idx].mlp
                expert = layer_mlp.experts[expert_idx]
                proj = getattr(expert, proj_name)

                # 获取权重
                weight = f.get_tensor(key)

                # 设置到对应的 LoRA 层
                if lora_type == "A":
                    proj.lora_A['default'].weight.data.copy_(weight)
                else:
                    proj.lora_B['default'].weight.data.copy_(weight)

                loaded_count += 1

    print(f"[FIX] Manually loaded {loaded_count} MoE expert LoRA weights")
    return loaded_count


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


def register_hooks(model, recorder):
    """注册hooks到所有层"""
    hooks = []

    # 获取实际的base model（处理PeftModel包装）
    if hasattr(model, 'base_model'):
        # PeftModel: model.base_model.model
        base_model = model.base_model.model
    else:
        # 普通model
        base_model = model

    # 注册到每个transformer层
    for i, layer in enumerate(base_model.model.layers):
        # 注册到整个层的输出
        hook = layer.register_forward_hook(recorder.make_hook(f"layer_{i}"))
        hooks.append(hook)

        # 注册到MoE层
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            hook = layer.mlp.register_forward_hook(recorder.make_hook(f"layer_{i}_moe"))
            hooks.append(hook)

    # 注册到最后的norm和lm_head
    if hasattr(base_model.model, 'norm'):
        hook = base_model.model.norm.register_forward_hook(recorder.make_hook("final_norm"))
        hooks.append(hook)

    # lm_head通常在最外层model
    lm_head = None
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(base_model, 'lm_head'):
        lm_head = base_model.lm_head

    if lm_head is not None:
        hook = lm_head.register_forward_hook(recorder.make_hook("lm_head"))
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
            use_cache=False,  # 禁用cache以避免兼容性问题
        )

    # 解码（跳过输入部分）
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


def main():
    parser = argparse.ArgumentParser(description="Dump Transformers layer outputs")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/data/models/DeepSeek-V2-Lite-Chat",
        help="Path to base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (optional)",
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
        default="/tmp/transformers_dumps",
        help="Directory to save dumps",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Transformers Layer Output Dump (FIXED - with original_moe LoRA loading)")
    print("=" * 80)

    # 1. 加载 tokenizer
    print(f"\n[1/5] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # 2. 加载 base model
    print(f"[2/5] Loading base model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # 3. 加载 LoRA（如果指定）
    if args.lora_path:
        print(f"[3/5] Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(
            model,
            args.lora_path,
        )

        # 4. 修复加载 original_moe 格式的 LoRA 权重
        print(f"[4/5] Fixing MoE expert LoRA loading...")
        fix_load_moe_lora(model, args.lora_path)

        output_filename = "transformers_lora_outputs.pkl"
    else:
        print(f"[3/5] No LoRA adapter specified, using base model")
        print(f"[4/5] Skipping LoRA fix (no adapter)")
        output_filename = "transformers_base_outputs.pkl"

    model.eval()

    # 5. 注册 hooks
    print(f"[5/5] Registering hooks...")
    recorder = LayerOutputRecorder(args.output_dir)
    hooks = register_hooks(model, recorder)
    print(f"       Registered {len(hooks)} hooks")

    # 6. 生成并记录
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

    # 7. 保存dumps
    recorder.save(output_filename)

    # 8. 清理hooks
    for hook in hooks:
        hook.remove()

    print(f"\n[DONE] Dump complete!")


if __name__ == "__main__":
    main()
