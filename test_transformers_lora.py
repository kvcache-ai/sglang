#!/usr/bin/env python3
"""
纯 Transformers + PEFT 实现的 LoRA chat 脚本（CPU-only）
用于对比验证 SGLang+KT 的 LoRA 实现是否正确
"""

import os
import sys

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0, use_cache=True):
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
        if not use_cache:
            # 直接禁用 cache
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        else:
            # 尝试使用 cache，如果失败则禁用
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            except AttributeError as e:
                # Fallback: 如果 cache 有问题，禁用它
                print(f"      Warning: KV cache error ({e}), retrying with use_cache=False...")
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
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
    parser = argparse.ArgumentParser(description="Transformers LoRA Chat (CPU-only)")
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
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs LoRA model",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache (slower but more compatible)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Transformers LoRA Chat (CPU-only)")
    print("=" * 80)

    # 加载 tokenizer
    print(f"\n[1/3] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # 加载 base model
    print(f"[2/3] Loading base model from {args.model_path}...")
    print("      (This may take a while on CPU...)")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",  # 不使用 flash_attn
    )

    print(f"      Model loaded: {base_model.num_parameters() / 1e9:.2f}B parameters")

    # 测试 base model
    if args.compare or args.lora_path is None:
        print(f"\n[3/3] Generating with BASE MODEL...")
        print(f"      Prompt: {args.prompt}")
        print(f"      Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
        print()

        base_response = generate_response(
            base_model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_cache=not args.no_cache,
        )

        print("=" * 80)
        print("BASE MODEL OUTPUT:")
        print("-" * 80)
        print(base_response)
        print("=" * 80)

    # 测试 LoRA model
    if args.lora_path:
        print(f"\n[LoRA] Loading LoRA adapter from {args.lora_path}...")
        lora_model = PeftModel.from_pretrained(
            base_model,
            args.lora_path,
        )

        # 获取 LoRA 配置信息
        lora_config = lora_model.peft_config['default']
        print(f"       LoRA rank: {lora_config.r}")
        print(f"       LoRA alpha: {lora_config.lora_alpha}")
        print(f"       Target modules: {lora_config.target_modules}")

        # 检查 LoRA 参数统计
        total_params = 0
        lora_params = 0
        for name, param in lora_model.named_parameters():
            total_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
                # 打印前几个 LoRA 层的统计
                if lora_params < 100000:  # 只打印前几个
                    print(f"       {name}: shape={param.shape}, mean={param.abs().mean():.6e}")

        print(f"       Total LoRA parameters: {lora_params / 1e6:.2f}M ({lora_params / total_params * 100:.2f}%)")

        print(f"\n[LoRA] Generating with LORA MODEL...")
        print(f"       Prompt: {args.prompt}")
        print()

        lora_response = generate_response(
            lora_model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_cache=not args.no_cache,
        )

        print("=" * 80)
        print("LORA MODEL OUTPUT:")
        print("-" * 80)
        print(lora_response)
        print("=" * 80)

        # 比较两个输出
        if args.compare:
            print("\n" + "=" * 80)
            print("COMPARISON:")
            print("=" * 80)
            print(f"Base:  {base_response}")
            print(f"LoRA:  {lora_response}")
            print(f"Same:  {base_response == lora_response}")
            print("=" * 80)


if __name__ == "__main__":
    main()
