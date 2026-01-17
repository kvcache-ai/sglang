#!/usr/bin/env python3
"""
MoE LoRA 调试脚本
用于验证权重形状、LoRA计算、专家映射等

Usage:
    python scripts/debug_moe_lora.py --converted /path/to/converted.pt --peft /path/to/adapter.safetensors
"""

import os
import sys
import torch


# ============================================================================
# 1. 验证转换后的权重
# ============================================================================

def verify_converted_weights(converted_path: str):
    """验证转换后的.pt文件"""
    print("=" * 60)
    print("Step 1: 验证转换后的权重")
    print("=" * 60)

    d = torch.load(converted_path, map_location="cpu", weights_only=True)

    print(f"\nMetadata: {d['metadata']}")

    layer_keys = [k for k in d.keys() if k.startswith('layer_')]
    print(f"\nLayers: {layer_keys[:5]}... (total: {len(layer_keys)})")

    # 找到第一个有MoE的层
    first_moe_layer = None
    for key in sorted(layer_keys, key=lambda x: int(x.split('_')[1])):
        if d[key]:
            first_moe_layer = key
            break

    if first_moe_layer:
        layer_data = d[first_moe_layer]
        print(f"\n{first_moe_layer} 权重形状:")
        for k, v in layer_data.items():
            print(f"  {k}: {v.shape}, dtype={v.dtype}")
            # 打印一些统计信息
            print(f"    mean={v.float().mean():.6f}, std={v.float().std():.6f}, min={v.float().min():.6f}, max={v.float().max():.6f}")

        # 验证形状是否符合预期
        metadata = d['metadata']
        E = metadata.get('num_experts', 64)
        R = metadata.get('lora_rank', 8)
        H = metadata.get('hidden_size', 2048)
        I = metadata.get('intermediate_size', 5632)

        print(f"\n预期形状 (E={E}, R={R}, H={H}, I={I}):")
        print(f"  gate_lora_a: [{E}, {R}, {H}]")
        print(f"  gate_lora_b: [{E}, {I}, {R}]")
        print(f"  up_lora_a: [{E}, {R}, {H}]")
        print(f"  up_lora_b: [{E}, {I}, {R}]")
        print(f"  down_lora_a: [{E}, {R}, {I}]")
        print(f"  down_lora_b: [{E}, {H}, {R}]")

        # 检查是否匹配
        print("\n形状验证:")
        expected_shapes = {
            "gate_lora_a": (E, R, H),
            "gate_lora_b": (E, I, R),
            "up_lora_a": (E, R, H),
            "up_lora_b": (E, I, R),
            "down_lora_a": (E, R, I),
            "down_lora_b": (E, H, R),
        }

        all_correct = True
        for name, expected in expected_shapes.items():
            if name in layer_data:
                actual = tuple(layer_data[name].shape)
                if actual == expected:
                    print(f"  ✓ {name}: {actual}")
                else:
                    print(f"  ✗ {name}: 实际{actual} vs 预期{expected}")
                    all_correct = False
            else:
                print(f"  ✗ {name}: 缺失")
                all_correct = False

        if all_correct:
            print("\n✓ 所有权重形状正确")
        else:
            print("\n✗ 存在形状不匹配的权重")

    return d


# ============================================================================
# 2. 验证原始PEFT权重
# ============================================================================

def verify_peft_weights(peft_path: str):
    """验证原始PEFT adapter权重"""
    print("\n" + "=" * 60)
    print("Step 2: 验证原始PEFT权重")
    print("=" * 60)

    try:
        from safetensors import safe_open
    except ImportError:
        print("警告: safetensors库未安装，跳过PEFT权重验证")
        return

    with safe_open(peft_path, framework='pt') as f:
        keys = list(f.keys())
        print(f"\n总共 {len(keys)} 个keys")

        # 查找MoE专家LoRA keys
        moe_keys = [k for k in keys if 'experts' in k and 'lora' in k]
        non_moe_keys = [k for k in keys if 'experts' not in k and 'lora' in k]

        print(f"MoE专家LoRA keys: {len(moe_keys)}")
        print(f"非MoE LoRA keys: {len(non_moe_keys)}")

        # 打印前几个MoE keys的形状
        if moe_keys:
            print("\n示例MoE LoRA权重形状 (前8个):")
            for key in sorted(moe_keys)[:8]:
                tensor = f.get_tensor(key)
                print(f"  {key}: {tensor.shape}")

        # 打印非MoE keys
        if non_moe_keys:
            print("\n非MoE LoRA keys (Attention/Shared Experts):")
            for key in sorted(non_moe_keys)[:8]:
                tensor = f.get_tensor(key)
                print(f"  {key}: {tensor.shape}")


# ============================================================================
# 3. 测试LoRA计算公式
# ============================================================================

def test_lora_computation():
    """测试LoRA计算公式是否正确"""
    print("\n" + "=" * 60)
    print("Step 3: 测试LoRA计算公式")
    print("=" * 60)

    # 小规模测试
    batch = 4
    hidden_size = 8
    intermediate_size = 16
    rank = 2
    alpha = 16.0
    scaling = alpha / rank

    # 创建测试数据
    torch.manual_seed(42)
    x = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    W = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
    lora_a = torch.randn(rank, hidden_size, dtype=torch.bfloat16) / 10
    lora_b = torch.randn(intermediate_size, rank, dtype=torch.bfloat16) / 10

    # 基础输出
    base_out = torch.mm(x, W.t())

    # LoRA输出 (kt-kernel公式: output = (x @ A.T @ B.T) * scaling)
    lora_out = torch.mm(torch.mm(x, lora_a.t()), lora_b.t()) * scaling

    # 完整输出
    output = base_out + lora_out

    print(f"\n参数配置:")
    print(f"  batch: {batch}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  rank: {rank}")
    print(f"  alpha: {alpha}")
    print(f"  scaling (alpha/rank): {scaling}")

    print(f"\n输入/权重形状:")
    print(f"  输入 x: {x.shape}")
    print(f"  基础权重 W: {W.shape}")
    print(f"  LoRA A: {lora_a.shape}")
    print(f"  LoRA B: {lora_b.shape}")

    print(f"\n计算结果:")
    print(f"  基础输出: {base_out.shape}, mean={base_out.abs().mean():.4f}")
    print(f"  LoRA输出: {lora_out.shape}, mean={lora_out.abs().mean():.4f}")
    print(f"  最终输出: {output.shape}, mean={output.abs().mean():.4f}")
    print(f"  LoRA对输出的相对贡献: {(lora_out.abs().mean() / output.abs().mean() * 100):.2f}%")

    print("\n✓ LoRA计算公式验证完成")
    print("  公式: output = base_out + (x @ A.T @ B.T) * (alpha / rank)")


# ============================================================================
# 4. 验证DeepSeek-V2-Lite的模型配置
# ============================================================================

def verify_model_config():
    """打印DeepSeek-V2-Lite的关键配置"""
    print("\n" + "=" * 60)
    print("Step 4: DeepSeek-V2-Lite 模型配置参考")
    print("=" * 60)

    print("""
DeepSeek-V2-Lite 关键配置:
  - hidden_size: 2048
  - intermediate_size (MoE): 5632 (=1408*4, 由于num_experts_per_tok=6和moe_intermediate_size=1408)
  - num_experts: 64
  - num_experts_per_tok: 6
  - num_layers: 27 (layer 0没有MoE, layer 1-26有MoE)

启动命令中的LoRA配置:
  - lora_rank: 8
  - lora_alpha: 16.0
  - scaling: 16.0 / 8 = 2.0

注意: 新版本使用AMXBF16_SFT模式，基础权重从HuggingFace模型加载
""")


# ============================================================================
# 5. 对比建议
# ============================================================================

def print_debug_suggestions():
    """打印调试建议"""
    print("\n" + "=" * 60)
    print("Step 5: 调试建议")
    print("=" * 60)
    print("""
调试步骤:

1. 首先运行此脚本验证权重形状是否正确

2. 启动新版本服务，观察控制台输出的调试信息:
   - [DEBUG MoE LoRA] 显示LoRA权重加载信息
   - [DEBUG forward_sft] 显示forward_sft的输入输出

3. 如果forward_sft输出异常 (NaN, Inf, 或值很大):
   - 尝试隔离测试: 修改_load_moe_lora_weights()使用零LoRA权重
   - 如果零LoRA权重后正常，问题在LoRA计算
   - 如果仍然异常，问题在基础MoE权重加载

4. 对比新旧版本:
   - 旧版本: 只有Attention+Shared LoRA，routed experts无LoRA
   - 新版本: 加上routed experts LoRA
   - 在相同输入下对比输出

5. 检查专家ID映射:
   - topk_ids的范围是否正确 (0 到 num_experts-1)
   - physical_to_logical_map_cpu是否正确
""")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE LoRA调试脚本")
    parser.add_argument("--converted", type=str,
                       default="/mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model_converted.pt",
                       help="转换后的.pt文件路径")
    parser.add_argument("--peft", type=str,
                       default="/mnt/data/lpl/kernel_new_test_adapter/Kllama2_deepseekV2_WEST_ALL/checkpoint-133/adapter_model.safetensors",
                       help="原始PEFT adapter路径")
    args = parser.parse_args()

    print("=" * 60)
    print("MoE LoRA 调试脚本")
    print("=" * 60)

    # Step 1: 验证转换后的权重
    if os.path.exists(args.converted):
        verify_converted_weights(args.converted)
    else:
        print(f"\n警告: 找不到转换后的文件 {args.converted}")
        print("请先运行转换脚本:")
        print("  python scripts/convert_moe_lora.py --input /path/to/adapter.safetensors --output /path/to/converted.pt")

    # Step 2: 验证原始PEFT权重
    if os.path.exists(args.peft):
        verify_peft_weights(args.peft)
    else:
        print(f"\n警告: 找不到PEFT文件 {args.peft}")

    # Step 3: 测试LoRA计算
    test_lora_computation()

    # Step 4: 模型配置参考
    verify_model_config()

    # Step 5: 调试建议
    print_debug_suggestions()

    print("\n" + "=" * 60)
    print("调试脚本完成")
    print("=" * 60)
