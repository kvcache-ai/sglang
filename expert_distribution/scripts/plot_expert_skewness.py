#!/usr/bin/env python3
"""
绘制专家激活分布的 skewness 可视化图

用法:
    # 图1: 各层 Top-16 专家激活次数之和
    python plot_expert_skewness.py <pt_file> --mode summary

    # 图2: 特定层的详细分布
    python plot_expert_skewness.py <pt_file> --mode layer --layer 0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_top16_summary(total_count: np.ndarray, output_path: Path):
    """
    图1: 横坐标 layer idx，纵坐标各层 Top-16 专家激活次数之和
    """
    num_layers = total_count.shape[0]
    top16_sums = []

    for layer_idx in range(num_layers):
        layer_counts = total_count[layer_idx]
        top16 = np.sort(layer_counts)[-16:]  # 取最大的16个
        top16_sums.append(top16.sum())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(num_layers), top16_sums, color='steelblue', alpha=0.8)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Sum of Top-16 Expert Activations')
    ax.set_title('Top-16 Expert Activation Sum per Layer')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = output_path / 'top16_per_layer.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()


def plot_layer_detail(layer_counts: np.ndarray, layer_idx: int, output_path: Path):
    """
    图2: 特定层的三合一图
    - 左: 按 Expert ID 排列的激活次数（不排序）
    - 中: 按激活次数降序排列
    - 右: 激活次数直方图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    num_experts = len(layer_counts)

    # 左图: 按 Expert ID 排列（不排序）
    ax = axes[0]
    ax.bar(range(num_experts), layer_counts, width=1.0, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(layer_counts), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(layer_counts):.1f}')
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Activation Count')
    ax.set_title(f'Layer {layer_idx}: Activation by Expert ID')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # 中图: 按激活次数降序排列
    ax = axes[1]
    sorted_counts = np.sort(layer_counts)[::-1]
    ax.bar(range(num_experts), sorted_counts, width=1.0, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(layer_counts), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(layer_counts):.1f}')

    # 标注 Top-16 专家贡献
    top16_sum = sorted_counts[:16].sum()
    total_sum = sorted_counts.sum()
    top16_pct = top16_sum / total_sum * 100 if total_sum > 0 else 0
    ax.axvline(x=16, color='orange', linestyle=':', linewidth=2)
    ax.text(18, sorted_counts.max() * 0.8,
            f'Top-16 experts\ncontribute {top16_pct:.1f}%',
            fontsize=10, color='orange')

    ax.set_xlabel('Expert Rank (sorted by activation)')
    ax.set_ylabel('Activation Count')
    ax.set_title(f'Layer {layer_idx}: Sorted Activation Distribution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # 右图: 激活次数直方图
    ax = axes[2]
    nonzero_counts = layer_counts[layer_counts > 0]
    zero_count = np.sum(layer_counts == 0)

    if len(nonzero_counts) > 0:
        ax.hist(nonzero_counts, bins=30, color='steelblue', alpha=0.7, edgecolor='white')

    mean_val = np.mean(layer_counts)
    std_val = np.std(layer_counts)

    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_val:.1f}')
    ax.axvline(x=np.median(layer_counts), color='green', linestyle=':', linewidth=2,
               label=f'Median = {np.median(layer_counts):.1f}')

    ax.set_xlabel('Activation Count')
    ax.set_ylabel('Number of Experts')
    ax.set_title(f'Layer {layer_idx}: Activation Distribution\n'
                 f'Std = {std_val:.1f}, Zero-activation = {zero_count}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = output_path / f'layer_{layer_idx}_detail.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存: {save_path}")
    plt.close()


def find_max_var_layer(total_count: np.ndarray) -> int:
    """找到方差最大的层"""
    variances = [np.var(total_count[i]) for i in range(total_count.shape[0])]
    return int(np.argmax(variances))


def find_min_var_layer(total_count: np.ndarray) -> int:
    """找到方差最小的层"""
    variances = [np.var(total_count[i]) for i in range(total_count.shape[0])]
    return int(np.argmin(variances))


def main():
    parser = argparse.ArgumentParser(description="绘制专家激活分布图")
    parser.add_argument("pt_file", type=str, help=".pt 文件路径")
    parser.add_argument("--mode", type=str, choices=['summary', 'layer', 'max_var', 'min_var'],
                        default='summary',
                        help="summary: Top-16汇总图; layer: 特定层详细图; max_var: 方差最大层; min_var: 方差最小层")
    parser.add_argument("--layer", type=int, default=0, help="--mode layer 时指定层号")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        print(f"错误: 文件不存在 - {pt_path}")
        return 1

    output_path = Path(args.output) if args.output else pt_path.parent
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"加载文件: {pt_path}")
    data = torch.load(pt_path, weights_only=False, map_location="cpu")

    if "logical_count" not in data:
        print("错误: 文件不是 stat 模式，暂不支持")
        return 1

    logical_count = data["logical_count"].numpy()
    total_count = logical_count.sum(axis=0)  # (num_layers, num_experts)
    num_layers = total_count.shape[0]

    print(f"层数: {num_layers}, 专家数: {total_count.shape[1]}")

    if args.mode == 'summary':
        plot_top16_summary(total_count, output_path)
    elif args.mode == 'layer':
        if 0 <= args.layer < num_layers:
            plot_layer_detail(total_count[args.layer], args.layer, output_path)
        else:
            print(f"错误: layer {args.layer} 超出范围 [0, {num_layers - 1}]")
            return 1
    elif args.mode == 'max_var':
        layer_idx = find_max_var_layer(total_count)
        var_val = np.var(total_count[layer_idx])
        print(f"方差最大的层: Layer {layer_idx} (variance = {var_val:.2f})")
        plot_layer_detail(total_count[layer_idx], layer_idx, output_path)
    elif args.mode == 'min_var':
        layer_idx = find_min_var_layer(total_count)
        var_val = np.var(total_count[layer_idx])
        print(f"方差最小的层: Layer {layer_idx} (variance = {var_val:.2f})")
        plot_layer_detail(total_count[layer_idx], layer_idx, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
