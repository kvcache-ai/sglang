#!/usr/bin/env python3
"""
对比 Transformers 和 KT 的每层输出差异
"""

import argparse
import pickle
import numpy as np
import os


def load_dumps(transformers_dump, kt_dump):
    """加载两个dump文件"""
    print(f"Loading Transformers dump from: {transformers_dump}")
    with open(transformers_dump, 'rb') as f:
        transformers_outputs = pickle.load(f)
    print(f"  Loaded {len(transformers_outputs)} layers")

    print(f"Loading KT dump from: {kt_dump}")
    with open(kt_dump, 'rb') as f:
        kt_outputs = pickle.load(f)
    print(f"  Loaded {len(kt_outputs)} layers")

    return transformers_outputs, kt_outputs


def compare_layer(name, transformers_output, kt_output):
    """对比单层输出"""
    # 计算差异
    abs_diff = np.abs(transformers_output - kt_output)
    rel_diff = abs_diff / (np.abs(transformers_output) + 1e-8)

    # 统计信息
    stats = {
        "name": name,
        "shape": transformers_output.shape,
        "transformers_mean": transformers_output.mean(),
        "transformers_std": transformers_output.std(),
        "transformers_min": transformers_output.min(),
        "transformers_max": transformers_output.max(),
        "kt_mean": kt_output.mean(),
        "kt_std": kt_output.std(),
        "kt_min": kt_output.min(),
        "kt_max": kt_output.max(),
        "abs_diff_mean": abs_diff.mean(),
        "abs_diff_max": abs_diff.max(),
        "abs_diff_std": abs_diff.std(),
        "rel_diff_mean": rel_diff.mean(),
        "rel_diff_max": rel_diff.max(),
        "cosine_similarity": compute_cosine_similarity(transformers_output, kt_output),
        "mse": np.mean((transformers_output - kt_output) ** 2),
    }

    return stats, abs_diff, rel_diff


def compute_cosine_similarity(a, b):
    """计算余弦相似度"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def print_comparison(stats_list):
    """打印对比结果"""
    print("\n" + "=" * 120)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 120)
    print(f"{'Layer':<20} {'Shape':<20} {'Trans Mean':<12} {'KT Mean':<12} {'Abs Diff':<12} {'Rel Diff':<12} {'Cos Sim':<10} {'MSE':<12}")
    print("-" * 120)

    for stats in stats_list:
        print(f"{stats['name']:<20} "
              f"{str(stats['shape']):<20} "
              f"{stats['transformers_mean']:>11.4e} "
              f"{stats['kt_mean']:>11.4e} "
              f"{stats['abs_diff_mean']:>11.4e} "
              f"{stats['rel_diff_mean']:>11.4e} "
              f"{stats['cosine_similarity']:>9.6f} "
              f"{stats['mse']:>11.4e}")

    print("=" * 120)


def print_detailed_stats(stats_list):
    """打印详细统计信息"""
    print("\n" + "=" * 120)
    print("DETAILED STATISTICS")
    print("=" * 120)

    for stats in stats_list:
        print(f"\n[{stats['name']}]")
        print(f"  Shape: {stats['shape']}")
        print(f"  Transformers: mean={stats['transformers_mean']:.6e}, std={stats['transformers_std']:.6e}, "
              f"min={stats['transformers_min']:.6e}, max={stats['transformers_max']:.6e}")
        print(f"  KT:           mean={stats['kt_mean']:.6e}, std={stats['kt_std']:.6e}, "
              f"min={stats['kt_min']:.6e}, max={stats['kt_max']:.6e}")
        print(f"  Abs Diff:     mean={stats['abs_diff_mean']:.6e}, std={stats['abs_diff_std']:.6e}, max={stats['abs_diff_max']:.6e}")
        print(f"  Rel Diff:     mean={stats['rel_diff_mean']:.6e}, max={stats['rel_diff_max']:.6e}")
        print(f"  Cosine Similarity: {stats['cosine_similarity']:.8f}")
        print(f"  MSE: {stats['mse']:.6e}")


def find_problematic_layers(stats_list, threshold=0.01):
    """找出差异较大的层"""
    print(f"\n" + "=" * 120)
    print(f"PROBLEMATIC LAYERS (abs_diff_mean > {threshold})")
    print("=" * 120)

    problematic = [s for s in stats_list if s['abs_diff_mean'] > threshold]

    if not problematic:
        print("  No problematic layers found!")
    else:
        for stats in problematic:
            print(f"  {stats['name']:<20} abs_diff_mean={stats['abs_diff_mean']:.6e}, "
                  f"rel_diff_mean={stats['rel_diff_mean']:.6e}, cos_sim={stats['cosine_similarity']:.6f}")


def analyze_moe_layers(stats_list):
    """分析MoE层的差异"""
    print(f"\n" + "=" * 120)
    print("MoE LAYER ANALYSIS")
    print("=" * 120)

    moe_stats = [s for s in stats_list if '_moe' in s['name']]

    if not moe_stats:
        print("  No MoE layers found!")
        return

    print(f"  Total MoE layers: {len(moe_stats)}")
    print(f"\n  {'Layer':<20} {'Abs Diff Mean':<15} {'Rel Diff Mean':<15} {'Cosine Sim':<12}")
    print("  " + "-" * 70)

    for stats in moe_stats:
        print(f"  {stats['name']:<20} {stats['abs_diff_mean']:>14.6e} {stats['rel_diff_mean']:>14.6e} {stats['cosine_similarity']:>11.6f}")


def save_summary(stats_list, output_file):
    """保存对比摘要"""
    with open(output_file, 'w') as f:
        f.write("Layer Comparison Summary\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Layer':<20} {'Shape':<20} {'Trans Mean':<12} {'KT Mean':<12} {'Abs Diff':<12} {'Rel Diff':<12} {'Cos Sim':<10} {'MSE':<12}\n")
        f.write("-" * 120 + "\n")

        for stats in stats_list:
            f.write(f"{stats['name']:<20} "
                   f"{str(stats['shape']):<20} "
                   f"{stats['transformers_mean']:>11.4e} "
                   f"{stats['kt_mean']:>11.4e} "
                   f"{stats['abs_diff_mean']:>11.4e} "
                   f"{stats['rel_diff_mean']:>11.4e} "
                   f"{stats['cosine_similarity']:>9.6f} "
                   f"{stats['mse']:>11.4e}\n")

        f.write("=" * 120 + "\n")

    print(f"\n[SAVE] Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare layer dumps")
    parser.add_argument(
        "--transformers-dump",
        type=str,
        default="/tmp/transformers_dumps/transformers_lora_outputs.pkl",
        help="Path to Transformers dump file",
    )
    parser.add_argument(
        "--kt-dump",
        type=str,
        default="/tmp/kt_dumps/kt_lora_outputs.pkl",
        help="Path to KT dump file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="layer_comparison_summary.txt",
        help="Output file for summary",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Threshold for problematic layers",
    )

    args = parser.parse_args()

    print("=" * 120)
    print("LAYER OUTPUT COMPARISON")
    print("=" * 120)

    # 加载dumps
    transformers_outputs, kt_outputs = load_dumps(args.transformers_dump, args.kt_dump)

    # 找到共同的层
    common_layers = sorted(set(transformers_outputs.keys()) & set(kt_outputs.keys()))
    print(f"\nCommon layers: {len(common_layers)}")

    if not common_layers:
        print("ERROR: No common layers found!")
        return

    # 对比每一层
    stats_list = []
    for layer_name in common_layers:
        transformers_output = transformers_outputs[layer_name]
        kt_output = kt_outputs[layer_name]

        # 检查shape是否匹配
        if transformers_output.shape != kt_output.shape:
            print(f"WARNING: Shape mismatch for {layer_name}: "
                  f"{transformers_output.shape} vs {kt_output.shape}")
            continue

        stats, abs_diff, rel_diff = compare_layer(layer_name, transformers_output, kt_output)
        stats_list.append(stats)

    # 打印结果
    print_comparison(stats_list)
    find_problematic_layers(stats_list, threshold=args.threshold)
    analyze_moe_layers(stats_list)
    print_detailed_stats(stats_list)

    # 保存摘要
    save_summary(stats_list, args.output)

    print("\n[DONE] Comparison complete!")


if __name__ == "__main__":
    main()
