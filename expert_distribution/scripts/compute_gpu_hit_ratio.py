#!/usr/bin/env python3
"""Compute GPU expert hit ratio from expert distribution and GPU expert mask files.

This script calculates the ratio of activated experts that are on GPU for each
forward pass and each layer.

Output tensor shape: (forward_pass, num_layer)
- Value at [i, j] = (activated experts on GPU at layer j in pass i) / (total activated experts at layer j in pass i)
"""

import argparse
import sys
import torch


def compute_gpu_hit_ratio(
    expert_dist_path: str,
    gpu_mask_path: str,
) -> torch.Tensor:
    """Compute GPU expert hit ratio.

    Args:
        expert_dist_path: Path to expert distribution .pt file (contains logical_count)
        gpu_mask_path: Path to GPU expert mask .pt file (contains gpu_expert_masks)

    Returns:
        Tensor of shape (forward_pass, num_layer) with hit ratios
    """
    # Load data and move to CPU for processing
    expert_dist = torch.load(expert_dist_path, weights_only=True, map_location="cpu")
    gpu_mask_data = torch.load(gpu_mask_path, weights_only=True, map_location="cpu")

    logical_count = expert_dist["logical_count"].cpu()  # [forward_pass, num_layer, num_experts]
    gpu_masks = gpu_mask_data["gpu_expert_masks"].cpu()  # [forward_pass, num_layer, num_experts]

    # Ensure same shape
    assert logical_count.shape == gpu_masks.shape, (
        f"Shape mismatch: logical_count {logical_count.shape} vs gpu_masks {gpu_masks.shape}"
    )

    forward_pass, num_layer, num_experts = logical_count.shape

    # Find activated experts (count > 0)
    activated = logical_count > 0  # [forward_pass, num_layer, num_experts], bool

    # Find activated experts that are on GPU
    activated_on_gpu = activated & gpu_masks  # [forward_pass, num_layer, num_experts], bool

    # Count per layer
    total_activated = activated.sum(dim=2).float()  # [forward_pass, num_layer]
    activated_on_gpu_count = activated_on_gpu.sum(dim=2).float()  # [forward_pass, num_layer]

    # Compute ratio (avoid division by zero)
    ratio = torch.where(
        total_activated > 0,
        activated_on_gpu_count / total_activated,
        torch.zeros_like(total_activated),
    )

    return ratio


def print_stats(ratio: torch.Tensor, file=None):
    """Print statistics about the computed ratios.

    Args:
        ratio: Tensor of shape (forward_pass, num_layer) with hit ratios
        file: Output file object (default: stdout)
    """
    if file is None:
        file = sys.stdout

    num_passes, num_layers = ratio.shape

    print("\n[基本统计]", file=file)
    print(f"  Mean ratio: {ratio.mean().item():.4f}", file=file)
    print(f"  Min ratio: {ratio.min().item():.4f}", file=file)
    print(f"  Max ratio: {ratio.max().item():.4f}", file=file)
    print(f"  Std ratio: {ratio.std().item():.4f}", file=file)

    # Per-layer stats: mean and best forward pass
    print(f"\n[每层统计]", file=file)
    print(f"  {'Layer':<8} {'Mean':>10} {'Max':>10} {'Best Pass':>12}", file=file)
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12}", file=file)

    layer_mean = ratio.mean(dim=0)  # [num_layer]
    layer_max = ratio.max(dim=0)  # values and indices

    for layer_idx in range(num_layers):
        mean_val = layer_mean[layer_idx].item()
        max_val = layer_max.values[layer_idx].item()
        # Find the earliest forward pass with max hit ratio for this layer
        layer_ratios = ratio[:, layer_idx]
        best_pass_indices = torch.where(layer_ratios == max_val)[0]
        best_pass = best_pass_indices[0].item() if len(best_pass_indices) > 0 else -1
        print(f"  {layer_idx:<8} {mean_val:>10.4f} {max_val:>10.4f} {best_pass:>12}", file=file)

    # Top 10 forward passes by average hit ratio across all layers
    print(f"\n[平均命中率最高的 10 个 forward pass]", file=file)
    pass_mean = ratio.mean(dim=1)  # [num_passes]
    top_k = min(10, num_passes)
    top_values, top_indices = torch.topk(pass_mean, top_k)

    print(f"  {'Rank':<6} {'Pass':>8} {'Avg Hit Ratio':>15}", file=file)
    print(f"  {'-'*6} {'-'*8} {'-'*15}", file=file)
    for rank, (pass_idx, avg_ratio) in enumerate(zip(top_indices.tolist(), top_values.tolist()), 1):
        print(f"  {rank:<6} {pass_idx:>8} {avg_ratio:>15.4f}", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Compute GPU expert hit ratio from distribution files"
    )
    parser.add_argument(
        "--expert-dist",
        type=str,
        required=True,
        help="Path to expert distribution .pt file (contains logical_count)",
    )
    parser.add_argument(
        "--gpu-mask",
        type=str,
        required=True,
        help="Path to GPU expert mask .pt file (contains gpu_expert_masks)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .pt file",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print statistics about the computed ratios",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="Path to output statistics file (text format)",
    )

    args = parser.parse_args()

    # Compute ratio
    ratio = compute_gpu_hit_ratio(args.expert_dist, args.gpu_mask)

    # Save output
    torch.save({"gpu_hit_ratio": ratio}, args.output)
    print(f"Saved GPU hit ratio to {args.output}")
    print(f"  Shape: {ratio.shape} (forward_pass, num_layer)")

    # Print stats to stdout
    if args.print_stats:
        print_stats(ratio)

    # Save stats to file
    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as f:
            f.write(f"# GPU Hit Ratio Statistics\n")
            f.write(f"# Input: expert_dist={args.expert_dist}, gpu_mask={args.gpu_mask}\n")
            f.write(f"# Shape: {ratio.shape} (forward_pass, num_layer)\n")
            print_stats(ratio, file=f)
        print(f"Saved statistics to {args.stats_output}")


if __name__ == "__main__":
    main()
