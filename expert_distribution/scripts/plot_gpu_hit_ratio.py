#!/usr/bin/env python3
"""Plot GPU expert hit ratio visualizations.

Three plot types:
1. layer-over-time: Hit ratio curve for a single layer across forward passes
2. layers-single-pass: Hit ratio across all layers for a single forward pass
3. heatmap: Heatmap of hit ratios across forward passes and layers
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np


def load_hit_ratio(path: str) -> torch.Tensor:
    """Load GPU hit ratio tensor from file."""
    data = torch.load(path, weights_only=True, map_location="cpu")
    return data["gpu_hit_ratio"]  # [forward_pass, num_layer]


def plot_layer_over_time(
    ratio: torch.Tensor,
    layer_idx: int,
    start_pass: int,
    end_pass: int,
    output_path: str,
):
    """Plot hit ratio curve for a single layer across forward passes."""
    # Extract data
    data = ratio[start_pass:end_pass, layer_idx].numpy()
    x = np.arange(start_pass, end_pass)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, data, linewidth=1)
    ax.set_xlabel("Forward Pass")
    ax.set_ylabel("GPU Hit Ratio")
    ax.set_title(f"GPU Hit Ratio for Layer {layer_idx} (Forward Pass {start_pass}-{end_pass})")
    ax.set_xlim(start_pass, end_pass - 1)
    ax.set_ylim(0, max(data.max() * 1.1, 0.1))  # At least 0.1 for visibility
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_val = data.mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved layer-over-time plot to {output_path}")


def plot_layers_single_pass(
    ratio: torch.Tensor,
    forward_pass: int,
    output_path: str,
):
    """Plot hit ratio across all layers for a single forward pass."""
    # Extract data
    data = ratio[forward_pass, :].numpy()
    x = np.arange(len(data))

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, data, width=0.8)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("GPU Hit Ratio")
    ax.set_title(f"GPU Hit Ratio per Layer (Forward Pass {forward_pass})")
    ax.set_xlim(-0.5, len(data) - 0.5)
    ax.set_ylim(0, max(data.max() * 1.1, 0.1))
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_val = data.mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved layers-single-pass plot to {output_path}")


def plot_heatmap(
    ratio: torch.Tensor,
    start_pass: int,
    end_pass: int,
    output_path: str,
):
    """Plot heatmap of hit ratios across forward passes and layers."""
    # Extract data: [forward_pass, num_layer]
    data = ratio[start_pass:end_pass, :].numpy()
    num_passes, num_layers = data.shape

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use 0 to max hit rate as the color scale
    vmax = data.max()
    if vmax == 0:
        vmax = 1.0  # Avoid division by zero in colorbar

    im = ax.imshow(
        data.T,  # Transpose so layers are on y-axis
        aspect='auto',
        cmap='hot',
        vmin=0,
        vmax=vmax,
        origin='lower',
        extent=[start_pass, end_pass, 0, num_layers],
    )

    ax.set_xlabel("Forward Pass")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"GPU Hit Ratio Heatmap (Forward Pass {start_pass}-{end_pass}, Max={vmax:.4f})")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("GPU Hit Ratio")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved heatmap plot to {output_path}")
    print(f"  Data range: 0 to {vmax:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPU expert hit ratio visualizations"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to GPU hit ratio .pt file (output from compute_gpu_hit_ratio.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output image file (e.g., plot.png)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        required=True,
        choices=["layer-over-time", "layers-single-pass", "heatmap"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Layer index (required for layer-over-time)",
    )
    parser.add_argument(
        "--forward-pass",
        type=int,
        help="Single forward pass index (required for layers-single-pass)",
    )
    parser.add_argument(
        "--start-pass",
        type=int,
        help="Start forward pass index (required for layer-over-time and heatmap)",
    )
    parser.add_argument(
        "--end-pass",
        type=int,
        help="End forward pass index (required for layer-over-time and heatmap)",
    )

    args = parser.parse_args()

    # Load data
    ratio = load_hit_ratio(args.input)
    num_passes, num_layers = ratio.shape
    print(f"Loaded data: {num_passes} forward passes, {num_layers} layers")

    # Validate and generate plot
    if args.plot_type == "layer-over-time":
        if args.layer is None or args.start_pass is None or args.end_pass is None:
            parser.error("layer-over-time requires --layer, --start-pass, and --end-pass")
        if not (0 <= args.layer < num_layers):
            parser.error(f"layer must be in [0, {num_layers})")
        if not (0 <= args.start_pass < args.end_pass <= num_passes):
            parser.error(f"start-pass and end-pass must be in [0, {num_passes}]")
        plot_layer_over_time(ratio, args.layer, args.start_pass, args.end_pass, args.output)

    elif args.plot_type == "layers-single-pass":
        if args.forward_pass is None:
            parser.error("layers-single-pass requires --forward-pass")
        if not (0 <= args.forward_pass < num_passes):
            parser.error(f"forward-pass must be in [0, {num_passes})")
        plot_layers_single_pass(ratio, args.forward_pass, args.output)

    elif args.plot_type == "heatmap":
        if args.start_pass is None or args.end_pass is None:
            parser.error("heatmap requires --start-pass and --end-pass")
        if not (0 <= args.start_pass < args.end_pass <= num_passes):
            parser.error(f"start-pass and end-pass must be in [0, {num_passes}]")
        plot_heatmap(ratio, args.start_pass, args.end_pass, args.output)


if __name__ == "__main__":
    main()
