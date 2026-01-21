#!/usr/bin/env python3
"""
Sweep CPU experts count and measure latency.

This script tests the latency of KTEPWrapperMethod.apply() as the number of
CPU experts varies from 0 to top_k (with GPU experts correspondingly varying
from top_k to 0).

Usage:
    python sweep_cpu_experts.py \
        --model /mnt/data/models/Qwen3-Next-80B-A3B-Instruct-FP8 \
        --kt-weight-path /mnt/data/models/Qwen3-Next-80B-A3B-Instruct-FP8 \
        --num-tokens 1 \
        --kt-method FP8 \
        --cuda-graph \
        --num-layers 5
"""

import argparse
import subprocess
import sys
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep CPU experts and plot latency")

    # Pass-through arguments for benchmark_kt_ep.py
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--kt-weight-path", type=str, required=True,
                        help="Path to KT CPU quantized weights")
    parser.add_argument("--kt-cpuinfer", type=int, default=60,
                        help="Number of CPU inference threads")
    parser.add_argument("--kt-threadpool-count", type=int, default=2,
                        help="Number of thread pools")
    parser.add_argument("--kt-method", type=str, default="FP8",
                        help="CPU computation method")
    parser.add_argument("--num-tokens", type=int, default=1,
                        help="Number of input tokens")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Enable CUDA graph mode")
    parser.add_argument("--warmup-iters", type=int, default=10,
                        help="Warmup iterations")
    parser.add_argument("--bench-iters", type=int, default=100,
                        help="Benchmark iterations")
    parser.add_argument("--output", type=str, default="cpu_experts_latency.png",
                        help="Output plot filename")
    parser.add_argument("--num-layers", type=int, default=5,
                        help="Number of layers to benchmark and average (reduces caching effects)")
    parser.add_argument("--throughput-mode", action="store_true",
                        help="Use throughput mode (no per-iteration sync, higher CPU utilization)")
    parser.add_argument("--zero-cpu-slots", action="store_true",
                        help="Force all slots to GPU (CPU receives 0 slots, measures hybrid overhead)")
    parser.add_argument("--zero-gpu-slots", action="store_true",
                        help="Force all slots to CPU (GPU receives 0 slots, measures hybrid overhead)")

    return parser.parse_args()


def get_model_top_k(model_path: str) -> int:
    """Get top_k from model config."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    top_k = getattr(config, "num_experts_per_tok", None)
    if top_k is None:
        top_k = getattr(config, "top_k", None)
    if top_k is None:
        top_k = 2  # Default

    return top_k


def run_benchmark(
    model: str,
    kt_weight_path: str,
    kt_cpuinfer: int,
    kt_threadpool_count: int,
    kt_method: str,
    num_tokens: int,
    gpu_slots: int,
    gpu_experts_active: int,
    cpu_experts_active: int,
    cuda_graph: bool,
    warmup_iters: int,
    bench_iters: int,
    num_layers: int,
    top_k: int,
    throughput_mode: bool = False,
    zero_cpu_slots: bool = False,
    zero_gpu_slots: bool = False,
    override_top_k: int = None,
) -> Tuple[float, float]:
    """Run benchmark and return (mean_latency, std_latency) in ms."""
    # Need at least 1 GPU expert for the benchmark to work
    kt_num_gpu_experts = max(1, gpu_experts_active)

    cmd = [
        sys.executable,
        "/mnt/data/djw/experts_sched/sglang/python/sglang/srt/layers/moe/benchmark_kt_ep.py",
        "--model", model,
        "--kt-weight-path", kt_weight_path,
        "--kt-cpuinfer", str(kt_cpuinfer),
        "--kt-threadpool-count", str(kt_threadpool_count),
        "--kt-method", kt_method,
        "--kt-num-gpu-experts", str(kt_num_gpu_experts),
        "--num-tokens", str(num_tokens),
        "--gpu-slots", str(gpu_slots),
        "--gpu-experts-active", str(max(1, gpu_experts_active)),  # At least 1
        "--cpu-experts-active", str(max(1, cpu_experts_active)),  # At least 1
        "--warmup-iters", str(warmup_iters),
        "--bench-iters", str(bench_iters),
        "--num-layers", str(num_layers),
    ]

    if cuda_graph:
        cmd.append("--cuda-graph")

    if throughput_mode:
        cmd.append("--throughput-mode")

    if zero_cpu_slots:
        cmd.append("--zero-cpu-slots")

    if zero_gpu_slots:
        cmd.append("--zero-gpu-slots")

    if override_top_k is not None:
        cmd.extend(["--override-top-k", str(override_top_k)])

    # Calculate effective slots for display
    effective_top_k = override_top_k if override_top_k is not None else top_k
    if zero_gpu_slots:
        display_gpu_slots = 0
        display_cpu_slots = effective_top_k * num_tokens
    elif zero_cpu_slots:
        display_gpu_slots = effective_top_k * num_tokens
        display_cpu_slots = 0
    else:
        display_gpu_slots = gpu_slots
        display_cpu_slots = top_k * num_tokens - gpu_slots

    print(f"Running: gpu_slots={display_gpu_slots}, cpu_slots={display_cpu_slots}, "
          f"gpu_experts_active={gpu_experts_active}, cpu_experts_active={cpu_experts_active}, "
          f"num_layers={num_layers}" + (f", override_top_k={override_top_k}" if override_top_k else ""))

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output to get mean and std latency
    output = result.stdout + result.stderr

    if cuda_graph:
        if throughput_mode:
            # Throughput mode: look for "Avg latency:  X.XXX ms"
            mean_match = re.search(r"Avg latency:\s+([\d.]+)\s*ms", output)
            std_match = None  # Throughput mode doesn't report std
        else:
            # Look for "Mean:  X.XXX ms" and "Std:   X.XXX ms"
            mean_match = re.search(r"Mean:\s+([\d.]+)\s*ms", output)
            std_match = re.search(r"Std:\s+([\d.]+)\s*ms", output)
    else:
        # Look for "Total apply():  X.XXX ms (std: X.XXX)"
        mean_match = re.search(r"Total apply\(\):\s+([\d.]+)\s*ms", output)
        std_match = re.search(r"Total apply\(\):\s+[\d.]+\s*ms\s+\(std:\s+([\d.]+)", output)

    mean_latency = float(mean_match.group(1)) if mean_match else float('nan')
    std_latency = float(std_match.group(1)) if std_match else 0.0

    if mean_match is None:
        print(f"Warning: Could not parse latency from output")
        print(output[-2000:])  # Print last 2000 chars for debugging

    return mean_latency, std_latency


def main():
    args = parse_args()

    # Get top_k from model config
    print(f"Loading model config from {args.model}...")
    top_k = get_model_top_k(args.model)
    print(f"Model top_k = {top_k}")
    print(f"Alternating {args.num_layers} layers per iteration to reduce caching effects")

    total_slots = args.num_tokens * top_k

    # Sweep from 0 CPU experts to top_k CPU experts
    # Each slot goes to a different expert
    results: List[Tuple[int, float, float]] = []  # (cpu_experts, mean_latency, std_latency)

    for cpu_experts in range(0, top_k + 1):
        gpu_experts = top_k - cpu_experts
        cpu_slots = cpu_experts  # 1 slot per CPU expert
        gpu_slots = gpu_experts  # 1 slot per GPU expert

        # Determine override_top_k for zero-slots modes
        # For --zero-gpu-slots: all slots go to CPU, so override_top_k = cpu_experts (number of CPU slots)
        # For --zero-cpu-slots: all slots go to GPU, so override_top_k = gpu_experts (number of GPU slots)
        if args.zero_gpu_slots:
            override_top_k = cpu_experts if cpu_experts > 0 else None
        elif args.zero_cpu_slots:
            override_top_k = gpu_experts if gpu_experts > 0 else None
        else:
            override_top_k = None

        # Skip if override_top_k would be 0 (can't have 0 slots)
        if (args.zero_gpu_slots or args.zero_cpu_slots) and override_top_k is None:
            print(f"Skipping cpu_experts={cpu_experts}: would result in 0 slots")
            results.append((cpu_experts, float('nan'), 0.0))
            continue

        # Run benchmark with multi-layer alternation
        mean_latency, std_latency = run_benchmark(
            model=args.model,
            kt_weight_path=args.kt_weight_path,
            kt_cpuinfer=args.kt_cpuinfer,
            kt_threadpool_count=args.kt_threadpool_count,
            kt_method=args.kt_method,
            num_tokens=args.num_tokens,
            gpu_slots=gpu_slots,
            gpu_experts_active=gpu_experts,
            cpu_experts_active=cpu_experts,
            cuda_graph=args.cuda_graph,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            num_layers=args.num_layers,
            top_k=top_k,
            throughput_mode=args.throughput_mode,
            zero_cpu_slots=args.zero_cpu_slots,
            zero_gpu_slots=args.zero_gpu_slots,
            override_top_k=override_top_k,
        )

        results.append((cpu_experts, mean_latency, std_latency))
        print(f"  => CPU experts: {cpu_experts}, GPU experts: {gpu_experts}, "
              f"Latency: {mean_latency:.3f} ms (std: {std_latency:.3f} ms)")

    # Plot results
    cpu_experts_list = [r[0] for r in results]
    latencies = [r[1] for r in results]
    std_devs = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(cpu_experts_list, latencies, yerr=std_devs, fmt='b-o',
                 linewidth=2, markersize=8, capsize=4, capthick=2)
    plt.xlabel('CPU Experts (out of top_k)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    mode_parts = []
    if args.throughput_mode:
        mode_parts.append("throughput")
    if args.zero_cpu_slots:
        mode_parts.append("zero-cpu-slots")
    if args.zero_gpu_slots:
        mode_parts.append("zero-gpu-slots")
    mode_str = ", " + "+".join(mode_parts) if mode_parts else ""
    plt.title(f'MoE Latency vs CPU Expert Count\n'
              f'(num_tokens={args.num_tokens}, top_k={top_k}, method={args.kt_method}, '
              f'alternating {args.num_layers} layers{mode_str})',
              fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, top_k + 1))

    # Add data labels
    for x, y, std in zip(cpu_experts_list, latencies, std_devs):
        if not (y != y):  # Check for NaN
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 65)
    mode_parts = []
    if args.throughput_mode:
        mode_parts.append("throughput")
    if args.zero_cpu_slots:
        mode_parts.append("zero-cpu-slots")
    if args.zero_gpu_slots:
        mode_parts.append("zero-gpu-slots")
    mode_info = ", " + "+".join(mode_parts) if mode_parts else ""
    print(f"Summary (alternating {args.num_layers} layers per iteration{mode_info})")
    print("=" * 65)
    print(f"{'CPU Experts':<15} {'GPU Experts':<15} {'Latency (ms)':<15} {'Std (ms)':<15}")
    print("-" * 65)
    for cpu_exp, latency, std in results:
        gpu_exp = top_k - cpu_exp
        print(f"{cpu_exp:<15} {gpu_exp:<15} {latency:<15.3f} {std:<15.3f}")


if __name__ == "__main__":
    main()
