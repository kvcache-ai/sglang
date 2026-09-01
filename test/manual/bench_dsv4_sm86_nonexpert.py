"""Synthetic A/B benchmark for DeepSeek V4 SM86 decode fast paths.

Run on an otherwise idle CUDA GPU.  The candidate kernels are BF16-only and
can be timed on SM89 as an algorithmic proxy; those numbers are not SM86
performance measurements.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from collections.abc import Callable

import torch

from sglang.srt.layers.attention.compressed.indexer import (
    bf16_direct_paged_mqa_logits_tilelang,
)
from sglang.srt.layers.attention.compressed.sm86_indexer import (
    bf16_direct_paged_mqa_logits_triton,
)
from sglang.srt.layers.attention.nsa.v4_triton_kernel import (
    decode_sparse_attention_bf16,
    decode_sparse_attention_bf16_legacy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--require-speedup", action="store_true")
    return parser.parse_args()


def time_us(call: Callable[[], object], args: argparse.Namespace) -> float:
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / args.repetitions


def bootstrap_median_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(20260901)
    medians = []
    for _ in range(10_000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return medians[249], medians[9749]


def paired_benchmark(
    label: str,
    legacy: Callable[[], object],
    candidate: Callable[[], object],
    args: argparse.Namespace,
) -> bool:
    old_times: list[float] = []
    new_times: list[float] = []
    legacy()
    candidate()
    torch.cuda.synchronize()
    for round_id in range(args.rounds):
        if round_id % 2:
            new_times.append(time_us(candidate, args))
            old_times.append(time_us(legacy, args))
        else:
            old_times.append(time_us(legacy, args))
            new_times.append(time_us(candidate, args))

    speedups = [old / new for old, new in zip(old_times, new_times)]
    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)
    ci_low, ci_high = bootstrap_median_ci(speedups)
    print(
        f"{label}: legacy={old_median:.3f} us candidate={new_median:.3f} us "
        f"speedup={old_median / new_median:.3f}x "
        f"paired_median_ci95=[{ci_low:.3f},{ci_high:.3f}]"
    )
    return ci_low > 1.0


def benchmark_indexer(seq_len: int, args: argparse.Namespace) -> bool:
    device = torch.device("cuda")
    num_pages = math.ceil(seq_len / 64)
    torch.manual_seed(100 + seq_len)
    query = (torch.randn(1, 1, 64, 128, device=device) * 0.1).to(torch.bfloat16)
    cache = (torch.randn(num_pages, 64, 128, device=device) * 0.1).to(torch.bfloat16)
    weights = torch.randn(1, 64, dtype=torch.float32, device=device)
    lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)

    return paired_benchmark(
        f"indexer-{seq_len}",
        lambda: bf16_direct_paged_mqa_logits_tilelang(
            query,
            cache,
            weights,
            lengths,
            page_table,
            None,
            seq_len,
            False,
        ),
        lambda: bf16_direct_paged_mqa_logits_triton(
            query,
            cache,
            weights,
            lengths,
            page_table,
            None,
            seq_len,
            False,
        ),
        args,
    )


def benchmark_c4_sparse(args: argparse.Namespace) -> bool:
    device = torch.device("cuda")
    torch.manual_seed(712)
    query = (torch.randn(1, 64, 512, device=device) * 0.05).to(torch.bfloat16)
    swa_cache = (torch.randn(1, 128, 512, device=device) * 0.05).to(torch.bfloat16)
    extra_cache = (torch.randn(8, 64, 512, device=device) * 0.05).to(torch.bfloat16)
    swa_indices = torch.arange(128, dtype=torch.int32, device=device).unsqueeze(0)
    extra_indices = torch.arange(512, dtype=torch.int32, device=device).unsqueeze(0)
    swa_lens = torch.tensor([128], dtype=torch.int32, device=device)
    extra_lens = torch.tensor([512], dtype=torch.int32, device=device)
    sink = torch.randn(64, dtype=torch.float32, device=device) * 0.1
    output = torch.empty_like(query)
    scale = 1.0 / math.sqrt(512)
    common = (
        query,
        swa_cache,
        swa_indices,
        swa_lens,
        scale,
        sink,
        output,
        extra_cache,
        extra_indices,
        extra_lens,
    )

    return paired_benchmark(
        "sparse-c4-640",
        lambda: decode_sparse_attention_bf16_legacy(*common),
        lambda: decode_sparse_attention_bf16(*common),
        args,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    print(torch.cuda.get_device_name())
    passed = [
        benchmark_indexer(4096, args),
        benchmark_indexer(16384, args),
        benchmark_c4_sparse(args),
    ]
    if args.require_speedup and not all(passed):
        raise SystemExit("at least one candidate lacks a positive 95% CI")


if __name__ == "__main__":
    main()
