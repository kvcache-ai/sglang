#!/usr/bin/env python3
"""Compare the current Triton W8A8 path with the upstream SM120 CUTLASS port."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    cutlass_w8a8_block_fp8_linear_with_fallback,
    triton_w8a8_block_fp8_linear,
)

SHAPES = (
    (512, 4096, 43),
    (1024, 4096, 43),
    (32768, 1024, 43),
    (4096, 8192, 43),
    (4096, 4096, 43),
    (4096, 2048, 43),
    (8192, 1024, 21),
)


def time_call(fn, x, weight, scale, flush, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        flush.add_(1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(x, weight, [128, 128], scale)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(1)
    device = torch.device("cuda")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    results = []

    # Trigger JIT compilation before allocating the largest shape.
    x0 = torch.randn(1, 128, dtype=torch.bfloat16, device=device)
    w0 = torch.randn(128, 128, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    s0 = torch.ones(1, 1, dtype=torch.float32, device=device)
    cutlass_w8a8_block_fp8_linear_with_fallback(x0, w0, [128, 128], s0)
    torch.cuda.synchronize()

    for n, k, calls in SHAPES:
        x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        scale = torch.ones(n // 128, k // 128, dtype=torch.float32, device=device)

        triton_out = triton_w8a8_block_fp8_linear(x, weight, [128, 128], scale)
        cutlass_out = cutlass_w8a8_block_fp8_linear_with_fallback(
            x, weight, [128, 128], scale
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cutlass_out, triton_out, rtol=0.03, atol=0.15)

        triton_us = time_call(
            triton_w8a8_block_fp8_linear,
            x,
            weight,
            scale,
            flush,
            args.repeats,
        )
        cutlass_us = time_call(
            cutlass_w8a8_block_fp8_linear_with_fallback,
            x,
            weight,
            scale,
            flush,
            args.repeats,
        )
        item = {
            "n": n,
            "k": k,
            "calls_per_token": calls,
            "triton_median_us": statistics.median(triton_us),
            "cutlass_median_us": statistics.median(cutlass_us),
        }
        item["speedup"] = item["triton_median_us"] / item["cutlass_median_us"]
        results.append(item)
        del x, weight, scale, triton_out, cutlass_out
        torch.cuda.empty_cache()

    triton_token_us = sum(x["triton_median_us"] * x["calls_per_token"] for x in results)
    cutlass_token_us = sum(
        x["cutlass_median_us"] * x["calls_per_token"] for x in results
    )
    payload = {
        "device": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "results": results,
        "weighted_triton_ms_per_token": triton_token_us / 1000.0,
        "weighted_cutlass_ms_per_token": cutlass_token_us / 1000.0,
        "weighted_speedup": triton_token_us / cutlass_token_us,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
