#!/usr/bin/env python3
"""Tune DSV4 decode-sized Triton W8A8 GEMMs on an RTX 4090.

The benchmark evicts L2 before every timed launch, takes the median within
three rounds, then selects from candidates within one percent of the fastest
result by preferring fewer warps and fewer pipeline stages.  The emitted JSON
uses SGLang's existing W8A8 configuration format.  M=9 deliberately restores
the pre-existing default config so decode tuning is not selected for prefill.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: E402
    _w8a8_block_fp8_matmul,
)

DEFAULT_SHAPES = (
    (512, 4096),
    (1024, 4096),
    (32768, 1024),
    (4096, 8192),
    (4096, 4096),
    (4096, 2048),
    (8192, 1024),
)
M_VALUES = (1, 2, 4, 8)
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}


def candidate_configs():
    for block_m, block_n, num_warps, num_stages in itertools.product(
        (16, 32, 64), (32, 64, 128), (4, 8), (2, 3, 4)
    ):
        yield {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }


def launch(a, b, out, a_scale, b_scale, config):
    m, k = a.shape
    n = b.shape[0]
    grid = (
        triton_cdiv(m, config["BLOCK_SIZE_M"]) * triton_cdiv(n, config["BLOCK_SIZE_N"]),
    )
    _w8a8_block_fp8_matmul[grid](
        a,
        b,
        out,
        a_scale,
        b_scale,
        m,
        n,
        k,
        128,
        128,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        out.stride(0),
        out.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(1),
        b_scale.stride(0),
        **config,
        needs_masking=False,
    )


def triton_cdiv(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def cold_l2_time_us(
    launch_fn, flush: torch.Tensor, rounds: int, samples: int, warmup: int
) -> float:
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()

    round_medians = []
    for round_index in range(rounds):
        timings = []
        for sample_index in range(samples):
            flush.fill_((round_index * samples + sample_index) & 0xFF)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch_fn()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000.0)
        round_medians.append(statistics.median(timings))
    return statistics.median(round_medians)


def choose_stable_fastest(results):
    fastest = min(time_us for time_us, _ in results)
    near_fastest = [item for item in results if item[0] <= fastest * 1.01]
    return min(
        near_fastest,
        key=lambda item: (
            item[1]["num_warps"],
            item[1]["num_stages"],
            item[0],
            item[1]["BLOCK_SIZE_M"],
            item[1]["BLOCK_SIZE_N"],
        ),
    )


def tune_shape(n, k, args, flush):
    tuned = {}
    measurements = {}
    for m in M_VALUES:
        torch.manual_seed(args.seed + m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        a_scale = torch.rand(m, k // 128, device="cuda", dtype=torch.float32) + 0.5
        b_scale = (
            torch.rand(n // 128, k // 128, device="cuda", dtype=torch.float32) + 0.5
        )
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        launch(a, b, out, a_scale, b_scale, DEFAULT_CONFIG)
        torch.cuda.synchronize()
        reference = out.clone()
        default_time_us = cold_l2_time_us(
            lambda: launch(a, b, out, a_scale, b_scale, DEFAULT_CONFIG),
            flush,
            args.rounds,
            args.samples,
            args.warmup,
        )
        results = []
        rejected = []
        for config in candidate_configs():
            try:
                launch(a, b, out, a_scale, b_scale, config)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, reference, rtol=args.rtol, atol=args.atol
                )
                time_us = cold_l2_time_us(
                    lambda: launch(a, b, out, a_scale, b_scale, config),
                    flush,
                    args.rounds,
                    args.samples,
                    args.warmup,
                )
                results.append((time_us, config))
            except (AssertionError, RuntimeError) as error:
                rejected.append(
                    {"config": config, "error": f"{type(error).__name__}: {error}"}
                )

        if not results:
            raise RuntimeError(f"all configurations failed for M={m}, N={n}, K={k}")
        time_us, config = choose_stable_fastest(results)
        tuned[str(m)] = config
        measurements[str(m)] = {
            "default_us": default_time_us,
            "selected_us": time_us,
            "speedup": default_time_us / time_us,
            "fastest_us": min(item[0] for item in results),
            "selected_config": config,
            "top5": [
                {"time_us": value, "config": candidate}
                for value, candidate in sorted(results, key=lambda item: item[0])[:5]
            ],
            "rejected": rejected,
        }
        print(
            json.dumps(
                {
                    "M": m,
                    "N": n,
                    "K": k,
                    "time_us": time_us,
                    "default_us": default_time_us,
                    "speedup": default_time_us / time_us,
                    "config": config,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # The loader chooses the closest M key.  This key restores the old default
    # for every M >= 9, including all ordinary prefill batches.
    tuned["9"] = DEFAULT_CONFIG
    return tuned, measurements


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, action="append")
    parser.add_argument("--k", type=int, action="append")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "quantization"
        / "configs",
    )
    parser.add_argument("--measurement-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--flush-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.15)
    args = parser.parse_args()
    if (args.n is None) != (args.k is None):
        parser.error("--n and --k must be supplied together")
    if args.n is not None and len(args.n) != len(args.k):
        parser.error("the number of --n and --k arguments must match")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError(
            f"this tuner targets SM89, got {torch.cuda.get_device_capability()}"
        )
    shapes = tuple(zip(args.n, args.k)) if args.n is not None else DEFAULT_SHAPES
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.measurement_dir is not None:
        args.measurement_dir.mkdir(parents=True, exist_ok=True)
    flush = torch.empty(args.flush_mib * 1024 * 1024, dtype=torch.uint8, device="cuda")

    for n, k in shapes:
        if n % 128 or k % 128:
            raise ValueError(f"N and K must be divisible by 128, got N={n}, K={k}")
        tuned, measurements = tune_shape(n, k, args, flush)
        filename = (
            f"N={n},K={k},device_name={device_name},dtype=fp8_w8a8,"
            "block_shape=[128, 128].json"
        )
        output_path = args.output_dir / filename
        output_path.write_text(json.dumps(tuned, indent=4) + "\n")
        if args.measurement_dir is not None:
            measurement_path = args.measurement_dir / filename
            measurement_path.write_text(
                json.dumps(
                    {
                        "device": device_name,
                        "rounds": args.rounds,
                        "samples": args.samples,
                        "flush_mib": args.flush_mib,
                        "N": n,
                        "K": k,
                        "measurements": measurements,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps({"wrote": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
