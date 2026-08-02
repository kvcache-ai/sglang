#!/usr/bin/env python3
"""Deterministic streaming E2E probe for DSV4 MXFP4 layerwise prefill."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp


def make_input_ids(length: int, salt: int) -> list[int]:
    return [
        1_000 + ((index * 7_919 + salt * 104_729) % 30_000) for index in range(length)
    ]


async def run_request(
    session: aiohttp.ClientSession,
    url: str,
    input_ids: list[int],
    name: str,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at = None
    finished_at = None
    output_ids: list[int] = []
    completion_tokens = 0
    prompt_tokens = 0
    finish_reason = None

    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for raw_line in response.content:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            message = json.loads(data)
            meta = message.get("meta_info") or {}
            prompt_tokens = int(meta.get("prompt_tokens") or prompt_tokens)
            current_completion = int(meta.get("completion_tokens") or 0)
            if current_completion > completion_tokens:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunk_ids = [int(token) for token in message.get("output_ids", [])]
                # Native /generate has shipped both cumulative and incremental
                # output_ids stream shapes.  Normalize either form here.
                if len(chunk_ids) == current_completion:
                    output_ids = chunk_ids
                else:
                    delta = current_completion - completion_tokens
                    output_ids.extend(chunk_ids[-delta:])
                completion_tokens = current_completion
            if meta.get("finish_reason") is not None:
                finish_reason = meta["finish_reason"]
                finished_at = time.perf_counter()

    finished_at = finished_at or time.perf_counter()
    if first_token_at is None:
        raise RuntimeError(f"{name}: response ended without an output token")
    if prompt_tokens != len(input_ids):
        raise RuntimeError(
            f"{name}: server reported {prompt_tokens} prompt tokens, expected {len(input_ids)}"
        )
    if completion_tokens != 8 or len(output_ids) != 8:
        raise RuntimeError(
            f"{name}: expected 8 output tokens, got completion={completion_tokens}, "
            f"ids={output_ids}"
        )
    ttft = first_token_at - started
    e2e = finished_at - started
    decode_time = e2e - ttft
    return {
        "name": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_ids": output_ids,
        "ttft_s": ttft,
        "e2e_s": e2e,
        "decode_tps": 7 / decode_time if decode_time > 0 else 0.0,
        "finish_reason": finish_reason,
    }


async def sample_gpus(stop: asyncio.Event, records: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        record: dict[str, Any] = {"unix_time": time.time()}
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            record["gpus"] = [
                {
                    "index": int(fields[0]),
                    "total_mib": float(fields[1]),
                    "used_mib": float(fields[2]),
                    "free_mib": float(fields[3]),
                    "utilization_pct": float(fields[4]),
                    "power_w": float(fields[5]),
                }
                for fields in (
                    [value.strip() for value in line.split(",")]
                    for line in output.splitlines()
                    if line.strip()
                )
                if int(fields[0]) in (0, 1)
            ]
        except Exception as exc:
            record["error"] = repr(exc)
        records.append(record)
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass


async def main_async(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    resources: list[dict[str, Any]] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(sample_gpus(stop, resources))
    prompt_2048 = make_input_ids(2048, 23)
    prompt_4096 = make_input_ids(4096, 41)
    results = []
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            results.append(
                await run_request(session, args.url, prompt_2048, "hybrid_2048")
            )
            results.append(
                await run_request(session, args.url, prompt_4096, "warmup_4096")
            )
            for repeat in range(3):
                results.append(
                    await run_request(
                        session,
                        args.url,
                        prompt_4096,
                        f"steady_4096_{repeat + 1}",
                    )
                )
    finally:
        stop.set()
        await sampler

    steady = [item for item in results if item["name"].startswith("steady_")]
    output = {
        "label": args.label,
        "url": args.url,
        "results": results,
        "steady_summary": {
            "median_ttft_s": statistics.median(item["ttft_s"] for item in steady),
            "median_decode_tps": statistics.median(
                item["decode_tps"] for item in steady
            ),
            "deterministic": len({tuple(item["output_ids"]) for item in steady}) == 1,
        },
        "resources": resources,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["steady_summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:40021/generate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
