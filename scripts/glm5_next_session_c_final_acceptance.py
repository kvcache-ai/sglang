#!/usr/bin/env python3
"""Fail-closed Session-C final acceptance driver.

This is an HTTP/log artifact harness, not serving code.  It covers the pieces
which the ordinary correctness oracle and performance benchmark deliberately
do not prove:

* exact layerwise-prefill chunk boundaries (128/4096/4097/8193);
* threshold-0 ordinary prefill versus threshold-1 layerwise parity;
* the final 500000-input + 1024-output integrated request;
* real decode CUDA Graph counter movement; and
* exact batch-size 1/2/4 graph replay for the Qwen regression.

The layerwise and long modes require access to the server log.  They snapshot
the log immediately before each request and accept only complete 42-layer
rounds with the expected cumulative counters and ``fallback_count=0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm5_next_session_c_benchmark import (  # noqa: E402
    FIXTURE_SEED,
    build_fixture,
    flush_cache,
    run_streaming_request,
)


SCHEMA_VERSION = 1
DEFAULT_VOCAB_SIZE = 154880
LAYERWISE_LENGTHS = (128, 4096, 4097, 8193)
LAYERWISE_CHUNK_SIZE = 4096
LAYERWISE_MOE_LAYERS = 42
LAYERWISE_PREFETCH_HITS_PER_ROUND = LAYERWISE_MOE_LAYERS - 1
LAYERWISE_FINAL_SLOT = (LAYERWISE_MOE_LAYERS - 1) % 2
DEFAULT_BOUNDARY_OUTPUT = 16
LAYERWISE_FIXTURE_SEED = 20260811
REQUIRED_TP_SIZE = 8
PREFILL_PARITY_COLLECTION_TOP_K = 256
PREFILL_PARITY_COMPARISON_TOP_K = 64
PREFILL_PARITY_LOGPROB_ATOL = 5e-2
PREFILL_PARITY_LOGPROB_RTOL = 5e-2
MIN_LONG_INPUT = 500_000
MIN_LONG_OUTPUT = 1024
BUCKET_INPUT_TOKENS = 128
BUCKET_OUTPUT_TOKENS = 8
BUCKET_TOP_K = 64

LAYERWISE_PATTERN = re.compile(
    r"KT GLM-5-Next FP8 layerwise prefill: "
    r"layer=(?P<layer>\d+) epoch=(?P<epoch>\d+) "
    r"slot=(?P<slot>\d+) (?P<load_kind>prefetch-hit|prime) "
    r"apply_count=(?P<apply_count>\d+) "
    r"prime_count=(?P<prime_count>\d+) "
    r"prefetch_hit_count=(?P<prefetch_hit_count>\d+) "
    r"completed_rounds=(?P<completed_rounds>\d+) "
    r"fallback_count=(?P<fallback_count>\d+)"
)

LAYERWISE_STARTUP_PATTERN = re.compile(
    r"KT GLM-5-Next FP8 layerwise prefill eagerly allocated two raw "
    r"E4M3\+FP32 full-layer slots on (?P<device>\S+) before KV profiling "
    r"\(total=(?P<total_gib>[0-9]+(?:\.[0-9]+)?) GiB, "
    r"contract=(?P<contract>[0-9a-f]{64}), "
    r"fallback_count=(?P<fallback_count>\d+)\)"
)

SERVER_ARGS_PATTERN = re.compile(r"server_args=ServerArgs\((?P<body>[^\n]+)\)")

PROMETHEUS_VALUE_PATTERN = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)

FATAL_LOG_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"CUDA error", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
    re.compile(r"device-side assert", re.IGNORECASE),
    re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"deadlock", re.IGNORECASE),
    re.compile(r"manager is poisoned", re.IGNORECASE),
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_token_rows(rows: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(8, "little", signed=False))
        for token_id in row:
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _get_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:1024]
        raise RuntimeError(
            f"GET {url} returned HTTP {error.code}: {detail!r}"
        ) from error
    if status < 200 or status >= 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}: {body[:1024]!r}")
    return body.decode("utf-8", errors="replace")


def fetch_metrics(base_url: str, *, timeout: float) -> str:
    return _get_text(f"{base_url.rstrip('/')}/metrics", timeout=timeout)


def prometheus_counter_value(
    text: str,
    metric_name: str,
    *,
    required_labels: dict[str, str] | None = None,
) -> float | None:
    """Sum all matching Prometheus samples, or return None if none exist."""
    required_labels = required_labels or {}
    total = 0.0
    matches = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_VALUE_PATTERN.match(line)
        if match is None or match.group("name") != metric_name:
            continue
        labels = dict(
            re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match.group("labels") or "")
        )
        if any(labels.get(key) != value for key, value in required_labels.items()):
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        total += value
        matches += 1
    return total if matches else None


def decode_graph_counter(text: str) -> float | None:
    return prometheus_counter_value(
        text,
        "sglang:cuda_graph_passes_total",
        required_labels={"mode": "decode_cuda_graph"},
    )


def cuda_graph_counters_by_rank(text: str) -> dict[str, dict[str, float]]:
    counters = {"decode_cuda_graph": {}, "decode_none": {}}
    for raw_line in text.splitlines():
        match = PROMETHEUS_VALUE_PATTERN.match(raw_line.strip())
        if match is None or match.group("name") != "sglang:cuda_graph_passes_total":
            continue
        labels = dict(
            re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', match.group("labels") or "")
        )
        mode = labels.get("mode")
        # ``tp_rank`` is the production metric contract.  Accepting the older,
        # ambiguous ``rank`` label can accidentally aggregate unrelated ranks
        # and must fail closed in final acceptance.
        rank = labels.get("tp_rank")
        if mode not in counters or rank is None:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        if rank in counters[mode]:
            raise ValueError(f"duplicate {mode} counter for TP rank {rank}")
        counters[mode][rank] = value
    return counters


def validate_graph_counter_progress(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    *,
    tp_size: int,
    minimum_graph_delta_per_rank: int,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_ranks = {str(rank) for rank in range(tp_size)}
    per_rank: dict[str, Any] = {}
    for mode in ("decode_cuda_graph", "decode_none"):
        if (
            set(before.get(mode, {})) != expected_ranks
            or set(after.get(mode, {})) != expected_ranks
        ):
            failures.append(
                f"{mode} counters do not cover exact TP ranks 0..{tp_size - 1}"
            )
    for rank in sorted(expected_ranks, key=int):
        graph_before = before.get("decode_cuda_graph", {}).get(rank)
        graph_after = after.get("decode_cuda_graph", {}).get(rank)
        none_before = before.get("decode_none", {}).get(rank)
        none_after = after.get("decode_none", {}).get(rank)
        graph_delta = (
            graph_after - graph_before
            if graph_before is not None and graph_after is not None
            else None
        )
        none_delta = (
            none_after - none_before
            if none_before is not None and none_after is not None
            else None
        )
        rank_failures: list[str] = []
        if graph_delta is not None and graph_delta < minimum_graph_delta_per_rank:
            rank_failures.append(
                f"graph delta {graph_delta} is below {minimum_graph_delta_per_rank}"
            )
        if none_delta is not None and none_delta != 0:
            rank_failures.append(f"decode_none delta is {none_delta}, expected 0")
        failures.extend(f"TP rank {rank}: {item}" for item in rank_failures)
        per_rank[rank] = {
            "graph_before": graph_before,
            "graph_after": graph_after,
            "graph_delta": graph_delta,
            "decode_none_before": none_before,
            "decode_none_after": none_after,
            "decode_none_delta": none_delta,
            "failures": rank_failures,
            "pass": not rank_failures
            and graph_delta is not None
            and none_delta is not None,
        }
    return {
        "expected_tp_ranks": sorted(expected_ranks, key=int),
        "minimum_graph_delta_per_rank": minimum_graph_delta_per_rank,
        "expected_decode_none_delta_per_rank": 0,
        "per_rank": per_rank,
        "failures": failures,
        "pass": not failures,
    }


def parse_layerwise_summaries(text: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for match in LAYERWISE_PATTERN.finditer(text):
        item: dict[str, Any] = {
            key: int(value) if key != "load_kind" else value
            for key, value in match.groupdict().items()
        }
        summaries.append(item)
    return summaries


def _zero_layerwise_summary() -> dict[str, int]:
    return {
        "layer": 44,
        "epoch": -1,
        "slot": -1,
        "apply_count": 0,
        "prime_count": 0,
        "prefetch_hit_count": 0,
        "completed_rounds": 0,
        "fallback_count": 0,
    }


def validate_layerwise_progress(
    previous: dict[str, Any] | None,
    current: list[dict[str, Any]],
    *,
    expected_rounds: int,
) -> list[str]:
    failures: list[str] = []
    prior = dict(previous or _zero_layerwise_summary())
    if len(current) != expected_rounds:
        failures.append(
            f"expected {expected_rounds} completed layerwise rounds, "
            f"observed {len(current)}"
        )
    for index, summary in enumerate(current):
        expected = {
            "layer": 44,
            "epoch": int(prior["epoch"]) + 1,
            "slot": LAYERWISE_FINAL_SLOT,
            "load_kind": "prefetch-hit",
            "apply_count": int(prior["apply_count"]) + LAYERWISE_MOE_LAYERS,
            "prime_count": int(prior["prime_count"]) + 1,
            "prefetch_hit_count": int(prior["prefetch_hit_count"])
            + LAYERWISE_PREFETCH_HITS_PER_ROUND,
            "completed_rounds": int(prior["completed_rounds"]) + 1,
            "fallback_count": 0,
        }
        for key, expected_value in expected.items():
            if summary.get(key) != expected_value:
                failures.append(
                    f"round {index} {key}: expected {expected_value}, "
                    f"observed {summary.get(key)}"
                )
        if summary.get("apply_count") != summary.get("prime_count", 0) + summary.get(
            "prefetch_hit_count", 0
        ):
            failures.append(
                f"round {index}: apply_count does not equal prime+prefetch_hit"
            )
        prior = summary
    return failures


def validate_layerwise_startup(text: str) -> dict[str, Any]:
    """Prove that the required two GPU slots were allocated before KV pools."""
    failures: list[str] = []
    matches = list(LAYERWISE_STARTUP_PATTERN.finditer(text))
    if len(matches) != 1:
        failures.append(
            "expected exactly one GLM-5-Next FP8 two-slot allocation marker, "
            f"observed {len(matches)}"
        )
    allocation: dict[str, Any] | None = None
    if matches:
        match = matches[0]
        total_gib = float(match.group("total_gib"))
        fallback_count = int(match.group("fallback_count"))
        allocation = {
            "offset": match.start(),
            "device": match.group("device"),
            "total_gib": total_gib,
            "contract_sha256": match.group("contract"),
            "fallback_count": fallback_count,
        }
        if total_gib <= 0:
            failures.append("layerwise two-slot allocation size is not positive")
        if fallback_count != 0:
            failures.append(
                f"startup layerwise fallback_count is {fallback_count}, expected 0"
            )

        memory_pool_offsets = [
            match.start() for match in re.finditer(r"Memory pool end\.", text)
        ]
        if not memory_pool_offsets:
            failures.append("server log has no completed KV memory-pool marker")
        elif match.start() >= min(memory_pool_offsets):
            failures.append(
                "layerwise two-slot allocation marker does not precede KV memory pool"
            )

        ready_offset = text.find("The server is fired up and ready to roll!")
        if ready_offset < 0:
            failures.append("server-ready marker is missing")
        elif match.start() >= ready_offset:
            failures.append(
                "layerwise two-slot allocation marker does not precede server ready"
            )
        first_round = LAYERWISE_PATTERN.search(text)
        if first_round is not None and match.start() >= first_round.start():
            failures.append("layerwise execution appears before startup allocation")

    return {
        "allocation": allocation,
        "allocation_marker_count": len(matches),
        "failures": failures,
        "pass": not failures,
    }


def _server_arg_value(body: str, name: str) -> str | None:
    match = re.search(rf"(?:^|, ){re.escape(name)}=(?P<value>.*?)(?=, \w+=|$)", body)
    return match.group("value") if match is not None else None


def validate_prefill_parity_server_log(text: str, *, threshold: int) -> dict[str, Any]:
    """Bind a parity collection to the exact TP8 threshold-0/1 server."""
    failures: list[str] = []
    server_args_matches = list(SERVER_ARGS_PATTERN.finditer(text))
    fields: dict[str, str | None] = {}
    if len(server_args_matches) != 1:
        failures.append(
            f"expected exactly one server_args record, observed {len(server_args_matches)}"
        )
    if server_args_matches:
        body = server_args_matches[0].group("body")
        fields = {
            name: _server_arg_value(body, name)
            for name in (
                "tp_size",
                "chunked_prefill_size",
                "kt_method",
                "kt_gpu_prefill_token_threshold",
            )
        }
        expected = {
            "tp_size": str(REQUIRED_TP_SIZE),
            "chunked_prefill_size": str(LAYERWISE_CHUNK_SIZE),
            "kt_method": "'FP8'",
            "kt_gpu_prefill_token_threshold": str(threshold),
        }
        for name, expected_value in expected.items():
            if fields.get(name) != expected_value:
                failures.append(
                    f"server_args {name}: expected {expected_value}, "
                    f"observed {fields.get(name)}"
                )

    if text.count("The server is fired up and ready to roll!") != 1:
        failures.append("server log must contain exactly one server-ready marker")

    startup = validate_layerwise_startup(text) if threshold == 1 else None
    if startup is not None:
        failures.extend(startup["failures"])
    else:
        if LAYERWISE_STARTUP_PATTERN.search(text) is not None:
            failures.append("threshold-0 server allocated the layerwise manager")
        if parse_layerwise_summaries(text):
            failures.append("threshold-0 server executed the layerwise manager")

    return {
        "threshold": threshold,
        "server_args": fields,
        "layerwise_startup": startup,
        "failures": failures,
        "pass": not failures,
    }


def fatal_log_lines(text: str) -> list[str]:
    failures: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in FATAL_LOG_PATTERNS):
            failures.append(line[:2048])
    return failures


def _read_tail(path: Path, *, maximum_bytes: int = 4 * 1024 * 1024) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _log_before(path: Path) -> tuple[int, dict[str, Any] | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    summaries = parse_layerwise_summaries(_read_tail(path))
    return size, summaries[-1] if summaries else None


def _log_after(path: Path, offset: int) -> tuple[int, str]:
    size = path.stat().st_size
    if size < offset:
        raise RuntimeError(
            f"server log was truncated during request: before={offset}, after={size}"
        )
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    return size, data.decode("utf-8", errors="replace")


def validate_completed_request(
    record: dict[str, Any], *, output_tokens: int, vocab_size: int
) -> list[str]:
    failures: list[str] = []
    if not record.get("success"):
        return [str(record.get("error", "request failed"))]
    if not _finite_tree(record):
        failures.append("request record contains NaN or Inf")
    finish_reason = record.get("finish_reason")
    finish_type = (
        finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
    )
    if finish_type != "length":
        failures.append(f"request finish_reason is {finish_reason!r}, expected length")
    output_ids = record.get("output_ids")
    if not isinstance(output_ids, list) or len(output_ids) != output_tokens:
        failures.append("request output token shape is invalid")
    elif any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or not 0 <= token_id < vocab_size
        for token_id in output_ids
    ):
        failures.append("request contains an out-of-range output token ID")
    return failures


def _capture_request_evidence(
    *,
    base_url: str,
    server_log: Path,
    input_ids: list[int],
    output_tokens: int,
    chunk_size: int,
    timeout: float,
    should_flush_cache: bool,
    tp_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    failures: list[str] = []
    before_offset, previous_summary = _log_before(server_log)
    metrics_before_text = fetch_metrics(base_url, timeout=timeout)
    graph_before = decode_graph_counter(metrics_before_text)
    counters_before = cuda_graph_counters_by_rank(metrics_before_text)
    record: dict[str, Any]
    try:
        if should_flush_cache:
            flush_cache(base_url, timeout=timeout)
        record = run_streaming_request(
            base_url,
            input_ids,
            output_tokens,
            timeout=timeout,
        )
    except Exception as error:
        record = {"success": False, "error": f"{type(error).__name__}: {error}"}
        failures.append(str(record["error"]))
    after_offset, log_segment = _log_after(server_log, before_offset)
    metrics_after_text = fetch_metrics(base_url, timeout=timeout)
    graph_after = decode_graph_counter(metrics_after_text)
    counters_after = cuda_graph_counters_by_rank(metrics_after_text)

    summaries = parse_layerwise_summaries(log_segment)
    expected_rounds = math.ceil(len(input_ids) / chunk_size)
    failures.extend(
        validate_layerwise_progress(
            previous_summary,
            summaries,
            expected_rounds=expected_rounds,
        )
    )
    fatal_lines = fatal_log_lines(log_segment)
    if fatal_lines:
        failures.append(f"server log contains {len(fatal_lines)} fatal line(s)")
    graph_delta = (
        graph_after - graph_before
        if graph_before is not None and graph_after is not None
        else None
    )
    graph_gate = validate_graph_counter_progress(
        counters_before,
        counters_after,
        tp_size=tp_size,
        minimum_graph_delta_per_rank=max(output_tokens - 1, 1),
    )
    failures.extend(graph_gate["failures"])

    request_failures = validate_completed_request(
        record, output_tokens=output_tokens, vocab_size=vocab_size
    )
    for failure in request_failures:
        if failure not in failures:
            failures.append(failure)
    return {
        "input_tokens": len(input_ids),
        "output_tokens": output_tokens,
        "expected_layerwise_rounds": expected_rounds,
        "previous_layerwise_summary": previous_summary,
        "layerwise_summaries": summaries,
        "decode_cuda_graph_counter": {
            "before": graph_before,
            "after": graph_after,
            "delta": graph_delta,
            "minimum_delta": max(output_tokens - 1, 1),
            "per_rank_gate": graph_gate,
        },
        "server_log_window": {
            "path": str(server_log.resolve()),
            "start_offset": before_offset,
            "end_offset": after_offset,
            "sha256": _sha256_bytes(log_segment.encode("utf-8")),
            "fatal_lines": fatal_lines,
        },
        "metrics_evidence": {
            "before_sha256": _sha256_bytes(metrics_before_text.encode("utf-8")),
            "after_sha256": _sha256_bytes(metrics_after_text.encode("utf-8")),
        },
        "record": record,
        "failures": failures,
        "pass": not failures,
    }


def run_layerwise(args: argparse.Namespace) -> int:
    lengths = tuple(args.lengths)
    failures: list[str] = []
    contract_compliant = (
        lengths == LAYERWISE_LENGTHS
        and args.chunk_size == LAYERWISE_CHUNK_SIZE
        and args.output_length == DEFAULT_BOUNDARY_OUTPUT
        and args.fixture_seed == LAYERWISE_FIXTURE_SEED
        and args.vocab_size == DEFAULT_VOCAB_SIZE
        and args.tp_size == REQUIRED_TP_SIZE
    )
    if not contract_compliant:
        failures.append(
            "layerwise matrix must use lengths 128/4096/4097/8193, "
            "chunk size 4096, exactly 16 output tokens, seed 20260811, "
            "TP=8, and vocab size 154880"
        )

    server_log = Path(args.server_log)
    try:
        startup_text = server_log.read_text(encoding="utf-8", errors="replace")
        startup_evidence = validate_layerwise_startup(startup_text)
    except Exception as error:
        startup_evidence = {
            "failures": [f"{type(error).__name__}: {error}"],
            "pass": False,
        }
    failures.extend(f"startup: {failure}" for failure in startup_evidence["failures"])

    cases: list[dict[str, Any]] = []
    for length in lengths if contract_compliant and startup_evidence["pass"] else ():
        fixture = build_fixture(
            length,
            args.vocab_size,
            salt=89,
            seed=args.fixture_seed,
        )
        input_ids = fixture.pop("input_ids")
        try:
            evidence = _capture_request_evidence(
                base_url=args.base_url,
                server_log=Path(args.server_log),
                input_ids=input_ids,
                output_tokens=args.output_length,
                chunk_size=args.chunk_size,
                timeout=args.request_timeout,
                should_flush_cache=not args.no_flush_cache,
                tp_size=args.tp_size,
                vocab_size=args.vocab_size,
            )
        except Exception as error:
            evidence = {
                "input_tokens": length,
                "output_tokens": args.output_length,
                "failures": [f"{type(error).__name__}: {error}"],
                "pass": False,
            }
        evidence["fixture"] = fixture
        cases.append(evidence)
        failures.extend(
            f"length {length}: {failure}" for failure in evidence["failures"]
        )
        if not evidence["pass"] and args.stop_on_failure:
            break

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_layerwise_boundaries",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "contract": {
            "lengths": list(lengths),
            "chunk_size": args.chunk_size,
            "output_tokens": args.output_length,
            "expected_rounds": {
                str(length): math.ceil(length / args.chunk_size) for length in lengths
            },
            "require_decode_cuda_graph": True,
            "require_layerwise_fallback_zero": True,
            "require_startup_two_slot_allocation_before_kv_pool": True,
            "fixture_seed": args.fixture_seed,
            "vocab_size": args.vocab_size,
            "tp_size": args.tp_size,
            "compliant": contract_compliant,
        },
        "startup_evidence": startup_evidence,
        "cases": cases,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def run_long(args: argparse.Namespace) -> int:
    fixture = build_fixture(
        args.input_length,
        args.vocab_size,
        salt=73,
        seed=args.fixture_seed,
    )
    input_ids = fixture.pop("input_ids")
    failures: list[str] = []
    contract_compliant = (
        args.input_length == MIN_LONG_INPUT
        and args.output_length == MIN_LONG_OUTPUT
        and args.chunk_size == LAYERWISE_CHUNK_SIZE
        and args.fixture_seed == FIXTURE_SEED
        and args.vocab_size == DEFAULT_VOCAB_SIZE
        and args.tp_size == REQUIRED_TP_SIZE
    )
    if not contract_compliant:
        failures.append(
            "final long request requires exactly 500000 input, 1024 output, "
            "layerwise chunk size 4096, seed 20260812, TP=8, and vocab 154880"
        )
    try:
        if not contract_compliant:
            raise ValueError("refusing to issue a non-contract final long request")
        evidence = _capture_request_evidence(
            base_url=args.base_url,
            server_log=Path(args.server_log),
            input_ids=input_ids,
            output_tokens=args.output_length,
            chunk_size=args.chunk_size,
            timeout=args.request_timeout,
            should_flush_cache=not args.no_flush_cache,
            tp_size=args.tp_size,
            vocab_size=args.vocab_size,
        )
    except Exception as error:
        evidence = {
            "input_tokens": args.input_length,
            "output_tokens": args.output_length,
            "failures": [f"{type(error).__name__}: {error}"],
            "pass": False,
        }
    failures.extend(evidence["failures"])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_final_long_context",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "contract": {
            "request_concurrency": 1,
            "input_tokens": MIN_LONG_INPUT,
            "output_tokens": MIN_LONG_OUTPUT,
            "chunk_size": args.chunk_size,
            "expected_layerwise_rounds": math.ceil(args.input_length / args.chunk_size),
            "require_decode_cuda_graph": True,
            "require_layerwise_fallback_zero": True,
            "throughput_gate": None,
            "fixture_seed": args.fixture_seed,
            "vocab_size": args.vocab_size,
            "tp_size": args.tp_size,
            "compliant": contract_compliant,
        },
        "fixture": fixture,
        "evidence": evidence,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def _post_batch(
    base_url: str,
    rows: list[list[int]],
    *,
    output_tokens: int,
    top_k: int,
    timeout: float,
    vocab_size: int,
) -> list[dict[str, Any]]:
    body = {
        "input_ids": rows,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "top_logprobs_num": top_k,
        "return_text_in_logprobs": False,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:1024]
        raise RuntimeError(
            f"generate returned HTTP {error.code}: {detail!r}"
        ) from error
    if status < 200 or status >= 300:
        raise RuntimeError(f"generate returned HTTP {status}: {raw[:1024]!r}")
    payload = json.loads(raw)
    rows_out = payload if isinstance(payload, list) else [payload]
    if len(rows_out) != len(rows):
        raise RuntimeError(
            f"batch response has {len(rows_out)} rows, expected {len(rows)}"
        )
    if not _finite_tree(rows_out):
        raise RuntimeError("batch response contains NaN or Inf")
    for index, row in enumerate(rows_out):
        if not isinstance(row, dict):
            raise RuntimeError(f"row {index} is not an object")
        output_ids = row.get("output_ids") or []
        if len(output_ids) != output_tokens:
            raise RuntimeError(
                f"row {index} has {len(output_ids)} output ids, "
                f"expected {output_tokens}"
            )
        for token_id in output_ids:
            if (
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < vocab_size
            ):
                raise RuntimeError(
                    f"row {index} has invalid output token id {token_id!r}"
                )
        meta = row.get("meta_info") or {}
        if meta.get("prompt_tokens") != len(rows[index]):
            raise RuntimeError(f"row {index} has invalid prompt_tokens")
        if meta.get("completion_tokens") != output_tokens:
            raise RuntimeError(f"row {index} has invalid completion_tokens")
        finish_reason = meta.get("finish_reason")
        finish_type = (
            finish_reason.get("type")
            if isinstance(finish_reason, dict)
            else finish_reason
        )
        if finish_type != "length":
            raise RuntimeError(
                f"row {index} has invalid finish_reason {finish_reason!r}"
            )
        raw_steps = meta.get("output_top_logprobs") or []
        if len(raw_steps) != output_tokens:
            raise RuntimeError(
                f"row {index} has {len(raw_steps)} top-logprob steps, "
                f"expected {output_tokens}"
            )
        for step_index, raw_top in enumerate(raw_steps):
            if len(raw_top or []) < top_k:
                raise RuntimeError(
                    f"row {index} step {step_index} has {len(raw_top or [])} "
                    f"top logprobs, expected at least {top_k}"
                )
            seen: set[int] = set()
            for entry in raw_top:
                if isinstance(entry, dict):
                    token_id, logprob = entry.get("token_id"), entry.get("logprob")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    logprob, token_id = entry[0], entry[1]
                else:
                    raise RuntimeError(
                        f"row {index} step {step_index} has invalid top logprob"
                    )
                if (
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or not 0 <= token_id < vocab_size
                    or token_id in seen
                ):
                    raise RuntimeError(
                        f"row {index} step {step_index} has invalid top token id {token_id!r}"
                    )
                seen.add(token_id)
                if (
                    isinstance(logprob, bool)
                    or not isinstance(logprob, (int, float))
                    or not math.isfinite(float(logprob))
                ):
                    raise RuntimeError(
                        f"row {index} step {step_index} has invalid logprob {logprob!r}"
                    )
            generated_id = output_ids[step_index]
            generated_entries = []
            for entry in raw_top:
                if isinstance(entry, dict):
                    token_id, logprob = entry.get("token_id"), entry.get("logprob")
                else:
                    logprob, token_id = entry[0], entry[1]
                if token_id == generated_id:
                    generated_entries.append(float(logprob))
            maximum_logprob = max(
                float(entry.get("logprob"))
                if isinstance(entry, dict)
                else float(entry[0])
                for entry in raw_top
            )
            if not generated_entries or generated_entries[0] != maximum_logprob:
                raise RuntimeError(
                    f"row {index} step {step_index} generated token "
                    f"{generated_id} is not the greedy top-logprob token"
                )
    return rows_out


def _normalize_top_logprobs(raw_steps: Any, top_k: int) -> list[list[dict[str, Any]]]:
    normalized: list[list[dict[str, Any]]] = []
    for raw_top in raw_steps or []:
        step: list[dict[str, Any]] = []
        for entry in (raw_top or [])[:top_k]:
            if isinstance(entry, dict):
                token_id = entry.get("token_id")
                logprob = entry.get("logprob")
            else:
                logprob, token_id = entry[0], entry[1]
            step.append({"token_id": int(token_id), "logprob": float(logprob)})
        normalized.append(step)
    return normalized


def _collect_prefill_parity_case(
    args: argparse.Namespace, *, length: int
) -> dict[str, Any]:
    fixture = build_fixture(
        length,
        args.vocab_size,
        salt=89,
        seed=args.fixture_seed,
    )
    input_ids = fixture.pop("input_ids")
    server_log = Path(args.server_log)
    before_offset, previous_summary = _log_before(server_log)
    if not args.no_flush_cache:
        flush_cache(args.base_url, timeout=args.request_timeout)
    response = _post_batch(
        args.base_url,
        [input_ids],
        output_tokens=args.output_length,
        top_k=args.top_k,
        timeout=args.request_timeout,
        vocab_size=args.vocab_size,
    )[0]
    after_offset, log_segment = _log_after(server_log, before_offset)
    summaries = parse_layerwise_summaries(log_segment)
    failures: list[str] = []
    expected_rounds = math.ceil(length / args.chunk_size)
    if args.threshold == 1:
        failures.extend(
            validate_layerwise_progress(
                previous_summary,
                summaries,
                expected_rounds=expected_rounds,
            )
        )
    elif summaries:
        failures.append(
            f"threshold-0 request emitted {len(summaries)} layerwise round(s)"
        )
    fatal_lines = fatal_log_lines(log_segment)
    if fatal_lines:
        failures.append(f"server log contains {len(fatal_lines)} fatal line(s)")

    output_ids = [int(token_id) for token_id in response.get("output_ids", [])]
    meta = response.get("meta_info") or {}
    top_logprobs = _normalize_top_logprobs(meta.get("output_top_logprobs"), args.top_k)
    return {
        "input_tokens": length,
        "fixture": fixture,
        "output_ids": output_ids,
        "output_sha256": _sha256_token_rows([output_ids]),
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "top_logprobs": top_logprobs,
        "expected_layerwise_rounds": expected_rounds if args.threshold == 1 else 0,
        "previous_layerwise_summary": previous_summary,
        "layerwise_summaries": summaries,
        "server_log_window": {
            "path": str(server_log.resolve()),
            "start_offset": before_offset,
            "end_offset": after_offset,
            "sha256": _sha256_bytes(log_segment.encode("utf-8")),
            "fatal_lines": fatal_lines,
        },
        "failures": failures,
        "pass": not failures,
    }


def run_prefill_parity(args: argparse.Namespace) -> int:
    lengths = tuple(args.lengths)
    contract_compliant = (
        args.threshold in (0, 1)
        and lengths == LAYERWISE_LENGTHS
        and args.output_length == DEFAULT_BOUNDARY_OUTPUT
        and args.chunk_size == LAYERWISE_CHUNK_SIZE
        and args.fixture_seed == LAYERWISE_FIXTURE_SEED
        and args.tp_size == REQUIRED_TP_SIZE
        and args.vocab_size == DEFAULT_VOCAB_SIZE
        and args.top_k == PREFILL_PARITY_COLLECTION_TOP_K
    )
    failures: list[str] = []
    if not contract_compliant:
        failures.append(
            "prefill parity requires threshold 0/1, lengths "
            "128/4096/4097/8193, output 16, chunk 4096, seed 20260811, "
            "TP=8, vocab 154880, and collected top-k 256"
        )

    server_log = Path(args.server_log)
    try:
        initial_log = server_log.read_text(encoding="utf-8", errors="replace")
        server_evidence = validate_prefill_parity_server_log(
            initial_log, threshold=args.threshold
        )
        server_evidence.update(
            {
                "path": str(server_log.resolve()),
                "initial_size": len(initial_log.encode("utf-8")),
                "initial_sha256": _sha256_bytes(initial_log.encode("utf-8")),
            }
        )
    except Exception as error:
        server_evidence = {
            "failures": [f"{type(error).__name__}: {error}"],
            "pass": False,
        }
    failures.extend(f"server: {failure}" for failure in server_evidence["failures"])

    cases: list[dict[str, Any]] = []
    for length in lengths if contract_compliant and server_evidence["pass"] else ():
        try:
            case = _collect_prefill_parity_case(args, length=length)
        except Exception as error:
            case = {
                "input_tokens": length,
                "failures": [f"{type(error).__name__}: {error}"],
                "pass": False,
            }
        cases.append(case)
        failures.extend(f"length {length}: {failure}" for failure in case["failures"])
        if not case["pass"] and args.stop_on_failure:
            break

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_prefill_threshold_collection",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "contract": {
            "threshold": args.threshold,
            "lengths": list(lengths),
            "output_tokens": args.output_length,
            "chunk_size": args.chunk_size,
            "fixture_seed": args.fixture_seed,
            "tp_size": args.tp_size,
            "vocab_size": args.vocab_size,
            "collection_top_k": args.top_k,
            "comparison_top_k": PREFILL_PARITY_COMPARISON_TOP_K,
            "temperature": 0,
            "ignore_eos": True,
            "require_finish_reason_length": True,
            "require_exact_greedy_tokens": True,
            "compliant": contract_compliant,
        },
        "server_evidence": server_evidence,
        "cases": cases,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def _validate_prefill_collection(
    payload: dict[str, Any], *, role: str, expected_threshold: int
) -> tuple[list[str], dict[int, dict[str, Any]]]:
    failures: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"{role}: unsupported schema")
    if payload.get("kind") != "glm5_next_session_c_prefill_threshold_collection":
        failures.append(f"{role}: payload kind is invalid")
    if payload.get("pass") is not True:
        failures.append(f"{role}: collection did not pass")
    server_evidence = payload.get("server_evidence")
    if not isinstance(server_evidence, dict) or server_evidence.get("pass") is not True:
        failures.append(f"{role}: server evidence did not pass")
    else:
        if server_evidence.get("threshold") != expected_threshold:
            failures.append(f"{role}: server-evidence threshold is invalid")
        server_args = server_evidence.get("server_args")
        expected_server_args = {
            "tp_size": str(REQUIRED_TP_SIZE),
            "chunked_prefill_size": str(LAYERWISE_CHUNK_SIZE),
            "kt_method": "'FP8'",
            "kt_gpu_prefill_token_threshold": str(expected_threshold),
        }
        if not isinstance(server_args, dict) or any(
            server_args.get(key) != value for key, value in expected_server_args.items()
        ):
            failures.append(f"{role}: server_args evidence is invalid")
        startup = server_evidence.get("layerwise_startup")
        if expected_threshold == 1 and (
            not isinstance(startup, dict) or startup.get("pass") is not True
        ):
            failures.append(f"{role}: layerwise startup evidence is invalid")
        if expected_threshold == 0 and startup is not None:
            failures.append(f"{role}: threshold-0 startup evidence is not null")
        if not isinstance(server_evidence.get("path"), str):
            failures.append(f"{role}: server-log path evidence is missing")
        initial_size = server_evidence.get("initial_size")
        if (
            isinstance(initial_size, bool)
            or not isinstance(initial_size, int)
            or initial_size <= 0
        ):
            failures.append(f"{role}: server-log size evidence is invalid")
        initial_sha256 = server_evidence.get("initial_sha256")
        if (
            not isinstance(initial_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", initial_sha256) is None
        ):
            failures.append(f"{role}: server-log digest evidence is invalid")
    contract = payload.get("contract")
    expected_contract = {
        "threshold": expected_threshold,
        "lengths": list(LAYERWISE_LENGTHS),
        "output_tokens": DEFAULT_BOUNDARY_OUTPUT,
        "chunk_size": LAYERWISE_CHUNK_SIZE,
        "fixture_seed": LAYERWISE_FIXTURE_SEED,
        "tp_size": REQUIRED_TP_SIZE,
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "collection_top_k": PREFILL_PARITY_COLLECTION_TOP_K,
        "comparison_top_k": PREFILL_PARITY_COMPARISON_TOP_K,
        "temperature": 0,
        "ignore_eos": True,
        "require_finish_reason_length": True,
        "require_exact_greedy_tokens": True,
        "compliant": True,
    }
    if not isinstance(contract, dict):
        failures.append(f"{role}: contract is missing")
    else:
        if set(contract) != set(expected_contract):
            failures.append(f"{role}: contract fields are not exact")
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                failures.append(
                    f"{role}: contract {key} is {contract.get(key)!r}, "
                    f"expected {expected!r}"
                )

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        failures.append(f"{role}: cases are missing")
        return failures, {}
    cases: dict[int, dict[str, Any]] = {}
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            failures.append(f"{role}: case {index} is not an object")
            continue
        length = case.get("input_tokens")
        if isinstance(length, bool) or not isinstance(length, int):
            failures.append(f"{role}: case {index} has invalid input length")
            continue
        if length in cases:
            failures.append(f"{role}: duplicate length {length}")
            continue
        cases[length] = case
    if set(cases) != set(LAYERWISE_LENGTHS):
        failures.append(f"{role}: case set is not exact 128/4096/4097/8193")

    for length, case in cases.items():
        if case.get("pass") is not True or case.get("failures") != []:
            failures.append(f"{role} length {length}: case did not pass cleanly")
        fixture = case.get("fixture")
        if not isinstance(fixture, dict):
            failures.append(f"{role} length {length}: fixture is missing")
        else:
            expected_fixture = build_fixture(
                length,
                DEFAULT_VOCAB_SIZE,
                salt=89,
                seed=LAYERWISE_FIXTURE_SEED,
            )
            expected_fixture.pop("input_ids")
            if fixture != expected_fixture:
                failures.append(f"{role} length {length}: fixture contract differs")
        finish_reason = case.get("finish_reason")
        finish_type = (
            finish_reason.get("type")
            if isinstance(finish_reason, dict)
            else finish_reason
        )
        if finish_type != "length":
            failures.append(f"{role} length {length}: finish_reason is not length")
        if case.get("prompt_tokens") != length:
            failures.append(f"{role} length {length}: prompt token count differs")
        if case.get("completion_tokens") != DEFAULT_BOUNDARY_OUTPUT:
            failures.append(f"{role} length {length}: completion token count differs")

        summaries = case.get("layerwise_summaries")
        expected_rounds = math.ceil(length / LAYERWISE_CHUNK_SIZE)
        if expected_threshold == 1:
            if not isinstance(summaries, list):
                failures.append(
                    f"{role} length {length}: layerwise summaries are missing"
                )
            else:
                progress_failures = validate_layerwise_progress(
                    case.get("previous_layerwise_summary"),
                    summaries,
                    expected_rounds=expected_rounds,
                )
                failures.extend(
                    f"{role} length {length}: {failure}"
                    for failure in progress_failures
                )
            if case.get("expected_layerwise_rounds") != expected_rounds:
                failures.append(
                    f"{role} length {length}: expected-round evidence differs"
                )
        elif summaries != [] or case.get("expected_layerwise_rounds") != 0:
            failures.append(
                f"{role} length {length}: threshold-0 layerwise evidence is invalid"
            )
        log_window = case.get("server_log_window")
        if not isinstance(log_window, dict):
            failures.append(f"{role} length {length}: log-window evidence is missing")
        else:
            start_offset = log_window.get("start_offset")
            end_offset = log_window.get("end_offset")
            window_sha256 = log_window.get("sha256")
            if (
                isinstance(start_offset, bool)
                or not isinstance(start_offset, int)
                or isinstance(end_offset, bool)
                or not isinstance(end_offset, int)
                or start_offset < 0
                or end_offset < start_offset
            ):
                failures.append(
                    f"{role} length {length}: log-window offsets are invalid"
                )
            if (
                not isinstance(window_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", window_sha256) is None
            ):
                failures.append(f"{role} length {length}: log-window digest is invalid")
            if log_window.get("fatal_lines") != []:
                failures.append(
                    f"{role} length {length}: log-window fatal evidence is not empty"
                )

        output_ids = case.get("output_ids")
        valid_output = (
            isinstance(output_ids, list)
            and len(output_ids) == DEFAULT_BOUNDARY_OUTPUT
            and all(
                not isinstance(token_id, bool)
                and isinstance(token_id, int)
                and 0 <= token_id < DEFAULT_VOCAB_SIZE
                for token_id in output_ids
            )
        )
        if not valid_output:
            failures.append(f"{role} length {length}: output token IDs are invalid")
        elif case.get("output_sha256") != _sha256_token_rows([output_ids]):
            failures.append(f"{role} length {length}: output digest is invalid")

        top = case.get("top_logprobs")
        valid_top = isinstance(top, list) and len(top) == DEFAULT_BOUNDARY_OUTPUT
        if valid_top:
            for step_index, step in enumerate(top):
                if (
                    not isinstance(step, list)
                    or len(step) != PREFILL_PARITY_COLLECTION_TOP_K
                ):
                    valid_top = False
                    break
                seen: set[int] = set()
                for item in step:
                    if not isinstance(item, dict):
                        valid_top = False
                        break
                    token_id = item.get("token_id")
                    logprob = item.get("logprob")
                    if (
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or not 0 <= token_id < DEFAULT_VOCAB_SIZE
                        or token_id in seen
                        or isinstance(logprob, bool)
                        or not isinstance(logprob, (int, float))
                        or not math.isfinite(float(logprob))
                    ):
                        valid_top = False
                        break
                    seen.add(token_id)
                if not valid_top:
                    break
                if valid_output:
                    generated_id = output_ids[step_index]
                    by_id = {item["token_id"]: float(item["logprob"]) for item in step}
                    if generated_id not in by_id or by_id[generated_id] != max(
                        by_id.values()
                    ):
                        valid_top = False
                        break
        if not valid_top:
            failures.append(f"{role} length {length}: top-logprob evidence is invalid")
    return failures, cases


def compare_prefill_parity_payloads(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    failures, baseline_cases = _validate_prefill_collection(
        baseline, role="threshold-0", expected_threshold=0
    )
    candidate_failures, candidate_cases = _validate_prefill_collection(
        candidate, role="threshold-1", expected_threshold=1
    )
    failures.extend(candidate_failures)
    diagnostics: list[dict[str, Any]] = []

    comparable_lengths = (
        sorted(set(baseline_cases) & set(candidate_cases)) if not failures else []
    )
    for length in comparable_lengths:
        expected = baseline_cases[length]
        actual = candidate_cases[length]
        if expected.get("fixture") != actual.get("fixture"):
            failures.append(f"length {length}: input fixture differs")
        expected_ids = expected.get("output_ids")
        actual_ids = actual.get("output_ids")
        if expected_ids != actual_ids:
            failures.append(f"length {length}: greedy output token IDs differ")
        expected_top = expected.get("top_logprobs")
        actual_top = actual.get("top_logprobs")
        if not isinstance(expected_top, list) or not isinstance(actual_top, list):
            failures.append(f"length {length}: top-logprob evidence is missing")
            continue
        for step_index, (expected_step, actual_step) in enumerate(
            zip(expected_top, actual_top)
        ):
            if not isinstance(expected_step, list) or not isinstance(actual_step, list):
                failures.append(f"length {length} step {step_index}: invalid top list")
                continue
            actual_by_id = {
                int(item["token_id"]): float(item["logprob"])
                for item in actual_step
                if isinstance(item, dict)
                and isinstance(item.get("token_id"), int)
                and not isinstance(item.get("token_id"), bool)
                and isinstance(item.get("logprob"), (int, float))
                and not isinstance(item.get("logprob"), bool)
            }
            reference = expected_step[:PREFILL_PARITY_COMPARISON_TOP_K]
            missing = [
                int(item["token_id"])
                for item in reference
                if isinstance(item, dict) and int(item["token_id"]) not in actual_by_id
            ]
            numerical_failures = 0
            max_abs = 0.0
            max_relative = 0.0
            for item in reference:
                token_id = int(item["token_id"])
                if token_id not in actual_by_id:
                    continue
                expected_value = float(item["logprob"])
                actual_value = actual_by_id[token_id]
                delta = abs(expected_value - actual_value)
                relative = delta / max(abs(expected_value), 1e-6)
                max_abs = max(max_abs, delta)
                max_relative = max(max_relative, relative)
                if delta > PREFILL_PARITY_LOGPROB_ATOL + (
                    PREFILL_PARITY_LOGPROB_RTOL * abs(expected_value)
                ):
                    numerical_failures += 1
            diagnostics.append(
                {
                    "input_tokens": length,
                    "step": step_index,
                    "missing_threshold_0_top_ids": missing,
                    "numerical_failures": numerical_failures,
                    "max_abs": max_abs,
                    "max_relative": max_relative,
                }
            )
            if missing:
                failures.append(
                    f"length {length} step {step_index}: {len(missing)} "
                    "threshold-0 top-64 IDs are absent from threshold-1 top-256"
                )
            if numerical_failures:
                failures.append(
                    f"length {length} step {step_index}: {numerical_failures} "
                    "top logprobs exceed atol=0.05, rtol=0.05"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_prefill_threshold_comparison",
        "created_unix": time.time(),
        "contract": {
            "baseline_threshold": 0,
            "candidate_threshold": 1,
            "lengths": list(LAYERWISE_LENGTHS),
            "output_tokens": DEFAULT_BOUNDARY_OUTPUT,
            "fixture_seed": LAYERWISE_FIXTURE_SEED,
            "tp_size": REQUIRED_TP_SIZE,
            "vocab_size": DEFAULT_VOCAB_SIZE,
            "collection_top_k": PREFILL_PARITY_COLLECTION_TOP_K,
            "comparison_top_k": PREFILL_PARITY_COMPARISON_TOP_K,
            "logprob_atol": PREFILL_PARITY_LOGPROB_ATOL,
            "logprob_rtol": PREFILL_PARITY_LOGPROB_RTOL,
            "require_exact_greedy_tokens": True,
        },
        "diagnostics": diagnostics,
        "failures": failures,
        "pass": not failures,
    }


def run_compare_prefill_parity(args: argparse.Namespace) -> int:
    baseline_path = Path(args.threshold_0)
    candidate_path = Path(args.threshold_1)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload = compare_prefill_parity_payloads(baseline, candidate)
    payload["inputs"] = {
        "threshold_0": str(baseline_path.resolve()),
        "threshold_1": str(candidate_path.resolve()),
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def run_buckets(args: argparse.Namespace) -> int:
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    contract_compliant = (
        args.vocab_size == DEFAULT_VOCAB_SIZE
        and args.fixture_seed == FIXTURE_SEED
        and args.input_length == BUCKET_INPUT_TOKENS
        and args.output_length == BUCKET_OUTPUT_TOKENS
        and args.top_k == BUCKET_TOP_K
    )
    if not contract_compliant:
        failures.append(
            "exact-bucket parameters differ from the frozen Session-C contract"
        )
    for batch_size in (1, 2, 4):
        fixtures = [
            build_fixture(
                args.input_length,
                args.vocab_size,
                salt=401 + batch_size * 17 + row,
                seed=args.fixture_seed,
            )
            for row in range(batch_size)
        ]
        rows = [fixture.pop("input_ids") for fixture in fixtures]
        try:
            if not args.no_flush_cache:
                flush_cache(args.base_url, timeout=args.request_timeout)
            metrics_before = fetch_metrics(args.base_url, timeout=args.request_timeout)
            graph_before = decode_graph_counter(metrics_before)
            response = _post_batch(
                args.base_url,
                rows,
                output_tokens=args.output_length,
                top_k=args.top_k,
                timeout=args.request_timeout,
                vocab_size=args.vocab_size,
            )
            metrics_after = fetch_metrics(args.base_url, timeout=args.request_timeout)
            graph_after = decode_graph_counter(metrics_after)
            counters_before = cuda_graph_counters_by_rank(metrics_before)
            counters_after = cuda_graph_counters_by_rank(metrics_after)
            if graph_before is None or graph_after is None:
                raise RuntimeError("decode CUDA Graph Prometheus counter is missing")
            graph_delta = graph_after - graph_before
            minimum_delta = max(args.output_length - 1, 1)
            graph_gate = validate_graph_counter_progress(
                counters_before,
                counters_after,
                tp_size=args.tp_size,
                minimum_graph_delta_per_rank=minimum_delta,
            )
            if not graph_gate["pass"]:
                raise RuntimeError("; ".join(graph_gate["failures"]))
            output_rows = [
                [int(token_id) for token_id in row.get("output_ids", [])]
                for row in response
            ]
            top_logprobs = [
                _normalize_top_logprobs(
                    (row.get("meta_info") or {}).get("output_top_logprobs"),
                    args.top_k,
                )
                for row in response
            ]
            case = {
                "batch_size": batch_size,
                "fixtures": fixtures,
                "input_rows_sha256": _sha256_token_rows(rows),
                "output_rows": output_rows,
                "output_rows_sha256": _sha256_token_rows(output_rows),
                "top_k": args.top_k,
                "top_logprobs": top_logprobs,
                "decode_cuda_graph_counter": {
                    "before": graph_before,
                    "after": graph_after,
                    "delta": graph_delta,
                    "minimum_delta": minimum_delta,
                    "per_rank_gate": graph_gate,
                },
                "failures": [],
                "pass": True,
            }
        except Exception as error:
            case = {
                "batch_size": batch_size,
                "fixtures": fixtures,
                "input_rows_sha256": _sha256_token_rows(rows),
                "failures": [f"{type(error).__name__}: {error}"],
                "pass": False,
            }
        cases.append(case)
        failures.extend(
            f"batch {batch_size}: {failure}" for failure in case["failures"]
        )
        if not case["pass"] and args.stop_on_failure:
            break

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "session_c_exact_cuda_graph_buckets",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "contract": {
            "batch_sizes": [1, 2, 4],
            "input_tokens": args.input_length,
            "output_tokens": args.output_length,
            "top_k": args.top_k,
            "vocab_size": args.vocab_size,
            "fixture_seed": args.fixture_seed,
            "require_decode_cuda_graph": True,
            "compliant": contract_compliant,
        },
        "cases": cases,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def compare_bucket_payloads(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    if baseline.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported baseline schema")
    if candidate.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported candidate schema")
    for role, payload in (("baseline", baseline), ("candidate", candidate)):
        if payload.get("kind") != "session_c_exact_cuda_graph_buckets":
            failures.append(f"{role} payload kind is invalid")
        contract = payload.get("contract", {})
        if contract.get("compliant") is not True:
            failures.append(f"{role} bucket contract is not compliant")
        if contract.get("batch_sizes") != [1, 2, 4]:
            failures.append(f"{role} bucket set contract is invalid")
        if contract.get("input_tokens") != BUCKET_INPUT_TOKENS:
            failures.append(f"{role} input-token contract is invalid")
        if contract.get("output_tokens") != BUCKET_OUTPUT_TOKENS:
            failures.append(f"{role} output-token contract is invalid")
        if contract.get("top_k") != BUCKET_TOP_K:
            failures.append(f"{role} top-k contract is invalid")
    if baseline.get("contract") != candidate.get("contract"):
        failures.append("baseline and candidate bucket contracts differ")
    if not baseline.get("pass"):
        failures.append("baseline bucket collection did not pass")
    if not candidate.get("pass"):
        failures.append("candidate bucket collection did not pass")
    baseline_cases = {
        int(case["batch_size"]): case for case in baseline.get("cases", [])
    }
    candidate_cases = {
        int(case["batch_size"]): case for case in candidate.get("cases", [])
    }
    if set(baseline_cases) != {1, 2, 4} or set(candidate_cases) != {1, 2, 4}:
        failures.append("both collections must contain exact buckets 1, 2, and 4")
    for batch_size in sorted(set(baseline_cases) & set(candidate_cases)):
        expected = baseline_cases[batch_size]
        actual = candidate_cases[batch_size]
        if expected.get("pass") is not True or actual.get("pass") is not True:
            failures.append(f"batch {batch_size}: case did not pass")
        if expected.get("input_rows_sha256") != actual.get("input_rows_sha256"):
            failures.append(f"batch {batch_size}: input fixture digest mismatch")
        if expected.get("output_rows") != actual.get("output_rows"):
            failures.append(f"batch {batch_size}: generated output IDs differ")
        if expected.get("top_k") != actual.get("top_k"):
            failures.append(f"batch {batch_size}: top-k contract differs")
        for role, case in (("baseline", expected), ("candidate", actual)):
            rows = case.get("output_rows")
            top = case.get("top_logprobs")
            if (
                not isinstance(rows, list)
                or len(rows) != batch_size
                or any(len(row) != BUCKET_OUTPUT_TOKENS for row in rows)
            ):
                failures.append(f"batch {batch_size}: {role} output shape is invalid")
            if (
                not isinstance(top, list)
                or len(top) != batch_size
                or any(
                    len(steps) != BUCKET_OUTPUT_TOKENS
                    or any(len(items) != BUCKET_TOP_K for items in steps)
                    for steps in top
                )
            ):
                failures.append(
                    f"batch {batch_size}: {role} top-logprob shape is invalid"
                )
        expected_top = expected.get("top_logprobs")
        actual_top = actual.get("top_logprobs")
        if expected_top is None or actual_top is None:
            failures.append(f"batch {batch_size}: top-logprob evidence is missing")
        elif expected_top != actual_top:
            failures.append(f"batch {batch_size}: top logprobs differ")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "session_c_exact_cuda_graph_bucket_comparison",
        "created_unix": time.time(),
        "failures": failures,
        "pass": not failures,
    }


def run_compare_buckets(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload = compare_bucket_payloads(baseline, candidate)
    payload["inputs"] = {
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def _add_common_request_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30100")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    parser.add_argument("--request-timeout", type=float, default=86_400.0)
    parser.add_argument("--no-flush-cache", action="store_true")
    parser.add_argument("--tp-size", type=int, default=8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    layerwise = subparsers.add_parser(
        "layerwise", help="run the exact 128/4096/4097/8193 boundary matrix"
    )
    _add_common_request_args(layerwise)
    layerwise.add_argument("--server-log", required=True)
    layerwise.add_argument(
        "--lengths", type=int, nargs="+", default=list(LAYERWISE_LENGTHS)
    )
    layerwise.add_argument("--chunk-size", type=int, default=LAYERWISE_CHUNK_SIZE)
    layerwise.add_argument("--output-length", type=int, default=DEFAULT_BOUNDARY_OUTPUT)
    layerwise.add_argument("--stop-on-failure", action="store_true")
    layerwise.set_defaults(
        func=run_layerwise,
        fixture_seed=LAYERWISE_FIXTURE_SEED,
    )

    prefill_parity = subparsers.add_parser(
        "prefill-parity",
        help="collect one exact ordinary/layerwise prefill threshold side",
    )
    _add_common_request_args(prefill_parity)
    prefill_parity.add_argument("--server-log", required=True)
    prefill_parity.add_argument("--threshold", type=int, choices=(0, 1), required=True)
    prefill_parity.add_argument(
        "--lengths", type=int, nargs="+", default=list(LAYERWISE_LENGTHS)
    )
    prefill_parity.add_argument("--chunk-size", type=int, default=LAYERWISE_CHUNK_SIZE)
    prefill_parity.add_argument(
        "--output-length", type=int, default=DEFAULT_BOUNDARY_OUTPUT
    )
    prefill_parity.add_argument(
        "--top-k", type=int, default=PREFILL_PARITY_COLLECTION_TOP_K
    )
    prefill_parity.add_argument("--stop-on-failure", action="store_true")
    prefill_parity.set_defaults(
        func=run_prefill_parity,
        fixture_seed=LAYERWISE_FIXTURE_SEED,
    )

    compare_prefill = subparsers.add_parser(
        "compare-prefill-parity",
        help="compare frozen threshold-0 and threshold-1 prefill collections",
    )
    compare_prefill.add_argument("--threshold-0", required=True)
    compare_prefill.add_argument("--threshold-1", required=True)
    compare_prefill.add_argument("--output", required=True)
    compare_prefill.set_defaults(func=run_compare_prefill_parity)

    long_context = subparsers.add_parser(
        "long", help="run the final integrated 500000-input + 1024-output gate"
    )
    _add_common_request_args(long_context)
    long_context.add_argument("--server-log", required=True)
    long_context.add_argument("--input-length", type=int, default=MIN_LONG_INPUT)
    long_context.add_argument("--output-length", type=int, default=MIN_LONG_OUTPUT)
    long_context.add_argument("--chunk-size", type=int, default=LAYERWISE_CHUNK_SIZE)
    long_context.set_defaults(func=run_long)

    buckets = subparsers.add_parser(
        "buckets", help="exercise exact decode CUDA Graph buckets 1/2/4"
    )
    _add_common_request_args(buckets)
    buckets.add_argument("--input-length", type=int, default=BUCKET_INPUT_TOKENS)
    buckets.add_argument("--output-length", type=int, default=BUCKET_OUTPUT_TOKENS)
    buckets.add_argument("--top-k", type=int, default=BUCKET_TOP_K)
    buckets.add_argument("--stop-on-failure", action="store_true")
    buckets.set_defaults(func=run_buckets)

    compare = subparsers.add_parser(
        "compare-buckets", help="compare baseline/candidate bucket output IDs"
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=run_compare_buckets)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
