#!/usr/bin/env python3
"""Session-C single-concurrency HTTP benchmark and acceptance collector.

This script intentionally does not import SGLang.  It drives the native
``/generate`` endpoint with deterministic token-id inputs and writes a
machine-readable record suitable for comparing eager and CUDA Graph servers.

The current SGLang response always reports ``meta_info.e2e_latency``.  With
``--enable-metrics`` it additionally reports ``meta_info.decode_throughput``
using this server-side definition::

    completion_tokens / (finished_time - first_token_time)

There are no direct TTFT or ITL response fields.  When both server fields are
present, this collector recovers the server decode interval from that exact
definition, then derives server TTFT and mean ITL.  Missing or inconsistent
fields remain ``null`` and fail the performance gate.  Client streaming clocks
are retained only as diagnostics and never substitute for server metrics.

Examples::

    python scripts/glm5_next_session_c_benchmark.py collect \
      --label eager --base-url http://127.0.0.1:30100 \
      --output /mnt/.../session_c/perf/eager.json

    python scripts/glm5_next_session_c_benchmark.py collect \
      --label graph --base-url http://127.0.0.1:30101 \
      --output /mnt/.../session_c/perf/graph.json

    python scripts/glm5_next_session_c_benchmark.py compare \
      --eager /mnt/.../session_c/perf/eager.json \
      --graph /mnt/.../session_c/perf/graph.json \
      --output /mnt/.../session_c/perf/graph-vs-eager.json

    python scripts/glm5_next_session_c_benchmark.py long \
      --label layerwise-500k-1024 --base-url http://127.0.0.1:30101 \
      --output /mnt/.../session_c/perf/layerwise-500k-1024.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_VERSION = 1
FIXTURE_SEED = 20260812
DEFAULT_VOCAB_SIZE = 154880
DEFAULT_WARMUPS = 2
DEFAULT_MEASURED = 5
DEFAULT_SHORT_INPUT = 128
DEFAULT_SHORT_OUTPUT = 128
DEFAULT_DECODE_INPUT = 4096
DEFAULT_DECODE_OUTPUT = 256
DEFAULT_LONG_INPUT = 500_000
DEFAULT_LONG_OUTPUT = 1024

SERVER_METRIC_DERIVATION = {
    "server_e2e_s": "meta_info.e2e_latency",
    "server_output_throughput_tps": "meta_info.decode_throughput",
    "server_decode_duration_s": (
        "meta_info.completion_tokens / meta_info.decode_throughput"
    ),
    "server_ttft_s": ("meta_info.e2e_latency - server_decode_duration_s"),
    "server_mean_itl_s": (
        "server_decode_duration_s / (meta_info.completion_tokens - 1)"
    ),
}

REQUIRED_MEASURED_METRICS = (
    "wall_e2e_s",
    "server_e2e_s",
    "server_ttft_s",
    "server_mean_itl_s",
    "server_output_throughput_tps",
)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256_token_ids(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def build_fixture(
    length: int,
    vocab_size: int,
    *,
    salt: int,
    seed: int = FIXTURE_SEED,
) -> dict[str, Any]:
    """Build a deterministic tokenizer-independent token-id fixture."""
    if length <= 0:
        raise ValueError(f"input length must be positive, got {length}")
    if vocab_size < 4096:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    span = vocab_size - 2048
    token_ids = [
        1024 + ((index * 7_919 + salt * 104_729 + seed) % span)
        for index in range(length)
    ]
    return {
        "input_tokens": length,
        "vocab_size": vocab_size,
        "fixture_seed": seed,
        "salt": salt,
        "input_sha256": _sha256_token_ids(token_ids),
        "input_ids": token_ids,
    }


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile (NumPy's default method)."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [record.get(key) for record in records]
    finite = [float(value) for value in values if _finite_number(value)]
    complete = len(finite) == len(records) and bool(records)
    return {
        "count": len(finite),
        "expected_count": len(records),
        "complete": complete,
        "median": statistics.median(finite) if complete else None,
        "p95": percentile(finite, 95.0) if complete else None,
        "values": finite,
    }


def extract_server_metrics(
    meta_info: dict[str, Any], completion_tokens: int
) -> dict[str, Any]:
    """Extract only genuine server timing and exact algebraic derivatives."""
    result: dict[str, Any] = {
        "server_e2e_s": None,
        "server_decode_duration_s": None,
        "server_ttft_s": None,
        "server_mean_itl_s": None,
        "server_output_throughput_tps": None,
        "server_metric_errors": [],
        "server_metric_derivation": SERVER_METRIC_DERIVATION,
    }
    e2e = meta_info.get("e2e_latency")
    throughput = meta_info.get("decode_throughput")

    if not _finite_number(e2e) or float(e2e) <= 0.0:
        result["server_metric_errors"].append(
            "missing or invalid meta_info.e2e_latency"
        )
    else:
        result["server_e2e_s"] = float(e2e)

    if not _finite_number(throughput) or float(throughput) <= 0.0:
        result["server_metric_errors"].append(
            "missing or invalid meta_info.decode_throughput; "
            "launch SGLang with --enable-metrics"
        )
        return result
    result["server_output_throughput_tps"] = float(throughput)

    if completion_tokens <= 1:
        result["server_metric_errors"].append(
            "at least two completion tokens are required for ITL"
        )
        return result
    if result["server_e2e_s"] is None:
        return result

    decode_duration = completion_tokens / float(throughput)
    ttft = float(e2e) - decode_duration
    tolerance = max(float(e2e) * 1e-9, 1e-9)
    if not math.isfinite(decode_duration) or decode_duration <= 0.0:
        result["server_metric_errors"].append(
            "derived server decode duration is invalid"
        )
        return result
    if ttft < -tolerance or ttft > float(e2e) + tolerance:
        result["server_metric_errors"].append(
            "server timing fields are inconsistent: recovered TTFT is outside "
            "the E2E interval"
        )
        return result

    result["server_decode_duration_s"] = decode_duration
    result["server_ttft_s"] = max(ttft, 0.0)
    result["server_mean_itl_s"] = decode_duration / (completion_tokens - 1)
    return result


def _post_json_request(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return opener(request, timeout=timeout)


def flush_cache(
    base_url: str,
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Flush before every request so repeated prompts cannot hit prefix cache."""
    with _post_json_request(
        f"{base_url.rstrip('/')}/flush_cache",
        {},
        timeout=timeout,
        opener=opener,
    ) as response:
        status = getattr(response, "status", response.getcode())
        payload = response.read()
    if status < 200 or status >= 300:
        raise RuntimeError(f"flush_cache returned HTTP {status}: {payload[:512]!r}")


def _update_output_ids(
    output_ids: list[int], raw_ids: Any, previous_count: int, current_count: int
) -> list[int]:
    chunk_ids = [int(token_id) for token_id in (raw_ids or [])]
    if len(chunk_ids) == current_count:
        return chunk_ids
    delta = current_count - previous_count
    if delta > 0 and len(chunk_ids) >= delta:
        output_ids.extend(chunk_ids[-delta:])
    return output_ids


def run_streaming_request(
    base_url: str,
    input_ids: list[int],
    output_tokens: int,
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one request and return raw, client, and server timing evidence."""
    request_body = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "stream": True,
    }
    started = clock()
    first_token_at: float | None = None
    token_arrival_times: list[float] = []
    output_ids: list[int] = []
    completion_tokens = 0
    final_meta: dict[str, Any] = {}
    finish_reason: Any = None
    stream_had_multi_token_chunk = False

    try:
        response_context = _post_json_request(
            f"{base_url.rstrip('/')}/generate",
            request_body,
            timeout=timeout,
            opener=opener,
        )
        with response_context as response:
            status = getattr(response, "status", response.getcode())
            if status < 200 or status >= 300:
                error_body = response.read()
                raise RuntimeError(
                    f"generate returned HTTP {status}: {error_body[:1024]!r}"
                )
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data:"):
                    continue
                raw_data = line[5:].strip()
                if raw_data == b"[DONE]":
                    break
                message = json.loads(raw_data)
                meta = message.get("meta_info") or {}
                current_count = int(meta.get("completion_tokens") or 0)
                if current_count > completion_tokens:
                    arrived = clock()
                    delta = current_count - completion_tokens
                    if delta != 1:
                        stream_had_multi_token_chunk = True
                    if first_token_at is None:
                        first_token_at = arrived
                    token_arrival_times.extend([arrived] * delta)
                    output_ids = _update_output_ids(
                        output_ids,
                        message.get("output_ids"),
                        completion_tokens,
                        current_count,
                    )
                    completion_tokens = current_count
                final_meta = meta
                if meta.get("finish_reason") is not None:
                    finish_reason = meta["finish_reason"]
    except urllib.error.HTTPError as error:
        detail = error.read()[:1024]
        raise RuntimeError(
            f"generate returned HTTP {error.code}: {detail!r}"
        ) from error

    finished = clock()
    if first_token_at is None:
        raise RuntimeError("stream ended without an output token")
    prompt_tokens = int(final_meta.get("prompt_tokens") or 0)
    if prompt_tokens != len(input_ids):
        raise RuntimeError(
            f"server reported {prompt_tokens} prompt tokens, expected {len(input_ids)}"
        )
    if completion_tokens != output_tokens or len(output_ids) != output_tokens:
        raise RuntimeError(
            "incomplete output: "
            f"completion_tokens={completion_tokens}, output_ids={len(output_ids)}, "
            f"expected={output_tokens}"
        )
    if finish_reason is None:
        raise RuntimeError("final response did not contain a finish_reason")

    wall_e2e = finished - started
    client_ttft = first_token_at - started
    client_itls: list[float] = []
    if not stream_had_multi_token_chunk and len(token_arrival_times) > 1:
        client_itls = [
            later - earlier
            for earlier, later in zip(token_arrival_times, token_arrival_times[1:])
        ]
    record: dict[str, Any] = {
        "success": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_ids": output_ids,
        "output_sha256": _sha256_token_ids(output_ids),
        "finish_reason": finish_reason,
        "wall_e2e_s": wall_e2e,
        "client_ttft_s": client_ttft,
        "client_mean_itl_s": (statistics.mean(client_itls) if client_itls else None),
        "client_output_throughput_tps": completion_tokens / wall_e2e,
        "client_stream_itl_complete": bool(client_itls),
        "raw_server_meta": {
            key: final_meta.get(key)
            for key in (
                "e2e_latency",
                "decode_throughput",
                "request_received_ts",
                "request_finished_ts",
                "forward_entry_time",
                "prefill_finished_time",
                "queue_time",
                "prefill_waiting_latency",
                "prefill_launch_latency",
                "prompt_tokens",
                "completion_tokens",
            )
            if key in final_meta
        },
    }
    record.update(extract_server_metrics(final_meta, completion_tokens))
    return record


def summarize_records(
    records: list[dict[str, Any]], *, expected_count: int
) -> dict[str, Any]:
    successful = [record for record in records if record.get("success")]
    request_complete = (
        len(records) == expected_count and len(successful) == expected_count
    )
    metric_summaries = {
        key: _metric_summary(successful, key) for key in REQUIRED_MEASURED_METRICS
    }
    output_digest_values = [record.get("output_sha256") for record in successful]
    output_evidence_complete = all(
        isinstance(value, str) and bool(value) for value in output_digest_values
    )
    output_digests = set(output_digest_values) if output_evidence_complete else set()
    failures = [
        str(record.get("error", "request failed"))
        for record in records
        if not record.get("success")
    ]
    for key, summary in metric_summaries.items():
        if not summary["complete"]:
            failures.append(f"measured metric {key} is incomplete")
    if not output_evidence_complete:
        failures.append("measured output digest evidence is incomplete")
    deterministic = (
        request_complete and output_evidence_complete and len(output_digests) == 1
    )
    if not deterministic:
        failures.append("measured output token ids are not deterministic")
    return {
        "expected_measured_requests": expected_count,
        "successful_measured_requests": len(successful),
        "requests_complete": request_complete,
        "deterministic_output_ids": deterministic,
        "output_digest_evidence_complete": output_evidence_complete,
        "metrics": metric_summaries,
        "failures": failures,
        "pass": not failures,
    }


def _failed_record(error: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "error": f"{type(error).__name__}: {error}",
        **{key: None for key in REQUIRED_MEASURED_METRICS},
    }


def collect_shape(
    *,
    name: str,
    role: str,
    base_url: str,
    input_tokens: int,
    output_tokens: int,
    vocab_size: int,
    fixture_seed: int,
    salt: int,
    warmups: int,
    measured: int,
    timeout: float,
    should_flush_cache: bool,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    fixture = build_fixture(input_tokens, vocab_size, salt=salt, seed=fixture_seed)
    input_ids = fixture.pop("input_ids")

    def invoke(phase: str, index: int) -> dict[str, Any]:
        try:
            if should_flush_cache:
                flush_cache(base_url, timeout=timeout, opener=opener)
            record = run_streaming_request(
                base_url,
                input_ids,
                output_tokens,
                timeout=timeout,
                opener=opener,
            )
        except Exception as error:  # Preserve partial acceptance evidence.
            record = _failed_record(error)
        record.update({"phase": phase, "index": index})
        return record

    warmup_records = [invoke("warmup", index) for index in range(warmups)]
    measured_records = [invoke("measured", index) for index in range(measured)]
    summary = summarize_records(measured_records, expected_count=measured)
    warmups_complete = len(warmup_records) == warmups and all(
        record.get("success") for record in warmup_records
    )
    failures = list(summary["failures"])
    if not warmups_complete:
        failures.append("one or more warmup requests failed")
    return {
        "name": name,
        "role": role,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "fixture": fixture,
        "warmups": warmup_records,
        "measured": measured_records,
        "summary": summary,
        "failures": failures,
        "pass": warmups_complete and summary["pass"],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def run_collect(args: argparse.Namespace) -> int:
    shapes = [
        collect_shape(
            name=f"short_{args.short_input}_{args.short_output}",
            role="short",
            base_url=args.base_url,
            input_tokens=args.short_input,
            output_tokens=args.short_output,
            vocab_size=args.vocab_size,
            fixture_seed=args.fixture_seed,
            salt=17,
            warmups=args.warmups,
            measured=args.measured,
            timeout=args.request_timeout,
            should_flush_cache=not args.no_flush_cache,
        ),
        collect_shape(
            name=f"decode_{args.decode_input}_{args.decode_output}",
            role="decode",
            base_url=args.base_url,
            input_tokens=args.decode_input,
            output_tokens=args.decode_output,
            vocab_size=args.vocab_size,
            fixture_seed=args.fixture_seed,
            salt=41,
            warmups=args.warmups,
            measured=args.measured,
            timeout=args.request_timeout,
            should_flush_cache=not args.no_flush_cache,
        ),
    ]
    contract_compliant = (
        args.warmups == DEFAULT_WARMUPS
        and args.measured == DEFAULT_MEASURED
        and args.short_input == DEFAULT_SHORT_INPUT
        and args.short_output == DEFAULT_SHORT_OUTPUT
        and args.decode_input == DEFAULT_DECODE_INPUT
        and args.decode_output == DEFAULT_DECODE_OUTPUT
    )
    failures = []
    if not contract_compliant:
        failures.append(
            "collection parameters differ from the frozen Session-C contract"
        )
    failures.extend(
        f"{shape['name']}: {failure}"
        for shape in shapes
        for failure in shape["failures"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_benchmark",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "contract": {
            "request_concurrency": 1,
            "warmups": args.warmups,
            "measured": args.measured,
            "flush_cache_before_each_request": not args.no_flush_cache,
            "server_metrics_required": True,
            "requires_server_enable_metrics": True,
            "metric_derivation": SERVER_METRIC_DERIVATION,
            "compliant": contract_compliant,
        },
        "shapes": shapes,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "pass": payload["pass"],
                "shape_pass": {shape["name"]: shape["pass"] for shape in shapes},
            },
            indent=2,
        )
    )
    return 0 if payload["pass"] else 1


def _shape_by_role(payload: dict[str, Any], role: str) -> dict[str, Any] | None:
    matches = [
        shape for shape in payload.get("shapes", []) if shape.get("role") == role
    ]
    return matches[0] if len(matches) == 1 else None


def _summary_value(
    shape: dict[str, Any] | None, metric: str, statistic: str
) -> float | None:
    if shape is None:
        return None
    value = shape.get("summary", {}).get("metrics", {}).get(metric, {}).get(statistic)
    return float(value) if _finite_number(value) else None


def compare_payloads(eager: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    eager_decode = _shape_by_role(eager, "decode")
    graph_decode = _shape_by_role(graph, "decode")
    failures: list[str] = []
    if eager.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported eager schema")
    if graph.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported graph schema")
    for role, payload in (("eager", eager), ("graph", graph)):
        if payload.get("kind") != "glm5_next_session_c_benchmark":
            failures.append(f"{role} payload kind is invalid")
        if payload.get("pass") is not True:
            failures.append(f"{role} collection did not pass")
        if payload.get("contract", {}).get("compliant") is not True:
            failures.append(f"{role} contract is not compliant")
    if eager.get("contract") != graph.get("contract"):
        failures.append("eager and graph contracts differ")
    if eager_decode is None or graph_decode is None:
        failures.append("each input must contain exactly one decode shape")
    elif (
        eager_decode.get("input_tokens"),
        eager_decode.get("output_tokens"),
        eager_decode.get("fixture", {}).get("input_sha256"),
    ) != (
        graph_decode.get("input_tokens"),
        graph_decode.get("output_tokens"),
        graph_decode.get("fixture", {}).get("input_sha256"),
    ):
        failures.append("eager and graph decode fixtures do not match")

    eager_itl_median = _summary_value(eager_decode, "server_mean_itl_s", "median")
    graph_itl_median = _summary_value(graph_decode, "server_mean_itl_s", "median")
    eager_itl_p95 = _summary_value(eager_decode, "server_mean_itl_s", "p95")
    graph_itl_p95 = _summary_value(graph_decode, "server_mean_itl_s", "p95")
    eager_throughput = _summary_value(
        eager_decode, "server_output_throughput_tps", "median"
    )
    graph_throughput = _summary_value(
        graph_decode, "server_output_throughput_tps", "median"
    )

    def upper_gate(
        name: str, actual: float | None, limit: float | None
    ) -> dict[str, Any]:
        available = actual is not None and limit is not None
        return {
            "name": name,
            "actual": actual,
            "limit": limit,
            "operator": "<=",
            "pass": bool(available and actual <= limit),
            "reason": None if available else "required server metric is null",
        }

    median_gate = upper_gate(
        "graph_median_itl_at_most_0.90_eager",
        graph_itl_median,
        eager_itl_median * 0.90 if eager_itl_median is not None else None,
    )
    p95_gate = upper_gate(
        "graph_p95_itl_at_most_1.05_eager",
        graph_itl_p95,
        eager_itl_p95 * 1.05 if eager_itl_p95 is not None else None,
    )
    best_throughput = (
        max(eager_throughput, graph_throughput)
        if eager_throughput is not None and graph_throughput is not None
        else None
    )
    throughput_available = graph_throughput is not None and best_throughput is not None
    throughput_gate = {
        "name": "graph_throughput_at_least_0.95_best_correct",
        "actual": graph_throughput,
        "limit": best_throughput * 0.95 if best_throughput is not None else None,
        "operator": ">=",
        "pass": bool(
            throughput_available and graph_throughput >= best_throughput * 0.95
        ),
        "reason": None if throughput_available else "required server metric is null",
    }
    gates = [median_gate, p95_gate, throughput_gate]
    failures.extend(gate["name"] for gate in gates if not gate["pass"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_graph_comparison",
        "created_unix": time.time(),
        "eager_label": eager.get("label"),
        "graph_label": graph.get("label"),
        "gates": gates,
        "failures": failures,
        "pass": not failures,
    }


def run_compare(args: argparse.Namespace) -> int:
    eager_path = Path(args.eager)
    graph_path = Path(args.graph)
    eager = json.loads(eager_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    payload = compare_payloads(eager, graph)
    payload["inputs"] = {
        "eager": str(eager_path.resolve()),
        "graph": str(graph_path.resolve()),
    }
    _atomic_write_json(Path(args.output), payload)
    print(
        json.dumps(
            {"output": args.output, "pass": payload["pass"], "gates": payload["gates"]},
            indent=2,
        )
    )
    return 0 if payload["pass"] else 1


def run_long(args: argparse.Namespace) -> int:
    fixture = build_fixture(
        args.input_length,
        args.vocab_size,
        salt=73,
        seed=args.fixture_seed,
    )
    input_ids = fixture.pop("input_ids")
    try:
        if not args.no_flush_cache:
            flush_cache(args.base_url, timeout=args.request_timeout)
        record = run_streaming_request(
            args.base_url,
            input_ids,
            args.output_length,
            timeout=args.request_timeout,
        )
    except Exception as error:
        record = _failed_record(error)
    contract_compliant = (
        args.input_length == DEFAULT_LONG_INPUT
        and args.output_length == DEFAULT_LONG_OUTPUT
    )
    failures = []
    if not record.get("success"):
        failures.append(str(record.get("error", "long request failed")))
    if not contract_compliant:
        failures.append(
            "long request must be exactly 500000 input + 1024 output tokens"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_long_context",
        "created_unix": time.time(),
        "label": args.label,
        "base_url": args.base_url.rstrip("/"),
        "contract": {
            "request_concurrency": 1,
            "input_tokens": DEFAULT_LONG_INPUT,
            "output_tokens": DEFAULT_LONG_OUTPUT,
            "throughput_gate": None,
            "compliant": contract_compliant,
        },
        "fixture": fixture,
        "input_tokens": args.input_length,
        "output_tokens": args.output_length,
        "record": record,
        "failures": failures,
        "pass": not failures,
    }
    _atomic_write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "pass": payload["pass"]}, indent=2))
    return 0 if payload["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    collect = subparsers.add_parser("collect", help="collect short and decode runs")
    collect.add_argument("--label", required=True)
    collect.add_argument("--base-url", default="http://127.0.0.1:30100")
    collect.add_argument("--output", required=True)
    collect.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    collect.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    collect.add_argument("--short-input", type=int, default=DEFAULT_SHORT_INPUT)
    collect.add_argument("--short-output", type=int, default=DEFAULT_SHORT_OUTPUT)
    collect.add_argument("--decode-input", type=int, default=DEFAULT_DECODE_INPUT)
    collect.add_argument("--decode-output", type=int, default=DEFAULT_DECODE_OUTPUT)
    collect.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    collect.add_argument("--measured", type=int, default=DEFAULT_MEASURED)
    collect.add_argument("--request-timeout", type=float, default=3600.0)
    collect.add_argument("--no-flush-cache", action="store_true")
    collect.set_defaults(func=run_collect)

    compare = subparsers.add_parser("compare", help="apply eager/graph gates")
    compare.add_argument("--eager", required=True)
    compare.add_argument("--graph", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=run_compare)

    long_context = subparsers.add_parser("long", help="run one 500K+1024 request")
    long_context.add_argument("--label", required=True)
    long_context.add_argument("--base-url", default="http://127.0.0.1:30100")
    long_context.add_argument("--output", required=True)
    long_context.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    long_context.add_argument("--fixture-seed", type=int, default=FIXTURE_SEED)
    long_context.add_argument("--input-length", type=int, default=DEFAULT_LONG_INPUT)
    long_context.add_argument("--output-length", type=int, default=DEFAULT_LONG_OUTPUT)
    long_context.add_argument("--request-timeout", type=float, default=7200.0)
    long_context.add_argument("--no-flush-cache", action="store_true")
    long_context.set_defaults(func=run_long)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
