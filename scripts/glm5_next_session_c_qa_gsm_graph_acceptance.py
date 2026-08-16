#!/usr/bin/env python3
"""Fail-closed Session-C graph acceptance for natural QA and GSM8K-200.

This driver deliberately talks only to public HTTP endpoints and the checked-in
GSM8K benchmark.  Both modes preserve the exact Prometheus text before and
after inference and require, for every configured TP rank, movement of the
official decode CUDA Graph counter with no positive movement of ``decode_none``.
Decode speed is taken from each native response's
``meta_info.decode_throughput``; inter-token-latency metrics are diagnostic only.

Examples::

    python scripts/glm5_next_session_c_qa_gsm_graph_acceptance.py qa \
      --base-url http://127.0.0.1:30100 --model /mnt/models/GLM-5-Next-0808 \
      --output /mnt/artifacts/qa.json --tp-size 4

    python scripts/glm5_next_session_c_qa_gsm_graph_acceptance.py gsm8k \
      --base-url http://127.0.0.1:30100 --data-path /mnt/data/test.jsonl \
      --result-file /mnt/artifacts/gsm/result.jsonl \
      --raw-result-file /mnt/artifacts/gsm/raw.jsonl \
      --work-dir /mnt/artifacts/gsm/work \
      --output /mnt/artifacts/gsm/acceptance.json --tp-size 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
QA_KIND = "glm5_next_session_c_natural_qa_graph_acceptance"
GSM_KIND = "glm5_next_session_c_gsm8k_200_graph_acceptance"
GRAPH_METRIC = "sglang:cuda_graph_passes_total"
ITL_SUM_METRIC = "sglang:inter_token_latency_seconds_sum"
ITL_COUNT_METRIC = "sglang:inter_token_latency_seconds_count"
GSM_QUESTIONS = 200
GSM_SHOTS = 5
GSM_PARALLEL = 1
GSM_MAX_NEW_TOKENS = 512
QA_SPEED_PROBE_MAX_NEW_TOKENS = 8
DEFAULT_GSM_DATA_SHA256 = (
    "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
)

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+[^\s]+)?$"
)
_METRIC_LABEL = re.compile(
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"'
)


GSM_BENCHMARK_WORKER = r"""
import json
import runpy
import sys
from pathlib import Path

benchmark_path = sys.argv[1]
speed_path = Path(sys.argv[2])
sys.argv = [benchmark_path, *sys.argv[3:]]

from sglang.test import test_utils

original_dump_bench_raw_result = test_utils.dump_bench_raw_result


def dump_bench_raw_result_with_speed(path, states, preds, labels):
    original_dump_bench_raw_result(path, states, preds, labels)
    rows = [
        {
            "prompt_id": prompt_id,
            "meta_info": state.get_meta_info("answer"),
        }
        for prompt_id, state in enumerate(states)
    ]
    encoded = "".join(
        json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    speed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = speed_path.with_suffix(speed_path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(speed_path)


test_utils.dump_bench_raw_result = dump_bench_raw_result_with_speed
runpy.run_path(benchmark_path, run_name="__main__")
""".strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(text, parse_constant=reject_constant)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _decode_prometheus_label(value: str) -> str:
    # Prometheus uses the same escapes needed here (\\, \", and \n) as a JSON
    # string.  json.loads also makes malformed escapes fail closed.
    return json.loads(f'"{value}"')


def parse_prometheus_metrics(text: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        labels_text = match.group("labels") or ""
        labels: dict[str, str] = {}
        for label_match in _METRIC_LABEL.finditer(labels_text):
            name = label_match.group("name")
            if name in labels:
                raise ValueError(
                    f"duplicate label {name!r} on metrics line {line_number}"
                )
            labels[name] = _decode_prometheus_label(label_match.group("value"))
        try:
            value = float(match.group("value"))
        except ValueError:
            value = math.nan
        samples.append(
            {
                "name": match.group("name"),
                "labels": labels,
                "value": value,
                "finite": math.isfinite(value),
                "line_number": line_number,
                "line": raw_line,
            }
        )
    return samples


def analyze_metrics(text: str) -> dict[str, Any]:
    """Extract graph counters and optional diagnostic ITL metrics."""

    failures: list[str] = []
    try:
        samples = parse_prometheus_metrics(text)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "graph_counters": {"decode_cuda_graph": {}, "decode_none": {}},
            "inter_token_latency": {
                "sum": None,
                "count": None,
                "available": False,
                "diagnostics": [],
            },
            "failures": [f"Prometheus parse failed: {error}"],
        }

    graph_counters: dict[str, dict[str, float]] = {
        "decode_cuda_graph": {},
        "decode_none": {},
    }
    for sample in samples:
        if sample["name"] != GRAPH_METRIC:
            continue
        labels = sample["labels"]
        mode = labels.get("mode")
        if mode not in graph_counters:
            continue
        rank = labels.get("tp_rank")
        if rank is None:
            failures.append(f"{GRAPH_METRIC} {mode} sample has no tp_rank label")
            continue
        if not sample["finite"]:
            failures.append(f"{GRAPH_METRIC} {mode} TP rank {rank} is non-finite")
            continue
        if sample["value"] < 0:
            failures.append(f"{GRAPH_METRIC} {mode} TP rank {rank} is negative")
            continue
        if rank in graph_counters[mode]:
            failures.append(f"duplicate {GRAPH_METRIC} {mode} TP rank {rank}")
            continue
        graph_counters[mode][rank] = float(sample["value"])

    itl: dict[str, Any] = {
        "sum": None,
        "count": None,
        "available": False,
        "diagnostics": [],
    }
    for field, metric_name in (
        ("sum", ITL_SUM_METRIC),
        ("count", ITL_COUNT_METRIC),
    ):
        matched = [sample for sample in samples if sample["name"] == metric_name]
        if not matched:
            itl["diagnostics"].append(f"missing optional metric {metric_name}")
            continue
        if any(not sample["finite"] for sample in matched):
            itl["diagnostics"].append(
                f"optional metric {metric_name} contains non-finite data"
            )
            continue
        itl[field] = sum(float(sample["value"]) for sample in matched)
        if not _finite_number(itl[field]):
            itl["diagnostics"].append(
                f"optional metric {metric_name} aggregate is non-finite"
            )
            itl[field] = None
            continue
        if itl[field] < 0:
            itl["diagnostics"].append(f"optional metric {metric_name} is negative")
        if field == "count" and not float(itl[field]).is_integer():
            itl["diagnostics"].append(
                f"optional metric {metric_name} is not an integer count"
            )
    itl["available"] = (
        _finite_number(itl["sum"])
        and _finite_number(itl["count"])
        and itl["sum"] >= 0
        and itl["count"] >= 0
        and float(itl["count"]).is_integer()
    )

    return {
        "graph_counters": graph_counters,
        "inter_token_latency": itl,
        "failures": failures,
    }


def compare_metrics(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tp_size: int,
    minimum_graph_delta_per_rank: float,
) -> dict[str, Any]:
    """Prove graph decode on every TP rank; ITL remains diagnostic only."""

    failures = [
        *(f"before: {failure}" for failure in before.get("failures", [])),
        *(f"after: {failure}" for failure in after.get("failures", [])),
    ]
    if (
        not _finite_number(minimum_graph_delta_per_rank)
        or minimum_graph_delta_per_rank <= 0
    ):
        failures.append("minimum graph delta per rank must be positive and finite")
    expected_ranks = {str(rank) for rank in range(tp_size)}
    before_graph = before.get("graph_counters", {})
    after_graph = after.get("graph_counters", {})
    per_rank: dict[str, dict[str, Any]] = {}

    graph_before_ranks = set(before_graph.get("decode_cuda_graph", {}))
    graph_after_ranks = set(after_graph.get("decode_cuda_graph", {}))
    if graph_before_ranks != expected_ranks or graph_after_ranks != expected_ranks:
        failures.append(
            "decode_cuda_graph TP rank set mismatch: "
            f"expected={sorted(expected_ranks)}, "
            f"before={sorted(graph_before_ranks)}, "
            f"after={sorted(graph_after_ranks)}"
        )

    none_before_ranks = set(before_graph.get("decode_none", {}))
    none_after_ranks = set(after_graph.get("decode_none", {}))
    unexpected_none_ranks = (none_before_ranks | none_after_ranks) - expected_ranks
    if unexpected_none_ranks:
        failures.append(
            f"decode_none contains unexpected TP ranks: {sorted(unexpected_none_ranks)}"
        )

    for rank in sorted(expected_ranks, key=int):
        graph_before = before_graph.get("decode_cuda_graph", {}).get(rank)
        graph_after = after_graph.get("decode_cuda_graph", {}).get(rank)
        # Prometheus does not emit a counter series before its first increment.
        # Therefore an absent decode_none series is semantically zero in either
        # snapshot, including a partially materialized per-rank vector.
        none_before = before_graph.get("decode_none", {}).get(rank, 0.0)
        none_after = after_graph.get("decode_none", {}).get(rank, 0.0)
        graph_delta = (
            graph_after - graph_before
            if graph_before is not None and graph_after is not None
            else None
        )
        none_delta = none_after - none_before
        rank_failures: list[str] = []
        if graph_delta is None:
            rank_failures.append("decode_cuda_graph delta is unavailable")
        elif graph_delta < minimum_graph_delta_per_rank:
            rank_failures.append(
                f"decode_cuda_graph delta {graph_delta} is below "
                f"{minimum_graph_delta_per_rank}"
            )
        if none_delta > 0:
            rank_failures.append(
                f"decode_none delta must not be positive, got {none_delta}"
            )
        failures.extend(f"TP rank {rank}: {failure}" for failure in rank_failures)
        per_rank[rank] = {
            "decode_cuda_graph_before": graph_before,
            "decode_cuda_graph_after": graph_after,
            "decode_cuda_graph_delta": graph_delta,
            "decode_none_before": none_before,
            "decode_none_after": none_after,
            "decode_none_delta": none_delta,
            "pass": not rank_failures,
        }

    before_itl = before.get("inter_token_latency", {})
    after_itl = after.get("inter_token_latency", {})
    sum_before = before_itl.get("sum")
    sum_after = after_itl.get("sum")
    count_before = before_itl.get("count")
    count_after = after_itl.get("count")
    sum_delta = (
        float(sum_after) - float(sum_before)
        if _finite_number(sum_before) and _finite_number(sum_after)
        else None
    )
    count_delta = (
        float(count_after) - float(count_before)
        if _finite_number(count_before) and _finite_number(count_after)
        else None
    )
    decode_tokens_per_second = None
    itl_diagnostics = [
        *(f"before: {item}" for item in before_itl.get("diagnostics", [])),
        *(f"after: {item}" for item in after_itl.get("diagnostics", [])),
    ]
    if sum_delta is None or count_delta is None:
        itl_diagnostics.append("inter-token latency delta is unavailable")
    elif count_delta <= 0:
        itl_diagnostics.append(
            f"inter-token latency count delta is not positive: {count_delta}"
        )
    elif sum_delta <= 0:
        itl_diagnostics.append(
            f"inter-token latency sum delta is not positive: {sum_delta}"
        )
    else:
        decode_tokens_per_second = count_delta / sum_delta
        if not math.isfinite(decode_tokens_per_second):
            itl_diagnostics.append("derived ITL decode token/s is non-finite")
            decode_tokens_per_second = None

    return {
        "pass": not failures,
        "failures": failures,
        "tp_size": tp_size,
        "minimum_graph_delta_per_rank": minimum_graph_delta_per_rank,
        "per_rank": per_rank,
        "inter_token_latency": {
            "sum_before_seconds": sum_before,
            "sum_after_seconds": sum_after,
            "sum_delta_seconds": sum_delta,
            "count_before_tokens": count_before,
            "count_after_tokens": count_after,
            "count_delta_tokens": count_delta,
            "decode_tokens_per_second": decode_tokens_per_second,
            "formula": "count_delta_tokens / sum_delta_seconds",
            "blocking": False,
            "available": decode_tokens_per_second is not None,
            "diagnostics": itl_diagnostics,
        },
    }


def _request_bytes(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[int, bytes, dict[str, str]]:
    try:
        with opener(request, timeout=timeout) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = response.read()
            headers = dict(response.headers.items())
    except urllib.error.HTTPError as error:
        detail = error.read()[:4096]
        raise RuntimeError(
            f"HTTP {error.code} from {request.full_url}: {detail!r}"
        ) from error
    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status} from {request.full_url}: {body[:4096]!r}")
    return status, body, headers


def fetch_metrics(
    base_url: str,
    *,
    timeout: float,
    raw_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/metrics", method="GET")
    status, body, _headers = _request_bytes(request, timeout=timeout, opener=opener)
    text = body.decode("utf-8", errors="strict")
    _atomic_write_bytes(raw_path, body)
    return {
        "http_status": status,
        "raw_path": str(raw_path.resolve()),
        "raw_sha256": _sha256_bytes(body),
        "raw_size_bytes": len(body),
        "raw_text": text,
        **analyze_metrics(text),
    }


def validate_qa_response(
    payload: Any,
    *,
    expected_content: str,
    expected_finish_reason: str,
) -> dict[str, Any]:
    failures: list[str] = []
    content = None
    finish_reason = None
    usage: Any = None
    if not isinstance(payload, dict):
        failures.append("response JSON must be an object")
    else:
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            failures.append("response choices must contain exactly one item")
        else:
            choice = choices[0]
            if not isinstance(choice, dict):
                failures.append("response choice must be an object")
            else:
                if choice.get("index") != 0:
                    failures.append("response choice index must be 0")
                finish_reason = choice.get("finish_reason")
                if finish_reason != expected_finish_reason:
                    failures.append(
                        f"finish_reason must be {expected_finish_reason!r}, "
                        f"got {finish_reason!r}"
                    )
                message = choice.get("message")
                if not isinstance(message, dict):
                    failures.append("response choice.message must be an object")
                else:
                    if message.get("role") != "assistant":
                        failures.append("response message role must be 'assistant'")
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        failures.append("response content must be a non-empty string")
                    elif expected_content not in content:
                        failures.append(
                            f"response content does not contain {expected_content!r}"
                        )

        usage = payload.get("usage")
        if not isinstance(usage, dict):
            failures.append("response usage must be an object")
        else:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if (
                not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or prompt_tokens <= 0
            ):
                failures.append("usage.prompt_tokens must be a positive integer")
            if (
                not isinstance(completion_tokens, int)
                or isinstance(completion_tokens, bool)
                or completion_tokens <= 0
            ):
                failures.append("usage.completion_tokens must be a positive integer")
            if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
                failures.append("usage.total_tokens must be an integer")
            elif (
                isinstance(prompt_tokens, int)
                and not isinstance(prompt_tokens, bool)
                and isinstance(completion_tokens, int)
                and not isinstance(completion_tokens, bool)
                and total_tokens != prompt_tokens + completion_tokens
            ):
                failures.append(
                    "usage.total_tokens must equal prompt_tokens + completion_tokens"
                )

    return {
        "pass": not failures,
        "failures": failures,
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def validate_decode_speed_responses(
    payload: Any,
    *,
    expected_count: int,
    require_prompt_ids: bool,
) -> dict[str, Any]:
    """Require a finite positive native decode throughput for every response."""

    failures: list[str] = []
    if isinstance(payload, dict) and expected_count == 1 and not require_prompt_ids:
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
        failures.append("decode-speed payload must be a response object or row list")

    if len(rows) != expected_count:
        failures.append(
            f"decode-speed response count must be {expected_count}, got {len(rows)}"
        )

    prompt_ids: list[Any] = []
    throughputs: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"decode-speed response {index} must be an object")
            continue
        if require_prompt_ids:
            prompt_ids.append(row.get("prompt_id"))
        meta_info = row.get("meta_info")
        if not isinstance(meta_info, dict):
            failures.append(
                f"decode-speed response {index} meta_info must be an object"
            )
            continue
        decode_throughput = meta_info.get("decode_throughput")
        if not _finite_number(decode_throughput) or float(decode_throughput) <= 0:
            failures.append(
                "decode-speed response "
                f"{index} meta_info.decode_throughput must be finite and positive"
            )
            continue
        throughputs.append(float(decode_throughput))

    if require_prompt_ids and prompt_ids != list(range(expected_count)):
        failures.append(
            f"decode-speed prompt_id sequence must be exactly 0..{expected_count - 1}"
        )

    minimum = min(throughputs) if throughputs else None
    maximum = max(throughputs) if throughputs else None
    mean = None
    median = None
    if throughputs:
        try:
            mean = math.fsum(throughputs) / len(throughputs)
        except OverflowError:
            mean = None
        ordered = sorted(throughputs)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            median = ordered[midpoint]
        else:
            median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
        if not _finite_number(mean):
            failures.append("decode-throughput mean is non-finite")
            mean = None
        if not _finite_number(median):
            failures.append("decode-throughput median is non-finite")
            median = None

    return {
        "pass": not failures,
        "failures": failures,
        "response_count": len(rows),
        "expected_response_count": expected_count,
        "decode_throughput": {
            "source_field": "meta_info.decode_throughput",
            "unit": "tokens/second",
            "valid_response_count": len(throughputs),
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "median": median,
        },
        "prompt_ids_sha256": (
            _sha256_bytes(json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8"))
            if require_prompt_ids
            else None
        ),
    }


def validate_qa_speed_probe(payload: Any) -> dict[str, Any]:
    failures: list[str] = []
    speed = validate_decode_speed_responses(
        payload,
        expected_count=1,
        require_prompt_ids=False,
    )
    failures.extend(speed["failures"])
    output_text = None
    completion_tokens = None
    if isinstance(payload, dict):
        output_text = payload.get("text")
        if not isinstance(output_text, str) or not output_text:
            failures.append("QA decode-speed probe text must be non-empty")
        meta_info = payload.get("meta_info")
        if isinstance(meta_info, dict):
            completion_tokens = meta_info.get("completion_tokens")
            if completion_tokens != QA_SPEED_PROBE_MAX_NEW_TOKENS:
                failures.append(
                    "QA decode-speed probe completion_tokens must be "
                    f"{QA_SPEED_PROBE_MAX_NEW_TOKENS}, got {completion_tokens!r}"
                )

    return {
        "pass": not failures,
        "failures": failures,
        "output_text": output_text,
        "completion_tokens": completion_tokens,
        "speed": speed,
    }


def _read_jsonl_strict(path: Path) -> tuple[list[Any], list[str]]:
    failures: list[str] = []
    rows: list[Any] = []
    if not path.is_file():
        return [], [f"missing JSONL file: {path}"]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [], [f"failed to read JSONL file {path}: {error}"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            failures.append(f"{path}: blank JSONL row {line_number}")
            continue
        try:
            rows.append(_strict_json_loads(line))
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(f"{path}: invalid JSON row {line_number}: {error}")
    return rows, failures


def validate_gsm_decode_speed_artifact(path: Path) -> dict[str, Any]:
    rows, failures = _read_jsonl_strict(path)
    validation = validate_decode_speed_responses(
        rows,
        expected_count=GSM_QUESTIONS,
        require_prompt_ids=True,
    )
    failures.extend(validation["failures"])
    artifact = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    return {
        **validation,
        "pass": not failures,
        "failures": failures,
        "artifact": artifact,
    }


def validate_gsm_artifacts(result_path: Path, raw_result_path: Path) -> dict[str, Any]:
    """Validate exact benchmark cardinality/completion; accuracy has no floor."""

    result_rows, failures = _read_jsonl_strict(result_path)
    raw_rows, raw_failures = _read_jsonl_strict(raw_result_path)
    failures.extend(raw_failures)
    if len(result_rows) != 1:
        failures.append(
            f"result JSONL must contain exactly 1 row, got {len(result_rows)}"
        )
    if len(raw_rows) != GSM_QUESTIONS:
        failures.append(
            f"raw JSONL must contain exactly {GSM_QUESTIONS} rows, got {len(raw_rows)}"
        )

    summary = result_rows[0] if len(result_rows) == 1 else None
    accuracy = None
    if not isinstance(summary, dict):
        if summary is not None:
            failures.append("result row must be an object")
    else:
        expected_scalars = {
            "task": "gsm8k",
            "backend": "srt",
            "num_requests": GSM_QUESTIONS,
        }
        for key, expected in expected_scalars.items():
            if summary.get(key) != expected:
                failures.append(
                    f"result {key} must be {expected!r}, got {summary.get(key)!r}"
                )
        other = summary.get("other")
        if not isinstance(other, dict):
            failures.append("result other must be an object")
        else:
            if other.get("num_questions") != GSM_QUESTIONS:
                failures.append(f"result other.num_questions must be {GSM_QUESTIONS}")
            if other.get("parallel") != GSM_PARALLEL:
                failures.append(f"result other.parallel must be {GSM_PARALLEL}")
        accuracy = summary.get("accuracy")
        if not _finite_number(accuracy) or not 0 <= float(accuracy) <= 1:
            failures.append("result accuracy must be finite and in [0, 1]")
        latency = summary.get("latency")
        if not _finite_number(latency) or float(latency) <= 0:
            failures.append("result latency must be a positive finite number")

    prompt_ids: list[Any] = []
    correct_count = 0
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            failures.append(f"raw row {index} must be an object")
            continue
        prompt_ids.append(row.get("prompt_id"))
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            failures.append(f"raw row {index} prompt must be non-empty")
        if not isinstance(row.get("output"), str) or not row["output"].strip():
            failures.append(
                f"raw row {index} output is not a completed non-empty string"
            )
        if not isinstance(row.get("correct"), bool):
            failures.append(f"raw row {index} correct must be boolean")
        elif row["correct"]:
            correct_count += 1
    if prompt_ids != list(range(GSM_QUESTIONS)):
        failures.append("raw prompt_id sequence must be exactly 0..199")

    raw_accuracy = (
        correct_count / GSM_QUESTIONS if len(raw_rows) == GSM_QUESTIONS else None
    )
    if (
        raw_accuracy is not None
        and _finite_number(accuracy)
        and round(raw_accuracy, 3) != float(accuracy)
    ):
        failures.append(
            f"summary accuracy {accuracy} does not match rounded raw accuracy {raw_accuracy}"
        )

    artifacts: dict[str, Any] = {}
    for name, path in (("result", result_path), ("raw_result", raw_result_path)):
        if path.is_file():
            artifacts[name] = {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            artifacts[name] = {
                "path": str(path.resolve()),
                "sha256": None,
                "size_bytes": None,
            }

    return {
        "pass": not failures,
        "failures": failures,
        "summary": summary,
        "raw_row_count": len(raw_rows),
        "prompt_ids_sha256": _sha256_bytes(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ),
        "accuracy": {
            "value": accuracy,
            "raw_value": raw_accuracy,
            "blocking": False,
            "minimum": None,
        },
        "artifacts": artifacts,
    }


def validate_evidence_envelope(
    payload: Any,
    *,
    expected_kind: str,
    expected_contract: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if not isinstance(payload, dict):
        return ["evidence must be an object"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("kind") != expected_kind:
        failures.append(f"kind must be {expected_kind!r}")
    if payload.get("contract") != expected_contract:
        failures.append(
            "contract does not exactly match the invoked acceptance contract"
        )
    if not isinstance(payload.get("pass"), bool):
        failures.append("pass must be boolean")
    evidence_failures = payload.get("failures")
    if not isinstance(evidence_failures, list) or any(
        not isinstance(item, str) for item in evidence_failures
    ):
        failures.append("failures must be a list of strings")
    elif payload.get("pass") != (not evidence_failures):
        failures.append("pass must be true exactly when failures is empty")
    return failures


def _metrics_paths(output: Path) -> tuple[Path, Path]:
    return (
        output.with_suffix(output.suffix + ".metrics.before.prom"),
        output.with_suffix(output.suffix + ".metrics.after.prom"),
    )


def _post_chat_completion(
    base_url: str,
    request_payload: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    body = json.dumps(request_payload, allow_nan=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, response_body, headers = _request_bytes(
        request, timeout=timeout, opener=opener
    )
    response_text = response_body.decode("utf-8", errors="strict")
    response_json = _strict_json_loads(response_text)
    return {
        "http_status": status,
        "headers": headers,
        "body_sha256": _sha256_bytes(response_body),
        "body_size_bytes": len(response_body),
        "body_text": response_text,
        "json": response_json,
    }


def _post_qa_speed_probe(
    base_url: str,
    prompt: str,
    *,
    timeout: float,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    request_payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": QA_SPEED_PROBE_MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "stream": False,
    }
    body = json.dumps(request_payload, allow_nan=False, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, response_body, headers = _request_bytes(
        request, timeout=timeout, opener=opener
    )
    response_text = response_body.decode("utf-8", errors="strict")
    response_json = _strict_json_loads(response_text)
    return {
        "request": request_payload,
        "http_status": status,
        "headers": headers,
        "body_sha256": _sha256_bytes(response_body),
        "body_size_bytes": len(response_body),
        "body_text": response_text,
        "json": response_json,
    }


def _finish_payload(
    payload: dict[str, Any],
    *,
    kind: str,
    contract: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "contract": contract,
            "failures": failures,
            "pass": not failures,
            "finished_at_utc": _utc_now(),
        }
    )
    envelope_failures = validate_evidence_envelope(
        payload, expected_kind=kind, expected_contract=contract
    )
    if envelope_failures:
        payload["failures"].extend(
            f"internal evidence envelope error: {failure}"
            for failure in envelope_failures
        )
        payload["pass"] = False
    return payload


def qa_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "stream": False,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "expected_content_substring": args.expected_content,
        "expected_finish_reason": args.expected_finish_reason,
        "tp_size": args.tp_size,
        "minimum_graph_delta_per_rank": args.minimum_graph_delta_per_rank,
        "decode_none_positive_delta_allowed": False,
        "missing_decode_none_series_value": 0,
        "decode_speed": {
            "source": "/generate response meta_info.decode_throughput",
            "response_count": 1,
            "finite_and_positive_for_every_response": True,
            "probe_max_new_tokens": QA_SPEED_PROBE_MAX_NEW_TOKENS,
            "probe_ignore_eos": True,
        },
        "inter_token_latency_required": False,
    }


def run_qa(args: argparse.Namespace) -> dict[str, Any]:
    contract = qa_contract(args)
    payload: dict[str, Any] = {"started_at_utc": _utc_now()}
    failures: list[str] = []
    before_path, after_path = _metrics_paths(args.output)
    request_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload["request"] = request_payload

    before = fetch_metrics(
        args.base_url, timeout=args.metrics_timeout, raw_path=before_path
    )
    response = _post_chat_completion(
        args.base_url, request_payload, timeout=args.request_timeout
    )
    speed_probe = _post_qa_speed_probe(
        args.base_url,
        args.prompt,
        timeout=args.request_timeout,
    )
    after = fetch_metrics(
        args.base_url, timeout=args.metrics_timeout, raw_path=after_path
    )
    response_validation = validate_qa_response(
        response["json"],
        expected_content=args.expected_content,
        expected_finish_reason=args.expected_finish_reason,
    )
    speed_validation = validate_qa_speed_probe(speed_probe["json"])
    metrics_comparison = compare_metrics(
        before,
        after,
        tp_size=args.tp_size,
        minimum_graph_delta_per_rank=args.minimum_graph_delta_per_rank,
    )
    failures.extend(f"QA: {item}" for item in response_validation["failures"])
    failures.extend(f"QA decode speed: {item}" for item in speed_validation["failures"])
    failures.extend(f"metrics: {item}" for item in metrics_comparison["failures"])
    payload.update(
        {
            "response": response,
            "response_validation": response_validation,
            "decode_speed_probe": speed_probe,
            "decode_speed_validation": speed_validation,
            "metrics": {
                "before": before,
                "after": after,
                "comparison": metrics_comparison,
            },
        }
    )
    return _finish_payload(payload, kind=QA_KIND, contract=contract, failures=failures)


def _parse_benchmark_endpoint(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("GSM benchmark base URL must be an absolute HTTP URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("base URL must not contain a path, query, or fragment")
    port = parsed.port or 80
    return parsed.hostname, port


def _gsm_decode_speed_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".decode-speed.jsonl")


def build_gsm_command(args: argparse.Namespace) -> list[str]:
    host, port = _parse_benchmark_endpoint(args.base_url)
    return [
        args.python,
        "-c",
        GSM_BENCHMARK_WORKER,
        str(args.benchmark_script),
        str(_gsm_decode_speed_path(args.output)),
        "--host",
        host,
        "--port",
        str(port),
        "--backend",
        "srt",
        "--data-path",
        str(args.data_path),
        "--num-questions",
        str(GSM_QUESTIONS),
        "--num-shots",
        str(GSM_SHOTS),
        "--parallel",
        str(GSM_PARALLEL),
        "--max-new-tokens",
        str(GSM_MAX_NEW_TOKENS),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--result-file",
        str(args.result_file),
        "--raw-result-file",
        str(args.raw_result_file),
    ]


def gsm_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "benchmark": "benchmark/gsm8k/bench_sglang.py",
        "backend": "srt",
        "num_questions": GSM_QUESTIONS,
        "num_shots": GSM_SHOTS,
        "parallel": GSM_PARALLEL,
        "max_new_tokens": GSM_MAX_NEW_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "result_rows": 1,
        "raw_result_rows": GSM_QUESTIONS,
        "raw_prompt_ids": "0..199",
        "all_outputs_non_empty": True,
        "accuracy_is_blocking": False,
        "tp_size": args.tp_size,
        "minimum_graph_delta_per_rank": args.minimum_graph_delta_per_rank,
        "decode_none_positive_delta_allowed": False,
        "missing_decode_none_series_value": 0,
        "decode_speed": {
            "source": "every GSM state meta_info.decode_throughput",
            "response_count": GSM_QUESTIONS,
            "prompt_ids": "0..199",
            "finite_and_positive_for_every_response": True,
        },
        "inter_token_latency_required": False,
        "data_sha256": args.expected_data_sha256,
    }


def run_gsm8k(args: argparse.Namespace) -> dict[str, Any]:
    contract = gsm_contract(args)
    payload: dict[str, Any] = {"started_at_utc": _utc_now()}
    failures: list[str] = []
    before_path, after_path = _metrics_paths(args.output)
    speed_path = _gsm_decode_speed_path(args.output)

    for path, name in (
        (args.data_path, "data path"),
        (args.benchmark_script, "benchmark script"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is not a file: {path}")
    for path in (args.result_file, args.raw_result_file, speed_path):
        if path.exists():
            raise FileExistsError(
                f"refusing append-contaminated GSM artifact; path already exists: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    command = build_gsm_command(args)
    data_sha256 = _sha256_file(args.data_path)
    payload["source"] = {
        "data_path": str(args.data_path.resolve()),
        "data_sha256": data_sha256,
        "benchmark_script": str(args.benchmark_script.resolve()),
        "benchmark_script_sha256": _sha256_file(args.benchmark_script),
        "benchmark_worker_sha256": _sha256_bytes(GSM_BENCHMARK_WORKER.encode("utf-8")),
    }
    if data_sha256 != args.expected_data_sha256:
        raise ValueError(
            "GSM data SHA-256 mismatch: "
            f"expected {args.expected_data_sha256}, got {data_sha256}"
        )
    payload["benchmark"] = {"command": command, "cwd": str(args.work_dir.resolve())}

    before = fetch_metrics(
        args.base_url, timeout=args.metrics_timeout, raw_path=before_path
    )
    completed = subprocess.run(
        command,
        cwd=args.work_dir,
        env=os.environ.copy(),
        capture_output=True,
        text=False,
        check=False,
        timeout=args.benchmark_timeout,
    )
    stdout_path = args.output.with_suffix(args.output.suffix + ".benchmark.stdout.log")
    stderr_path = args.output.with_suffix(args.output.suffix + ".benchmark.stderr.log")
    _atomic_write_bytes(stdout_path, completed.stdout)
    _atomic_write_bytes(stderr_path, completed.stderr)
    payload["benchmark"].update(
        {
            "returncode": completed.returncode,
            "stdout": {
                "path": str(stdout_path.resolve()),
                "sha256": _sha256_bytes(completed.stdout),
                "size_bytes": len(completed.stdout),
            },
            "stderr": {
                "path": str(stderr_path.resolve()),
                "sha256": _sha256_bytes(completed.stderr),
                "size_bytes": len(completed.stderr),
            },
        }
    )
    if completed.returncode != 0:
        failures.append(f"GSM benchmark exited with status {completed.returncode}")

    after = fetch_metrics(
        args.base_url, timeout=args.metrics_timeout, raw_path=after_path
    )
    artifact_validation = validate_gsm_artifacts(args.result_file, args.raw_result_file)
    speed_validation = validate_gsm_decode_speed_artifact(speed_path)
    metrics_comparison = compare_metrics(
        before,
        after,
        tp_size=args.tp_size,
        minimum_graph_delta_per_rank=args.minimum_graph_delta_per_rank,
    )
    failures.extend(f"GSM artifact: {item}" for item in artifact_validation["failures"])
    failures.extend(
        f"GSM decode speed: {item}" for item in speed_validation["failures"]
    )
    failures.extend(f"metrics: {item}" for item in metrics_comparison["failures"])
    payload.update(
        {
            "artifact_validation": artifact_validation,
            "decode_speed_validation": speed_validation,
            "metrics": {
                "before": before,
                "after": after,
                "comparison": metrics_comparison,
            },
        }
    )
    return _finish_payload(payload, kind=GSM_KIND, contract=contract, failures=failures)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def _sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tp-size", type=_positive_int, default=4)
    parser.add_argument(
        "--minimum-graph-delta-per-rank", type=_positive_float, default=1.0
    )
    parser.add_argument("--metrics-timeout", type=_positive_float, default=30.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    qa = subparsers.add_parser("qa", help="Run one natural OpenAI-compatible QA")
    _common_parser(qa)
    qa.add_argument("--model", required=True)
    qa.add_argument(
        "--prompt",
        default=(
            "Answer in one short sentence: What is the chemical formula for water? "
            "Include the formula exactly as plain ASCII."
        ),
    )
    qa.add_argument("--expected-content", default="H2O")
    qa.add_argument("--expected-finish-reason", default="stop")
    qa.add_argument("--max-tokens", type=_positive_int, default=64)
    qa.add_argument("--request-timeout", type=_positive_float, default=300.0)

    gsm = subparsers.add_parser("gsm8k", help="Run graph GSM8K with exactly 200 rows")
    _common_parser(gsm)
    gsm.add_argument("--data-path", required=True, type=Path)
    gsm.add_argument(
        "--expected-data-sha256",
        type=_sha256,
        default=DEFAULT_GSM_DATA_SHA256,
        help="Frozen GSM8K JSONL digest; override only for a separately reviewed fixture",
    )
    gsm.add_argument("--result-file", required=True, type=Path)
    gsm.add_argument("--raw-result-file", required=True, type=Path)
    gsm.add_argument("--work-dir", required=True, type=Path)
    gsm.add_argument("--python", default=sys.executable)
    gsm.set_defaults(
        benchmark_script=(
            Path(__file__).resolve().parents[1] / "benchmark/gsm8k/bench_sglang.py"
        )
    )
    gsm.add_argument("--benchmark-timeout", type=_positive_float, default=7200.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    if args.mode == "gsm8k":
        for name in (
            "data_path",
            "result_file",
            "raw_result_file",
            "work_dir",
            "benchmark_script",
        ):
            setattr(args, name, getattr(args, name).resolve())
        protected_paths = {
            "output": args.output,
            "data": args.data_path,
            "benchmark": args.benchmark_script,
            "result": args.result_file,
            "raw_result": args.raw_result_file,
            "decode_speed": _gsm_decode_speed_path(args.output),
        }
        if len(set(protected_paths.values())) != len(protected_paths):
            collisions = ", ".join(
                f"{name}={path}" for name, path in protected_paths.items()
            )
            raise SystemExit(f"GSM input/output paths must be distinct: {collisions}")
    try:
        payload = run_qa(args) if args.mode == "qa" else run_gsm8k(args)
    except Exception as error:
        kind = QA_KIND if args.mode == "qa" else GSM_KIND
        contract = qa_contract(args) if args.mode == "qa" else gsm_contract(args)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "pass": False,
            "contract": contract,
            "failures": [f"{type(error).__name__}: {error}"],
            "finished_at_utc": _utc_now(),
        }
    _atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "pass": payload["pass"]}))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
