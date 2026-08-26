#!/usr/bin/env python3
"""HTTP acceptance harness for GLM-5-Next decode CUDA Graphs.

The harness intentionally has no dependency on SGLang internals.  It sends
three deterministic batched requests (A, poison, A) for each frozen graph
bucket, validates eight greedy decode steps and their top log-probabilities,
and stores the complete HTTP and Prometheus evidence in one JSON file.

An eager baseline can be collected with ``--mode eager`` and supplied to the
graph run with ``--eager-json``.  Graph use is accepted only from a real
``/metrics`` counter carrying the ``decode_cuda_graph`` mode.  Server log text
is deliberately not an input to this program and remains separate evidence.

Examples::

    python scripts/glm5_next_session_c_graph_acceptance.py \
      --mode eager --base-url http://127.0.0.1:30100 \
      --output /mnt/.../graph_acceptance/eager.json

    python scripts/glm5_next_session_c_graph_acceptance.py \
      --mode graph --base-url http://127.0.0.1:30101 \
      --eager-json /mnt/.../graph_acceptance/eager.json \
      --output /mnt/.../graph_acceptance/graph.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
FIXTURE_SEED = 20260812
BATCH_SIZES = (1, 2, 4)
PHASES = ("a_before", "poison", "a_after")
PROMPT_TOKENS = 1
OUTPUT_TOKENS = 8
DEFAULT_VOCAB_SIZE = 154880
DEFAULT_TOP_K = 64
TOP_LOGPROB_ATOL = 5e-2
TOP_LOGPROB_RTOL = 5e-2

_METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)"
    r"(?:\s+[^\s]+)?$"
)
_METRIC_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Keep malformed non-finite server values representable in strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixtures(vocab_size: int) -> list[dict[str, Any]]:
    """Build stable one-token A/poison/A batches for graph buckets 1/2/4."""

    if vocab_size < 4096:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    span = vocab_size - 2048
    fixtures: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        a_tokens = [
            1024 + ((FIXTURE_SEED + batch_size * 104729 + row * 7919) % span)
            for row in range(batch_size)
        ]
        poison_tokens = [
            1024 + ((FIXTURE_SEED + 97 + batch_size * 130363 + row * 1543) % span)
            for row in range(batch_size)
        ]
        if any(a == poison for a, poison in zip(a_tokens, poison_tokens)):
            raise RuntimeError(f"A and poison fixtures overlap for bs={batch_size}")
        fixtures.append(
            {
                "batch_size": batch_size,
                "prompt_length": PROMPT_TOKENS,
                "a_tokens": a_tokens,
                "poison_tokens": poison_tokens,
                "a_sha256": _sha256_json(a_tokens),
                "poison_sha256": _sha256_json(poison_tokens),
            }
        )
    return fixtures


def _request_bytes(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> bytes:
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:4096]
        raise RuntimeError(
            f"HTTP {error.code} from {request.full_url}: {detail!r}"
        ) from error
    if status < 200 or status >= 300:
        raise RuntimeError(f"HTTP {status} from {request.full_url}: {payload[:4096]!r}")
    return payload


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    payload = _request_bytes(request, timeout=timeout, opener=opener)
    return json.loads(payload)


def _get_text(
    url: str,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> str:
    request = urllib.request.Request(url, method="GET")
    payload = _request_bytes(request, timeout=timeout, opener=opener)
    return payload.decode("utf-8")


def _parse_metric_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    labels: dict[str, str] = {}
    consumed = []
    for match in _METRIC_LABEL.finditer(raw):
        key, encoded_value = match.groups()
        try:
            value = json.loads(f'"{encoded_value}"')
        except json.JSONDecodeError:
            value = encoded_value
        labels[key] = value
        consumed.append(match.span())

    # Reject partially parsed label blocks instead of accidentally matching a
    # malformed line.  Only commas and whitespace may remain between matches.
    cursor = 0
    for start, end in consumed:
        if raw[cursor:start].strip(" ,\t"):
            return {}
        cursor = end
    if raw[cursor:].strip(" ,\t"):
        return {}
    return labels


def parse_prometheus_metrics(text: str) -> list[dict[str, Any]]:
    """Parse numeric Prometheus samples while retaining their source lines."""

    samples: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.fullmatch(line)
        if match is None:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples.append(
            {
                "name": name,
                "labels": _parse_metric_labels(raw_labels),
                "value": value,
                "finite": math.isfinite(value),
                "line": line,
            }
        )
    return samples


def _is_decode_cuda_graph_counter(sample: dict[str, Any]) -> bool:
    name = str(sample.get("name", "")).lower().replace(":", "_")
    labels = {
        str(key).lower(): str(value).lower()
        for key, value in sample.get("labels", {}).items()
    }
    counter_like = name.endswith("_total") or name.endswith("_count")
    labeled_mode = any(
        value == "decode_cuda_graph"
        for key, value in labels.items()
        if key in {"mode", "forward_mode", "graph_mode", "cuda_graph_mode"}
    )
    labeled_counter = labeled_mode and "cuda_graph" in name and counter_like
    dedicated_counter = "decode_cuda_graph" in name and counter_like
    return bool(sample.get("finite") and (labeled_counter or dedicated_counter))


def decode_cuda_graph_counter(text: str) -> dict[str, Any]:
    matched = [
        sample
        for sample in parse_prometheus_metrics(text)
        if _is_decode_cuda_graph_counter(sample)
    ]
    return {
        "available": bool(matched),
        "total": sum(float(sample["value"]) for sample in matched) if matched else None,
        "samples": matched,
    }


def cuda_graph_counters_by_rank(text: str) -> dict[str, dict[str, float]]:
    """Return the production CUDA-graph counter split by mode and TP rank."""

    counters = {"decode_cuda_graph": {}, "decode_none": {}}
    for sample in parse_prometheus_metrics(text):
        if sample.get("name") != "sglang:cuda_graph_passes_total":
            continue
        labels = sample.get("labels", {})
        mode = labels.get("mode")
        rank = labels.get("tp_rank", labels.get("rank"))
        if mode not in counters or rank is None or not sample.get("finite"):
            continue
        if rank in counters[mode]:
            raise ValueError(f"duplicate {mode} counter for TP rank {rank}")
        counters[mode][str(rank)] = float(sample["value"])
    return counters


def compare_metric_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tp_size: int = 1,
    minimum_graph_delta_per_rank: float = 1.0,
) -> dict[str, Any]:
    before_counter = before.get("decode_cuda_graph_counter", {})
    after_counter = after.get("decode_cuda_graph_counter", {})
    before_total = before_counter.get("total")
    after_total = after_counter.get("total")
    failures: list[str] = []
    delta = None
    expected_ranks = {str(rank) for rank in range(tp_size)}
    before_by_rank = before.get("cuda_graph_counters_by_rank", {})
    after_by_rank = after.get("cuda_graph_counters_by_rank", {})
    per_rank: dict[str, Any] = {}

    if not before.get("success"):
        failures.append("the pre-run /metrics snapshot failed")
    if not after.get("success"):
        failures.append("the post-run /metrics snapshot failed")
    if not before_counter.get("available"):
        failures.append("no decode_cuda_graph counter was exposed before the run")
    if not after_counter.get("available"):
        failures.append("no decode_cuda_graph counter was exposed after the run")
    if before_total is not None and after_total is not None:
        delta = float(after_total) - float(before_total)
    for mode in ("decode_cuda_graph", "decode_none"):
        before_ranks = set(before_by_rank.get(mode, {}))
        after_ranks = set(after_by_rank.get(mode, {}))
        if before_ranks != expected_ranks or after_ranks != expected_ranks:
            failures.append(
                f"{mode} TP rank set mismatch: expected={sorted(expected_ranks)}, "
                f"before={sorted(before_ranks)}, after={sorted(after_ranks)}"
            )
    for rank in sorted(expected_ranks, key=int):
        graph_before = before_by_rank.get("decode_cuda_graph", {}).get(rank)
        graph_after = after_by_rank.get("decode_cuda_graph", {}).get(rank)
        none_before = before_by_rank.get("decode_none", {}).get(rank)
        none_after = after_by_rank.get("decode_none", {}).get(rank)
        rank_failures: list[str] = []
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
        if graph_delta is not None and graph_delta < minimum_graph_delta_per_rank:
            rank_failures.append(
                f"graph delta {graph_delta} is below {minimum_graph_delta_per_rank}"
            )
        if none_delta is not None and none_delta != 0:
            rank_failures.append(f"decode_none delta is {none_delta}, expected 0")
        failures.extend(f"TP rank {rank}: {failure}" for failure in rank_failures)
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
        "source": "HTTP /metrics only",
        "log_evidence_used": False,
        "before_total": before_total,
        "after_total": after_total,
        "delta": delta,
        "expected_tp_ranks": sorted(expected_ranks, key=int),
        "minimum_graph_delta_per_rank": minimum_graph_delta_per_rank,
        "expected_decode_none_delta_per_rank": 0,
        "per_rank": per_rank,
        "failures": failures,
        "pass": not failures,
    }


def _metrics_snapshot(
    base_url: str,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        text = _get_text(url, timeout=timeout, opener=opener)
        return {
            "success": True,
            "url": url,
            "retrieved_unix": time.time(),
            "raw_text": text,
            "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "decode_cuda_graph_counter": decode_cuda_graph_counter(text),
            "cuda_graph_counters_by_rank": cuda_graph_counters_by_rank(text),
        }
    except Exception as error:
        return {
            "success": False,
            "url": url,
            "retrieved_unix": time.time(),
            "error": f"{type(error).__name__}: {error}",
            "raw_text": None,
            "raw_sha256": None,
            "decode_cuda_graph_counter": {
                "available": False,
                "total": None,
                "samples": [],
            },
            "cuda_graph_counters_by_rank": {
                "decode_cuda_graph": {},
                "decode_none": {},
            },
        }


def _parse_top_logprobs(raw: Any) -> list[dict[str, float | int]]:
    parsed: list[dict[str, float | int]] = []
    seen: set[int] = set()
    if not isinstance(raw, list):
        raise TypeError(f"top logprobs must be a list, got {type(raw).__name__}")
    for entry in raw:
        if isinstance(entry, dict):
            token_id = entry.get("token_id")
            logprob = entry.get("logprob")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            logprob, token_id = entry[0], entry[1]
        else:
            raise TypeError(f"invalid top-logprob entry: {entry!r}")
        if isinstance(token_id, bool) or not isinstance(token_id, (int, float)):
            raise TypeError(f"invalid top-logprob token id: {token_id!r}")
        if not math.isfinite(float(token_id)) or int(token_id) != token_id:
            raise ValueError(f"invalid top-logprob token id: {token_id!r}")
        token_id = int(token_id)
        if token_id in seen:
            raise ValueError(f"duplicate top-logprob token id: {token_id}")
        seen.add(token_id)
        try:
            logprob = float(logprob)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"invalid logprob for token {token_id}: {logprob!r}"
            ) from error
        if not math.isfinite(logprob):
            raise ValueError(f"non-finite logprob for token {token_id}: {logprob}")
        parsed.append({"token_id": token_id, "logprob": logprob})
    return parsed


def _normalize_generate_response(
    raw_response: Any,
    *,
    prompt_tokens: list[int],
    output_tokens: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    batch_size = len(prompt_tokens)
    failures: list[str] = []
    if isinstance(raw_response, dict) and batch_size == 1:
        responses = [raw_response]
    elif isinstance(raw_response, list):
        responses = raw_response
    else:
        return [], [
            f"expected a list of {batch_size} responses, got "
            f"{type(raw_response).__name__}"
        ]
    if len(responses) != batch_size:
        failures.append(
            f"response batch size mismatch: expected {batch_size}, got {len(responses)}"
        )

    sequences: list[dict[str, Any]] = []
    for sequence_index, (prompt_token, response) in enumerate(
        zip(prompt_tokens, responses)
    ):
        sequence_failures: list[str] = []
        if not isinstance(response, dict):
            failures.append(
                f"sequence {sequence_index}: response is {type(response).__name__}"
            )
            continue
        raw_ids = response.get("output_ids")
        if not isinstance(raw_ids, list):
            generated_ids: list[int] = []
            sequence_failures.append("output_ids is not a list")
        else:
            generated_ids = []
            for token_id in raw_ids:
                if isinstance(token_id, bool) or not isinstance(token_id, (int, float)):
                    sequence_failures.append(f"invalid output token id: {token_id!r}")
                    continue
                if not math.isfinite(float(token_id)) or int(token_id) != token_id:
                    sequence_failures.append(f"invalid output token id: {token_id!r}")
                    continue
                generated_ids.append(int(token_id))
        if len(generated_ids) != output_tokens:
            sequence_failures.append(
                f"expected {output_tokens} output ids, got {len(generated_ids)}"
            )

        meta = response.get("meta_info")
        if not isinstance(meta, dict):
            meta = {}
            sequence_failures.append("meta_info is not an object")
        completion_tokens = meta.get("completion_tokens")
        if completion_tokens is not None and completion_tokens != output_tokens:
            sequence_failures.append(
                f"meta_info.completion_tokens={completion_tokens}, expected {output_tokens}"
            )
        raw_steps = meta.get("output_top_logprobs")
        if not isinstance(raw_steps, list):
            raw_steps = []
            sequence_failures.append("meta_info.output_top_logprobs is not a list")
        if len(raw_steps) != output_tokens:
            sequence_failures.append(
                f"expected {output_tokens} top-logprob steps, got {len(raw_steps)}"
            )

        steps: list[dict[str, Any]] = []
        for step_index, raw_top in enumerate(raw_steps[:output_tokens]):
            try:
                top_logprobs = _parse_top_logprobs(raw_top)
                if len(top_logprobs) < top_k:
                    raise ValueError(
                        f"expected at least {top_k} top logprobs, got {len(top_logprobs)}"
                    )
                top_logprobs = top_logprobs[:top_k]
            except Exception as error:
                sequence_failures.append(
                    f"step {step_index}: {type(error).__name__}: {error}"
                )
                top_logprobs = []
            steps.append(
                {
                    "step": step_index,
                    "token_id": (
                        generated_ids[step_index]
                        if step_index < len(generated_ids)
                        else None
                    ),
                    "top_logprobs": top_logprobs,
                }
            )

        if sequence_failures:
            failures.extend(
                f"sequence {sequence_index}: {failure}" for failure in sequence_failures
            )
        sequences.append(
            {
                "sequence_index": sequence_index,
                "input_ids": [prompt_token],
                "generated_ids": generated_ids,
                "steps": steps,
                "failures": sequence_failures,
                "pass": not sequence_failures,
            }
        )
    return sequences, failures


def _run_phase(
    base_url: str,
    *,
    batch_size: int,
    phase: str,
    prompt_tokens: list[int],
    top_k: int,
    timeout: float,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    request_body = {
        "input_ids": [[token_id] for token_id in prompt_tokens],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "top_logprobs_num": top_k,
        "return_text_in_logprobs": False,
    }
    started = time.perf_counter()
    raw_response: Any = None
    try:
        raw_response = _post_json(
            f"{base_url.rstrip('/')}/generate",
            request_body,
            timeout=timeout,
            opener=opener,
        )
    except Exception as error:
        return {
            "phase": phase,
            "batch_size": batch_size,
            "request": request_body,
            "raw_response": None,
            "sequences": [],
            "elapsed_seconds": time.perf_counter() - started,
            "failures": [f"HTTP request failed: {type(error).__name__}: {error}"],
            "pass": False,
        }

    try:
        sequences, failures = _normalize_generate_response(
            raw_response,
            prompt_tokens=prompt_tokens,
            output_tokens=OUTPUT_TOKENS,
            top_k=top_k,
        )
        return {
            "phase": phase,
            "batch_size": batch_size,
            "request": request_body,
            "raw_response": _json_safe(raw_response),
            "sequences": sequences,
            "elapsed_seconds": time.perf_counter() - started,
            "failures": failures,
            "pass": not failures,
        }
    except Exception as error:
        return {
            "phase": phase,
            "batch_size": batch_size,
            "request": request_body,
            "raw_response": _json_safe(raw_response),
            "sequences": [],
            "elapsed_seconds": time.perf_counter() - started,
            "failures": [
                f"response validation failed: {type(error).__name__}: {error}"
            ],
            "pass": False,
        }


def _compare_phase_records(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    context: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    failures: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    expected_sequences = expected.get("sequences", [])
    actual_sequences = actual.get("sequences", [])
    if len(expected_sequences) != len(actual_sequences):
        failures.append(
            f"{context}: sequence count mismatch: "
            f"{len(expected_sequences)} != {len(actual_sequences)}"
        )

    for sequence_index, (expected_sequence, actual_sequence) in enumerate(
        zip(expected_sequences, actual_sequences)
    ):
        prefix = f"{context} sequence {sequence_index}"
        if expected_sequence.get("input_ids") != actual_sequence.get("input_ids"):
            failures.append(f"{prefix}: input ids mismatch")
        expected_ids = expected_sequence.get("generated_ids", [])
        actual_ids = actual_sequence.get("generated_ids", [])
        if expected_ids != actual_ids:
            failures.append(
                f"{prefix}: generated token mismatch: {expected_ids} != {actual_ids}"
            )
        first_divergent_step = next(
            (
                index
                for index, (expected_id, actual_id) in enumerate(
                    zip(expected_ids, actual_ids)
                )
                if expected_id != actual_id
            ),
            None,
        )
        if first_divergent_step is None and len(expected_ids) != len(actual_ids):
            first_divergent_step = min(len(expected_ids), len(actual_ids))

        expected_steps = expected_sequence.get("steps", [])
        actual_steps = actual_sequence.get("steps", [])
        if len(expected_steps) != len(actual_steps):
            failures.append(
                f"{prefix}: step count mismatch: "
                f"{len(expected_steps)} != {len(actual_steps)}"
            )
        for step_index, (expected_step, actual_step) in enumerate(
            zip(expected_steps, actual_steps)
        ):
            if first_divergent_step is not None and step_index > first_divergent_step:
                diagnostics.append(
                    {
                        "context": context,
                        "sequence": sequence_index,
                        "step": step_index,
                        "status": "skipped_due_to_prefix_divergence",
                    }
                )
                continue
            expected_top = expected_step.get("top_logprobs", [])
            actual_by_id = {
                int(item["token_id"]): float(item["logprob"])
                for item in actual_step.get("top_logprobs", [])
            }
            missing_ids: list[int] = []
            outside_tolerance: list[dict[str, Any]] = []
            max_abs = 0.0
            max_relative = 0.0
            for item in expected_top:
                token_id = int(item["token_id"])
                if token_id not in actual_by_id:
                    missing_ids.append(token_id)
                    continue
                expected_value = float(item["logprob"])
                actual_value = actual_by_id[token_id]
                delta = abs(expected_value - actual_value)
                relative = delta / max(abs(expected_value), 1e-6)
                max_abs = max(max_abs, delta)
                max_relative = max(max_relative, relative)
                if delta > atol + rtol * abs(expected_value):
                    outside_tolerance.append(
                        {
                            "token_id": token_id,
                            "expected": expected_value,
                            "actual": actual_value,
                            "absolute_delta": delta,
                            "relative_delta": relative,
                        }
                    )
            diagnostics.append(
                {
                    "context": context,
                    "sequence": sequence_index,
                    "step": step_index,
                    "status": "compared",
                    "max_abs": max_abs,
                    "max_relative": max_relative,
                    "missing_reference_top_ids": missing_ids,
                    "outside_tolerance": outside_tolerance,
                }
            )
            if missing_ids:
                failures.append(
                    f"{prefix} step {step_index}: {len(missing_ids)} reference "
                    "top-logprob token ids are missing"
                )
            if outside_tolerance:
                failures.append(
                    f"{prefix} step {step_index}: {len(outside_tolerance)} "
                    f"logprobs outside atol={atol}, rtol={rtol}"
                )

    return {
        "context": context,
        "atol": atol,
        "rtol": rtol,
        "failures": failures,
        "diagnostics": diagnostics,
        "pass": not failures,
    }


def _poison_was_observed(
    a_phase: dict[str, Any], poison_phase: dict[str, Any]
) -> dict[str, Any]:
    for sequence_index, (a_sequence, poison_sequence) in enumerate(
        zip(a_phase.get("sequences", []), poison_phase.get("sequences", []))
    ):
        if a_sequence.get("generated_ids") != poison_sequence.get("generated_ids"):
            return {
                "observed": True,
                "sequence": sequence_index,
                "reason": "generated_ids_changed",
            }
        for step_index, (a_step, poison_step) in enumerate(
            zip(a_sequence.get("steps", []), poison_sequence.get("steps", []))
        ):
            a_values = {
                int(item["token_id"]): float(item["logprob"])
                for item in a_step.get("top_logprobs", [])
            }
            poison_values = {
                int(item["token_id"]): float(item["logprob"])
                for item in poison_step.get("top_logprobs", [])
            }
            if set(a_values) != set(poison_values):
                return {
                    "observed": True,
                    "sequence": sequence_index,
                    "step": step_index,
                    "reason": "top_logprob_token_ids_changed",
                }
            if any(
                not math.isclose(
                    a_values[token_id],
                    poison_values[token_id],
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                )
                for token_id in a_values
            ):
                return {
                    "observed": True,
                    "sequence": sequence_index,
                    "step": step_index,
                    "reason": "top_logprob_values_changed",
                }
    return {
        "observed": False,
        "reason": "poison request reproduced all A outputs and top logprobs",
    }


def _evaluate_a_poison_a(case: dict[str, Any]) -> dict[str, Any]:
    by_phase = {phase["phase"]: phase for phase in case.get("phases", [])}
    failures: list[str] = []
    if set(by_phase) != set(PHASES):
        failures.append(
            f"phase set mismatch: expected {list(PHASES)}, got {sorted(by_phase)}"
        )
        return {"pass": False, "failures": failures}
    for phase_name, phase in by_phase.items():
        if not phase.get("pass"):
            failures.extend(
                f"{phase_name}: {failure}" for failure in phase.get("failures", [])
            )

    repeat = _compare_phase_records(
        by_phase["a_before"],
        by_phase["a_after"],
        context=f"bs={case['batch_size']} A repeat",
        atol=TOP_LOGPROB_ATOL,
        rtol=TOP_LOGPROB_RTOL,
    )
    if not repeat["pass"]:
        failures.extend(repeat["failures"])
    poison = _poison_was_observed(by_phase["a_before"], by_phase["poison"])
    if not poison["observed"]:
        failures.append(str(poison["reason"]))
    return {
        "a_repeat": repeat,
        "poison": poison,
        "failures": failures,
        "pass": not failures,
    }


def _case_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for case in payload.get("cases", []):
        batch_size = int(case["batch_size"])
        if batch_size in result:
            raise ValueError(f"duplicate batch size in payload: {batch_size}")
        result[batch_size] = case
    return result


def compare_acceptance_payloads(
    eager: dict[str, Any],
    graph: dict[str, Any],
    *,
    atol: float = TOP_LOGPROB_ATOL,
    rtol: float = TOP_LOGPROB_RTOL,
) -> dict[str, Any]:
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    if eager.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported eager schema")
    if graph.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported graph schema")
    if eager.get("mode") != "eager":
        failures.append("reference payload is not an eager run")
    if graph.get("mode") != "graph":
        failures.append("candidate payload is not a graph run")
    if eager.get("contract") != graph.get("contract"):
        failures.append("eager and graph contracts differ")
    if not eager.get("collection_pass", eager.get("pass", False)):
        failures.append("eager collection did not pass")
    if not graph.get("collection_pass", False):
        failures.append("graph collection did not pass")

    try:
        eager_cases = _case_map(eager)
        graph_cases = _case_map(graph)
    except Exception as error:
        failures.append(f"invalid case map: {type(error).__name__}: {error}")
        eager_cases, graph_cases = {}, {}
    if set(eager_cases) != set(BATCH_SIZES) or set(graph_cases) != set(BATCH_SIZES):
        failures.append(
            f"case set mismatch: eager={sorted(eager_cases)}, "
            f"graph={sorted(graph_cases)}"
        )

    for batch_size in sorted(set(eager_cases) & set(graph_cases)):
        eager_case = eager_cases[batch_size]
        graph_case = graph_cases[batch_size]
        if eager_case.get("fixture") != graph_case.get("fixture"):
            failures.append(f"bs={batch_size}: fixture mismatch")
            continue
        eager_phases = {phase["phase"]: phase for phase in eager_case.get("phases", [])}
        graph_phases = {phase["phase"]: phase for phase in graph_case.get("phases", [])}
        if set(eager_phases) != set(graph_phases):
            failures.append(f"bs={batch_size}: phase set mismatch")
        for phase in PHASES:
            if phase not in eager_phases or phase not in graph_phases:
                continue
            result = _compare_phase_records(
                eager_phases[phase],
                graph_phases[phase],
                context=f"bs={batch_size} phase={phase} eager-vs-graph",
                atol=atol,
                rtol=rtol,
            )
            comparisons.append(result)
            failures.extend(result["failures"])

    return {
        "kind": "glm5_next_session_c_eager_graph_comparison",
        "atol": atol,
        "rtol": rtol,
        "tokens_exact_required": True,
        "comparisons": comparisons,
        "failures": failures,
        "pass": not failures,
    }


def _flush_cache(
    base_url: str,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/flush_cache"
    try:
        request = urllib.request.Request(
            url,
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = _request_bytes(request, timeout=timeout, opener=opener)
        return {
            "success": True,
            "url": url,
            "response_text": response.decode("utf-8", errors="replace"),
        }
    except Exception as error:
        return {
            "success": False,
            "url": url,
            "error": f"{type(error).__name__}: {error}",
        }


def collect_acceptance(
    args: argparse.Namespace,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    if args.mode == "graph" and not args.eager_json:
        raise ValueError("--eager-json is required in graph mode")
    started = time.time()
    base_url = args.base_url.rstrip("/")
    fixtures = build_fixtures(args.vocab_size)
    failures: list[str] = []

    metrics_before = _metrics_snapshot(
        base_url, timeout=args.request_timeout, opener=opener
    )
    flush = (
        {"success": True, "skipped": True}
        if args.no_flush_cache
        else _flush_cache(base_url, timeout=args.request_timeout, opener=opener)
    )
    if not flush.get("success"):
        failures.append(f"flush_cache failed: {flush.get('error')}")

    cases: list[dict[str, Any]] = []
    for fixture in fixtures:
        batch_size = int(fixture["batch_size"])
        phase_inputs = {
            "a_before": fixture["a_tokens"],
            "poison": fixture["poison_tokens"],
            "a_after": fixture["a_tokens"],
        }
        phases = [
            _run_phase(
                base_url,
                batch_size=batch_size,
                phase=phase,
                prompt_tokens=phase_inputs[phase],
                top_k=args.top_k,
                timeout=args.request_timeout,
                opener=opener,
            )
            for phase in PHASES
        ]
        case = {"batch_size": batch_size, "fixture": fixture, "phases": phases}
        case["a_poison_a"] = _evaluate_a_poison_a(case)
        case["pass"] = bool(case["a_poison_a"]["pass"])
        if not case["pass"]:
            failures.extend(
                f"bs={batch_size}: {failure}"
                for failure in case["a_poison_a"]["failures"]
            )
        cases.append(case)

    metrics_after = _metrics_snapshot(
        base_url, timeout=args.request_timeout, opener=opener
    )
    graph_metric_gate = compare_metric_snapshots(
        metrics_before,
        metrics_after,
        tp_size=args.tp_size,
        minimum_graph_delta_per_rank=len(BATCH_SIZES)
        * len(PHASES)
        * (OUTPUT_TOKENS - 1),
    )
    graph_metric_gate["required"] = args.mode == "graph"
    if args.mode == "graph" and not graph_metric_gate["pass"]:
        failures.extend(graph_metric_gate["failures"])

    contract = {
        "batch_sizes": list(BATCH_SIZES),
        "request_concurrency": "one native HTTP batch at a time",
        "prompt_tokens_per_request": PROMPT_TOKENS,
        "output_tokens_per_request": OUTPUT_TOKENS,
        "phases": list(PHASES),
        "temperature": 0,
        "ignore_eos": True,
        "top_k": args.top_k,
        "top_logprob_atol": TOP_LOGPROB_ATOL,
        "top_logprob_rtol": TOP_LOGPROB_RTOL,
        "requires_http_metrics_for_graph": True,
        "metrics_evidence_version": 2,
        "tp_size": args.tp_size,
        "server_log_evidence_is_separate": True,
    }
    collection_pass = all(case.get("pass") for case in cases) and flush.get("success")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_session_c_graph_acceptance",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "label": args.label,
        "mode": args.mode,
        "base_url": base_url,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "fixture_seed": FIXTURE_SEED,
        "vocab_size": args.vocab_size,
        "contract": contract,
        "flush_cache": flush,
        "cases": cases,
        "metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "decode_cuda_graph_gate": graph_metric_gate,
        },
        "collection_pass": collection_pass,
        "comparison": None,
        "failures": failures,
    }

    if args.eager_json is not None:
        eager_path = Path(args.eager_json)
        try:
            eager = json.loads(eager_path.read_text(encoding="utf-8"))
            comparison = compare_acceptance_payloads(eager, payload)
            comparison["eager_json"] = str(eager_path.resolve())
            comparison["eager_sha256"] = _sha256_file(eager_path)
            payload["comparison"] = comparison
            if not comparison["pass"]:
                failures.extend(comparison["failures"])
        except Exception as error:
            message = f"eager comparison failed: {type(error).__name__}: {error}"
            payload["comparison"] = {"pass": False, "failures": [message]}
            failures.append(message)

    payload["pass"] = bool(
        collection_pass
        and (args.mode != "graph" or graph_metric_gate["pass"])
        and (payload["comparison"] is None or payload["comparison"]["pass"])
        and not failures
    )
    return payload


def run(
    args: argparse.Namespace,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> int:
    try:
        payload = collect_acceptance(args, opener=opener)
    except Exception as error:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "kind": "glm5_next_session_c_graph_acceptance",
            "created_unix": time.time(),
            "label": getattr(args, "label", None),
            "mode": getattr(args, "mode", None),
            "base_url": getattr(args, "base_url", None),
            "collection_pass": False,
            "comparison": None,
            "failures": [f"{type(error).__name__}: {error}"],
            "pass": False,
        }
    output = Path(args.output)
    _atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "pass": payload["pass"],
                "collection_pass": payload.get("collection_pass"),
                "graph_metrics_pass": payload.get("metrics", {})
                .get("decode_cuda_graph_gate", {})
                .get("pass"),
                "comparison_pass": (
                    payload.get("comparison", {}).get("pass")
                    if payload.get("comparison") is not None
                    else None
                ),
                "failures": payload.get("failures", []),
            },
            indent=2,
        )
    )
    return 0 if payload["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="graph")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--eager-json")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--no-flush-cache", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.mode == "eager" and args.eager_json is not None:
        raise ValueError("--eager-json is only valid in graph mode")
    if args.mode == "graph" and args.eager_json is None:
        raise ValueError("--eager-json is required in graph mode")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
