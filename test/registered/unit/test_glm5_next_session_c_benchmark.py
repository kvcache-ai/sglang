import importlib.util
import json
import math
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "glm5_next_session_c_benchmark.py"
SPEC = importlib.util.spec_from_file_location("glm5_next_session_c_benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)


def test_fixture_is_stable_distinct_and_inside_normal_token_band():
    first = BENCHMARK.build_fixture(128, 154880, salt=17)
    repeated = BENCHMARK.build_fixture(128, 154880, salt=17)
    other = BENCHMARK.build_fixture(128, 154880, salt=41)

    assert first == repeated
    assert first["input_sha256"] != other["input_sha256"]
    assert min(first["input_ids"]) >= 1024
    assert max(first["input_ids"]) < 154880 - 1024


def test_long_context_contract_defaults_to_500k_plus_1024():
    args = BENCHMARK.build_parser().parse_args(
        ["long", "--label", "final", "--output", "/tmp/final.json"]
    )

    assert args.input_length == 500_000
    assert args.output_length == 1024


def test_long_context_contract_is_exact_not_a_minimum(monkeypatch, tmp_path):
    monkeypatch.setattr(BENCHMARK, "flush_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        BENCHMARK,
        "run_streaming_request",
        lambda *args, **kwargs: {"success": True},
    )
    output = tmp_path / "long.json"
    args = BENCHMARK.build_parser().parse_args(
        [
            "long",
            "--label",
            "too-long",
            "--output",
            str(output),
            "--input-length",
            "500001",
            "--output-length",
            "1025",
        ]
    )

    assert BENCHMARK.run_long(args) == 1
    payload = json.loads(output.read_text())
    assert not payload["contract"]["compliant"]
    assert payload["contract"]["input_tokens"] == 500_000
    assert payload["contract"]["output_tokens"] == 1024


def test_percentile_matches_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert BENCHMARK.percentile(values, 50) == 3.0
    assert math.isclose(BENCHMARK.percentile(values, 95), 4.8)


def test_server_metrics_are_recovered_from_sglang_meta_definition():
    metrics = BENCHMARK.extract_server_metrics(
        {"e2e_latency": 10.0, "decode_throughput": 20.0},
        completion_tokens=100,
    )

    assert metrics["server_e2e_s"] == 10.0
    assert metrics["server_output_throughput_tps"] == 20.0
    assert metrics["server_decode_duration_s"] == 5.0
    assert metrics["server_ttft_s"] == 5.0
    assert math.isclose(metrics["server_mean_itl_s"], 5.0 / 99.0)
    assert not metrics["server_metric_errors"]


def test_missing_or_inconsistent_server_metrics_remain_null():
    missing = BENCHMARK.extract_server_metrics(
        {"e2e_latency": 10.0}, completion_tokens=100
    )
    assert missing["server_ttft_s"] is None
    assert missing["server_mean_itl_s"] is None
    assert missing["server_output_throughput_tps"] is None
    assert any("--enable-metrics" in error for error in missing["server_metric_errors"])

    inconsistent = BENCHMARK.extract_server_metrics(
        {"e2e_latency": 1.0, "decode_throughput": 1.0},
        completion_tokens=100,
    )
    assert inconsistent["server_ttft_s"] is None
    assert inconsistent["server_mean_itl_s"] is None
    assert any(
        "inconsistent" in error for error in inconsistent["server_metric_errors"]
    )


class _FakeStreamingResponse:
    status = 200

    def __init__(self, messages):
        self._lines = [
            b"data: " + json.dumps(message).encode() + b"\n" for message in messages
        ] + [b"data: [DONE]\n"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)

    def getcode(self):
        return self.status


def test_streaming_request_collects_incremental_ids_and_separates_clocks():
    messages = [
        {
            "output_ids": [7],
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1},
        },
        {
            "output_ids": [8],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
                "e2e_latency": 4.0,
                "decode_throughput": 1.0,
            },
        },
    ]

    def opener(request, timeout):
        assert request.full_url == "http://example.test/generate"
        assert timeout == 30
        return _FakeStreamingResponse(messages)

    timestamps = iter((0.0, 2.0, 3.0, 5.0))
    result = BENCHMARK.run_streaming_request(
        "http://example.test",
        [1, 2, 3],
        2,
        timeout=30,
        opener=opener,
        clock=lambda: next(timestamps),
    )

    assert result["output_ids"] == [7, 8]
    assert result["wall_e2e_s"] == 5.0
    assert result["client_ttft_s"] == 2.0
    assert result["client_mean_itl_s"] == 1.0
    assert result["server_e2e_s"] == 4.0
    assert result["server_ttft_s"] == 2.0
    assert result["server_mean_itl_s"] == 2.0


def _record(value: float, *, digest: str = "same"):
    return {
        "success": True,
        "output_sha256": digest,
        "wall_e2e_s": value * 10,
        "server_e2e_s": value * 9,
        "server_ttft_s": value * 2,
        "server_mean_itl_s": value,
        "server_output_throughput_tps": 1.0 / value,
    }


def test_summary_reports_median_p95_and_fails_closed_on_null():
    records = [_record(value) for value in (1, 2, 3, 4, 5)]
    summary = BENCHMARK.summarize_records(records, expected_count=5)
    assert summary["pass"]
    assert summary["metrics"]["server_mean_itl_s"]["median"] == 3.0
    assert math.isclose(summary["metrics"]["server_mean_itl_s"]["p95"], 4.8)

    records[-1]["server_mean_itl_s"] = None
    incomplete = BENCHMARK.summarize_records(records, expected_count=5)
    assert not incomplete["pass"]
    assert incomplete["metrics"]["server_mean_itl_s"]["median"] is None
    assert any("incomplete" in failure for failure in incomplete["failures"])


def _collection(*, itls, throughputs, passed=True):
    records = []
    for itl, throughput in zip(itls, throughputs):
        record = _record(itl)
        record["server_output_throughput_tps"] = throughput
        records.append(record)
    summary = BENCHMARK.summarize_records(records, expected_count=5)
    shape = {
        "role": "decode",
        "input_tokens": 4096,
        "output_tokens": 256,
        "fixture": {"input_sha256": "fixture"},
        "summary": summary,
    }
    return {
        "schema_version": BENCHMARK.SCHEMA_VERSION,
        "kind": "glm5_next_session_c_benchmark",
        "label": "candidate",
        "contract": {
            "request_concurrency": 1,
            "warmups": BENCHMARK.DEFAULT_WARMUPS,
            "measured": BENCHMARK.DEFAULT_MEASURED,
            "flush_cache_before_each_request": True,
            "server_metrics_required": True,
            "requires_server_enable_metrics": True,
            "metric_derivation": BENCHMARK.SERVER_METRIC_DERIVATION,
            "compliant": True,
        },
        "shapes": [shape],
        "pass": passed and summary["pass"],
    }


def test_graph_comparison_accepts_frozen_relative_gates():
    eager = _collection(itls=[1.0] * 5, throughputs=[100.0] * 5)
    graph = _collection(itls=[0.89] * 5, throughputs=[96.0] * 5)

    result = BENCHMARK.compare_payloads(eager, graph)

    assert result["pass"]
    assert all(gate["pass"] for gate in result["gates"])


def test_graph_comparison_rejects_slow_or_missing_metrics():
    eager = _collection(itls=[1.0] * 5, throughputs=[100.0] * 5)
    slow_graph = _collection(itls=[0.95] * 5, throughputs=[90.0] * 5)
    slow_result = BENCHMARK.compare_payloads(eager, slow_graph)
    assert not slow_result["pass"]
    assert not all(gate["pass"] for gate in slow_result["gates"])

    graph = _collection(itls=[0.89] * 5, throughputs=[96.0] * 5)
    graph["shapes"][0]["summary"]["metrics"]["server_mean_itl_s"]["median"] = None
    missing_result = BENCHMARK.compare_payloads(eager, graph)
    assert not missing_result["pass"]
    assert any(gate["reason"] for gate in missing_result["gates"])


def test_summary_fails_closed_when_any_output_digest_is_missing():
    records = [_record(value) for value in (1, 2, 3, 4, 5)]
    records[-1].pop("output_sha256")

    result = BENCHMARK.summarize_records(records, expected_count=5)

    assert not result["pass"]
    assert not result["output_digest_evidence_complete"]


def test_graph_comparison_rejects_wrong_kind_or_contract():
    eager = _collection(itls=[1.0] * 5, throughputs=[100.0] * 5)
    graph = _collection(itls=[0.89] * 5, throughputs=[96.0] * 5)
    graph["kind"] = "wrong"
    graph["contract"]["compliant"] = False

    result = BENCHMARK.compare_payloads(eager, graph)

    assert not result["pass"]
    assert any("kind" in failure for failure in result["failures"])
    assert any("contract" in failure for failure in result["failures"])
