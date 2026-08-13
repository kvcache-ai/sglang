from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/glm5_next_session_c_qa_gsm_graph_acceptance.py"
SPEC = importlib.util.spec_from_file_location("qa_gsm_graph_acceptance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def _metrics(
    *,
    graph: list[float],
    decode_none: list[float] | None = None,
    itl_sum: float | None = None,
    itl_count: float | None = None,
) -> str:
    lines = ["# HELP sglang:cuda_graph_passes_total graph forward passes"]
    for rank, value in enumerate(graph):
        lines.append(
            "sglang:cuda_graph_passes_total"
            f'{{mode="decode_cuda_graph",tp_rank="{rank}"}} {value}'
        )
    if decode_none is not None:
        for rank, value in enumerate(decode_none):
            lines.append(
                "sglang:cuda_graph_passes_total"
                f'{{mode="decode_none",tp_rank="{rank}"}} {value}'
            )
    if itl_sum is not None:
        lines.append(
            f'sglang:inter_token_latency_seconds_sum{{model_name="glm"}} {itl_sum}'
        )
    if itl_count is not None:
        lines.append(
            f'sglang:inter_token_latency_seconds_count{{model_name="glm"}} {itl_count}'
        )
    return "\n".join(lines) + "\n"


def _qa_payload(*, content: str = "Water is H2O.", finish_reason: str = "stop"):
    return {
        "id": "chatcmpl-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _write_gsm_artifacts(
    result_path: Path,
    raw_path: Path,
    *,
    correct_count: int,
) -> None:
    raw_rows = [
        {
            "prompt_id": index,
            "prompt": f"Question {index}",
            "output": f"Answer {index}",
            "correct": index < correct_count,
        }
        for index in range(acceptance.GSM_QUESTIONS)
    ]
    summary = {
        "task": "gsm8k",
        "backend": "srt",
        "num_gpus": 1,
        "latency": 12.5,
        "accuracy": round(correct_count / acceptance.GSM_QUESTIONS, 3),
        "num_requests": acceptance.GSM_QUESTIONS,
        "other": {
            "num_questions": acceptance.GSM_QUESTIONS,
            "parallel": acceptance.GSM_PARALLEL,
        },
    }
    result_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    raw_path.write_text(
        "\n".join(json.dumps(row) for row in raw_rows), encoding="utf-8"
    )


def _write_gsm_speed_artifact(path: Path, *, throughput: float = 12.5) -> None:
    rows = [
        {
            "prompt_id": index,
            "meta_info": {
                "completion_tokens": 8,
                "decode_throughput": throughput + index,
            },
        }
        for index in range(acceptance.GSM_QUESTIONS)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_metrics_require_graph_on_every_rank_and_keep_itl_diagnostic():
    before = acceptance.analyze_metrics(
        _metrics(graph=[10, 11], decode_none=[3, 4], itl_sum=2.0, itl_count=20)
    )
    after = acceptance.analyze_metrics(
        _metrics(graph=[13, 15], decode_none=[3, 4], itl_sum=4.0, itl_count=60)
    )

    comparison = acceptance.compare_metrics(
        before, after, tp_size=2, minimum_graph_delta_per_rank=3
    )

    assert comparison["pass"] is True
    assert comparison["per_rank"]["0"]["decode_cuda_graph_delta"] == 3
    assert comparison["per_rank"]["1"]["decode_cuda_graph_delta"] == 4
    assert comparison["per_rank"]["0"]["decode_none_delta"] == 0
    assert comparison["inter_token_latency"]["decode_tokens_per_second"] == 20.0


@pytest.mark.parametrize(
    "after_text, expected_failure",
    [
        (
            _metrics(graph=[12], decode_none=[3], itl_sum=3.0, itl_count=30),
            "TP rank set mismatch",
        ),
        (
            _metrics(graph=[12, 13], decode_none=[4, 4], itl_sum=3.0, itl_count=30),
            "decode_none delta must not be positive",
        ),
    ],
)
def test_metrics_fail_closed_on_missing_graph_rank_or_positive_decode_none(
    after_text: str, expected_failure: str
):
    before = acceptance.analyze_metrics(
        _metrics(graph=[10, 11], decode_none=[3, 4], itl_sum=2.0, itl_count=20)
    )
    after = acceptance.analyze_metrics(after_text)

    comparison = acceptance.compare_metrics(
        before, after, tp_size=2, minimum_graph_delta_per_rank=1
    )

    assert comparison["pass"] is False
    assert any(expected_failure in failure for failure in comparison["failures"])


def test_metrics_treat_missing_decode_none_as_zero_and_itl_as_optional():
    before = acceptance.analyze_metrics(_metrics(graph=[10, 11]))
    after = acceptance.analyze_metrics(_metrics(graph=[12, 14]))

    comparison = acceptance.compare_metrics(
        before, after, tp_size=2, minimum_graph_delta_per_rank=1
    )

    assert comparison["pass"] is True
    assert comparison["per_rank"]["0"]["decode_none_before"] == 0
    assert comparison["per_rank"]["0"]["decode_none_after"] == 0
    assert comparison["per_rank"]["1"]["decode_none_delta"] == 0
    assert comparison["inter_token_latency"]["available"] is False
    assert comparison["inter_token_latency"]["blocking"] is False
    assert comparison["inter_token_latency"]["diagnostics"]


def test_metrics_allow_decode_none_counter_disappearance_but_not_increase():
    before = acceptance.analyze_metrics(_metrics(graph=[10, 11], decode_none=[3, 4]))
    after = acceptance.analyze_metrics(_metrics(graph=[12, 13]))

    comparison = acceptance.compare_metrics(
        before, after, tp_size=2, minimum_graph_delta_per_rank=1
    )

    assert comparison["pass"] is True
    assert comparison["per_rank"]["0"]["decode_none_delta"] == -3
    assert comparison["per_rank"]["1"]["decode_none_delta"] == -4


def test_metrics_require_positive_cuda_graph_delta_on_all_eight_ranks():
    before = acceptance.analyze_metrics(_metrics(graph=[10] * 8))
    after = acceptance.analyze_metrics(_metrics(graph=[11] * 7 + [10]))

    comparison = acceptance.compare_metrics(
        before, after, tp_size=8, minimum_graph_delta_per_rank=1
    )

    assert comparison["pass"] is False
    assert comparison["per_rank"]["7"]["pass"] is False
    assert any("TP rank 7" in failure for failure in comparison["failures"])


def test_qa_requires_choice_content_usage_and_stop_finish():
    valid = acceptance.validate_qa_response(
        _qa_payload(), expected_content="H2O", expected_finish_reason="stop"
    )
    invalid_payload = _qa_payload(content="Water.", finish_reason="length")
    invalid_payload["usage"]["total_tokens"] = 99
    invalid = acceptance.validate_qa_response(
        invalid_payload, expected_content="H2O", expected_finish_reason="stop"
    )

    assert valid["pass"] is True
    assert invalid["pass"] is False
    assert any("does not contain 'H2O'" in failure for failure in invalid["failures"])
    assert any("finish_reason" in failure for failure in invalid["failures"])
    assert any("total_tokens" in failure for failure in invalid["failures"])


def test_qa_decode_speed_probe_requires_exact_field_and_positive_finite_value():
    valid_payload = {
        "text": "probe output",
        "meta_info": {
            "completion_tokens": acceptance.QA_SPEED_PROBE_MAX_NEW_TOKENS,
            "decode_throughput": 12.5,
        },
    }
    missing_field = {
        "text": "probe output",
        "meta_info": {
            "completion_tokens": acceptance.QA_SPEED_PROBE_MAX_NEW_TOKENS,
            "decode_tokens_per_second": 12.5,
        },
    }
    non_finite = {
        "text": "probe output",
        "meta_info": {
            "completion_tokens": acceptance.QA_SPEED_PROBE_MAX_NEW_TOKENS,
            "decode_throughput": float("inf"),
        },
    }

    valid = acceptance.validate_qa_speed_probe(valid_payload)
    missing = acceptance.validate_qa_speed_probe(missing_field)
    invalid = acceptance.validate_qa_speed_probe(non_finite)

    assert valid["pass"] is True
    assert valid["speed"]["decode_throughput"]["mean"] == 12.5
    assert missing["pass"] is False
    assert invalid["pass"] is False
    assert any("meta_info.decode_throughput" in item for item in missing["failures"])


def test_gsm_exact_200_rows_and_accuracy_is_non_blocking(tmp_path: Path):
    result_path = tmp_path / "result.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    _write_gsm_artifacts(result_path, raw_path, correct_count=0)

    validation = acceptance.validate_gsm_artifacts(result_path, raw_path)

    assert validation["pass"] is True
    assert validation["raw_row_count"] == 200
    assert validation["accuracy"] == {
        "value": 0.0,
        "raw_value": 0.0,
        "blocking": False,
        "minimum": None,
    }
    assert validation["artifacts"]["result"]["sha256"]
    assert validation["artifacts"]["raw_result"]["sha256"]


def test_gsm_decode_speed_requires_all_200_meta_info_rows(tmp_path: Path):
    speed_path = tmp_path / "decode-speed.jsonl"
    _write_gsm_speed_artifact(speed_path, throughput=10.0)

    validation = acceptance.validate_gsm_decode_speed_artifact(speed_path)

    assert validation["pass"] is True
    assert validation["response_count"] == acceptance.GSM_QUESTIONS
    assert validation["decode_throughput"]["valid_response_count"] == 200
    assert validation["decode_throughput"]["minimum"] == 10.0
    assert validation["decode_throughput"]["maximum"] == 209.0
    assert validation["artifact"]["sha256"]


@pytest.mark.parametrize("bad_value", [None, 0, -1, "inf", True])
def test_gsm_decode_speed_fails_closed_for_any_bad_response(
    tmp_path: Path, bad_value: object
):
    speed_path = tmp_path / "decode-speed.jsonl"
    _write_gsm_speed_artifact(speed_path)
    rows = [json.loads(line) for line in speed_path.read_text().splitlines()]
    rows[137]["meta_info"]["decode_throughput"] = bad_value
    speed_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    validation = acceptance.validate_gsm_decode_speed_artifact(speed_path)

    assert validation["pass"] is False
    assert validation["decode_throughput"]["valid_response_count"] == 199
    assert any("response 137" in item for item in validation["failures"])


def test_gsm_fails_on_incomplete_output_and_non_exact_prompt_ids(tmp_path: Path):
    result_path = tmp_path / "result.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    _write_gsm_artifacts(result_path, raw_path, correct_count=100)
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[7]["output"] = ""
    rows[8]["prompt_id"] = 7
    raw_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    validation = acceptance.validate_gsm_artifacts(result_path, raw_path)

    assert validation["pass"] is False
    assert any("output is not a completed" in item for item in validation["failures"])
    assert any("exactly 0..199" in item for item in validation["failures"])


def test_gsm_command_freezes_session_c_parameters(tmp_path: Path):
    args = acceptance.parse_args(
        [
            "gsm8k",
            "--base-url",
            "http://127.0.0.1:30100",
            "--data-path",
            str(tmp_path / "test.jsonl"),
            "--result-file",
            str(tmp_path / "result.jsonl"),
            "--raw-result-file",
            str(tmp_path / "raw.jsonl"),
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "acceptance.json"),
        ]
    )

    command = acceptance.build_gsm_command(args)

    assert command[0] == args.python
    assert command[1] == "-c"
    assert command[2] == acceptance.GSM_BENCHMARK_WORKER
    assert command[3] == str(args.benchmark_script)
    assert command[4] == str(acceptance._gsm_decode_speed_path(args.output))
    assert command[command.index("--num-questions") + 1] == "200"
    assert command[command.index("--num-shots") + 1] == "5"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--max-new-tokens") + 1] == "512"
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--top-p") + 1] == "1"


def test_gsm_worker_wraps_checked_in_dumper_and_captures_meta_info(tmp_path: Path):
    package = tmp_path / "sglang" / "test"
    package.mkdir(parents=True)
    (tmp_path / "sglang" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "test_utils.py").write_text(
        """
from pathlib import Path


def dump_bench_raw_result(path, states, preds, labels):
    Path(path).write_text("original dumper called", encoding="utf-8")
""".lstrip(),
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.py"
    benchmark.write_text(
        """
import sys

from sglang.test.test_utils import dump_bench_raw_result


class State:
    def __init__(self, value):
        self.value = value

    def get_meta_info(self, name):
        assert name == "answer"
        return {"completion_tokens": 8, "decode_throughput": self.value}


states = [State(10.0), State(20.0)]
dump_bench_raw_result(sys.argv[1], states, [0, 0], [0, 0])
""".lstrip(),
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw.jsonl"
    speed_path = tmp_path / "speed.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            acceptance.GSM_BENCHMARK_WORKER,
            str(benchmark),
            str(speed_path),
            str(raw_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert raw_path.read_text() == "original dumper called"
    rows = [json.loads(line) for line in speed_path.read_text().splitlines()]
    assert [row["prompt_id"] for row in rows] == [0, 1]
    assert [row["meta_info"]["decode_throughput"] for row in rows] == [10.0, 20.0]


def test_metric_snapshot_preserves_exact_text_and_sha(tmp_path: Path):
    body = _metrics(graph=[1], decode_none=[0], itl_sum=1.0, itl_count=2).encode()

    class Response:
        status = 200
        headers = {"content-type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self.status

        def read(self):
            return body

    raw_path = tmp_path / "before.prom"
    snapshot = acceptance.fetch_metrics(
        "http://server",
        timeout=1,
        raw_path=raw_path,
        opener=lambda *_a, **_k: Response(),
    )

    assert raw_path.read_bytes() == body
    assert snapshot["raw_text"].encode() == body
    assert snapshot["raw_sha256"] == acceptance._sha256_bytes(body)


def test_exception_artifact_keeps_strict_contract(tmp_path: Path, monkeypatch):
    output = tmp_path / "qa.json"
    argv = [
        "qa",
        "--base-url",
        "http://server",
        "--model",
        "glm",
        "--output",
        str(output),
        "--tp-size",
        "2",
    ]
    parsed = acceptance.parse_args(argv)
    monkeypatch.setattr(
        acceptance,
        "run_qa",
        lambda _args: (_ for _ in ()).throw(RuntimeError("deliberate")),
    )

    returncode = acceptance.main(argv)
    payload = json.loads(output.read_text())

    assert returncode == 1
    assert (
        acceptance.validate_evidence_envelope(
            payload,
            expected_kind=acceptance.QA_KIND,
            expected_contract=acceptance.qa_contract(parsed),
        )
        == []
    )
    assert payload["pass"] is False
    assert payload["failures"] == ["RuntimeError: deliberate"]
