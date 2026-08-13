import copy
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "glm5_next_session_c_graph_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "glm5_next_session_c_graph_acceptance", SCRIPT
)
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ACCEPTANCE)


class _FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.payload


class _FakeSGLang:
    def __init__(self, *, expose_graph_counter=True, nonfinite=False):
        self.expose_graph_counter = expose_graph_counter
        self.nonfinite = nonfinite
        self.metrics_calls = 0
        self.flush_calls = 0
        self.generate_bodies = []

    def _metrics(self):
        self.metrics_calls += 1
        if not self.expose_graph_counter:
            return b"# graph counter deliberately absent\nsglang:running_requests 0\n"
        value = 10 if self.metrics_calls == 1 else 73
        lines = ["# TYPE sglang:cuda_graph_passes_total counter"]
        for rank in range(8):
            lines.append(
                "sglang:cuda_graph_passes_total"
                f'{{mode="decode_cuda_graph",tp_rank="{rank}"}} {value}'
            )
            lines.append(
                "sglang:cuda_graph_passes_total"
                f'{{mode="decode_none",tp_rank="{rank}"}} 99'
            )
        return ("\n".join(lines) + "\n").encode()

    def _generate(self, body):
        result = []
        for row_index, input_ids in enumerate(body["input_ids"]):
            prompt_token = input_ids[0]
            output_ids = [prompt_token + step + 1 for step in range(8)]
            output_top_logprobs = []
            for step, output_id in enumerate(output_ids):
                first_logprob = -0.1 - step / 100
                if (
                    self.nonfinite
                    and not self.generate_bodies[:-1]
                    and row_index == 0
                    and step == 0
                ):
                    first_logprob = float("nan")
                output_top_logprobs.append(
                    [
                        [first_logprob, output_id],
                        [-1.1 - step / 100, output_id + 777],
                    ]
                )
            result.append(
                {
                    "output_ids": output_ids,
                    "meta_info": {
                        "completion_tokens": 8,
                        "output_top_logprobs": output_top_logprobs,
                    },
                }
            )
        return json.dumps(result).encode()

    def __call__(self, request, timeout):
        assert timeout == 30
        if request.full_url.endswith("/metrics"):
            assert request.get_method() == "GET"
            return _FakeResponse(self._metrics())
        if request.full_url.endswith("/flush_cache"):
            assert request.get_method() == "POST"
            assert json.loads(request.data) == {}
            self.flush_calls += 1
            return _FakeResponse(b"Cache flushed.\n")
        if request.full_url.endswith("/generate"):
            assert request.get_method() == "POST"
            body = json.loads(request.data)
            self.generate_bodies.append(body)
            return _FakeResponse(self._generate(body))
        raise AssertionError(f"unexpected URL: {request.full_url}")


def _args(tmp_path, *, mode="graph", eager_json=None):
    argv = [
        "--base-url",
        "http://example.test",
        "--output",
        str(tmp_path / f"{mode}.json"),
        "--mode",
        mode,
        "--top-k",
        "2",
        "--request-timeout",
        "30",
    ]
    if eager_json is not None:
        argv.extend(["--eager-json", str(eager_json)])
    return ACCEPTANCE.build_parser().parse_args(argv)


def _collect(tmp_path, *, mode="graph", expose_graph_counter=True):
    eager_json = None
    if mode == "graph":
        eager, _ = _collect(tmp_path, mode="eager")
        eager_json = tmp_path / "eager-reference.json"
        ACCEPTANCE._atomic_write_json(eager_json, eager)
    fake = _FakeSGLang(expose_graph_counter=expose_graph_counter)
    payload = ACCEPTANCE.collect_acceptance(
        _args(tmp_path, mode=mode, eager_json=eager_json), opener=fake
    )
    return payload, fake


def test_fixtures_are_stable_distinct_and_exact_graph_buckets():
    first = ACCEPTANCE.build_fixtures(154880)
    second = ACCEPTANCE.build_fixtures(154880)

    assert first == second
    assert [fixture["batch_size"] for fixture in first] == [1, 2, 4]
    for fixture in first:
        batch_size = fixture["batch_size"]
        assert fixture["prompt_length"] == 1
        assert len(fixture["a_tokens"]) == batch_size
        assert len(fixture["poison_tokens"]) == batch_size
        assert all(
            a_token != poison_token
            for a_token, poison_token in zip(
                fixture["a_tokens"], fixture["poison_tokens"]
            )
        )


def test_metrics_parser_accepts_real_and_compatible_counter_formats_only():
    text = """
# HELP sglang:cuda_graph_passes_total Forward passes by mode.
sglang:cuda_graph_passes_total{mode="decode_cuda_graph",tp_rank="0"} 10
sglang_cuda_graph_passes_total{forward_mode="decode_cuda_graph"} 3
runtime_decode_cuda_graph_invocations_count 4
sglang:cuda_graph_passes_total{mode="decode_none"} 100
sglang:is_cuda_graph 1
sglang:decode_cuda_graph_created 123
cuda graph: True
"""

    counter = ACCEPTANCE.decode_cuda_graph_counter(text)

    assert counter["available"]
    assert counter["total"] == 17
    assert {sample["name"] for sample in counter["samples"]} == {
        "sglang:cuda_graph_passes_total",
        "sglang_cuda_graph_passes_total",
        "runtime_decode_cuda_graph_invocations_count",
    }


def test_metrics_gate_requires_two_http_counter_snapshots_and_growth():
    def snapshot(total=None, *, success=True):
        by_rank = (
            {
                "decode_cuda_graph": {"0": total},
                "decode_none": {"0": 0},
            }
            if total is not None
            else {"decode_cuda_graph": {}, "decode_none": {}}
        )
        return {
            "success": success,
            "decode_cuda_graph_counter": {
                "available": total is not None,
                "total": total,
                "samples": [],
            },
            "cuda_graph_counters_by_rank": by_rank,
        }

    passing = ACCEPTANCE.compare_metric_snapshots(
        snapshot(10), snapshot(19), tp_size=1, minimum_graph_delta_per_rank=9
    )
    assert passing["pass"]
    assert passing["delta"] == 9
    assert passing["source"] == "HTTP /metrics only"
    assert passing["log_evidence_used"] is False

    assert not ACCEPTANCE.compare_metric_snapshots(snapshot(None), snapshot(19))["pass"]
    assert not ACCEPTANCE.compare_metric_snapshots(snapshot(19), snapshot(19))["pass"]
    assert not ACCEPTANCE.decode_cuda_graph_counter("cuda graph: True\n")["available"]


def test_mocked_http_graph_acceptance_covers_exact_a_poison_a_contract(tmp_path):
    eager, _ = _collect(tmp_path, mode="eager")
    eager_path = tmp_path / "eager-reference.json"
    ACCEPTANCE._atomic_write_json(eager_path, eager)
    args = _args(tmp_path, eager_json=eager_path)
    fake = _FakeSGLang()

    assert ACCEPTANCE.run(args, opener=fake) == 0

    payload = json.loads(Path(args.output).read_text())
    assert payload["pass"]
    assert payload["collection_pass"]
    assert [case["batch_size"] for case in payload["cases"]] == [1, 2, 4]
    assert fake.metrics_calls == 2
    assert fake.flush_calls == 1
    assert len(fake.generate_bodies) == 9
    assert [len(body["input_ids"]) for body in fake.generate_bodies] == [
        1,
        1,
        1,
        2,
        2,
        2,
        4,
        4,
        4,
    ]

    for offset in range(0, len(fake.generate_bodies), 3):
        a_before, poison, a_after = fake.generate_bodies[offset : offset + 3]
        assert a_before["input_ids"] == a_after["input_ids"]
        assert a_before["input_ids"] != poison["input_ids"]

    for body in fake.generate_bodies:
        assert all(len(row) == 1 for row in body["input_ids"])
        assert body["sampling_params"] == {
            "temperature": 0,
            "max_new_tokens": 8,
            "ignore_eos": True,
            "skip_special_tokens": False,
        }
        assert body["return_logprob"] is True
        assert body["top_logprobs_num"] == 2

    for case in payload["cases"]:
        assert [phase["phase"] for phase in case["phases"]] == [
            "a_before",
            "poison",
            "a_after",
        ]
        assert case["a_poison_a"]["a_repeat"]["pass"]
        assert case["a_poison_a"]["poison"]["observed"]
        for phase in case["phases"]:
            assert phase["raw_response"]
            for sequence in phase["sequences"]:
                assert len(sequence["generated_ids"]) == 8
                assert len(sequence["steps"]) == 8
                assert all(len(step["top_logprobs"]) == 2 for step in sequence["steps"])

    gate = payload["metrics"]["decode_cuda_graph_gate"]
    assert gate["pass"]
    assert gate["delta"] == 504
    assert all(item["graph_delta"] == 63 for item in gate["per_rank"].values())
    assert all(item["decode_none_delta"] == 0 for item in gate["per_rank"].values())
    assert gate["source"] == "HTTP /metrics only"
    assert gate["log_evidence_used"] is False
    assert payload["metrics"]["before"]["raw_text"]
    assert payload["metrics"]["after"]["raw_text"]


def test_eager_baseline_does_not_require_graph_counter(tmp_path):
    payload, fake = _collect(tmp_path, mode="eager", expose_graph_counter=False)

    assert payload["pass"]
    assert payload["collection_pass"]
    assert payload["metrics"]["decode_cuda_graph_gate"]["required"] is False
    assert not payload["metrics"]["decode_cuda_graph_gate"]["pass"]
    assert fake.metrics_calls == 2


def test_graph_mode_fails_without_http_counter_even_when_outputs_pass(tmp_path):
    payload, _ = _collect(tmp_path, mode="graph", expose_graph_counter=False)

    assert payload["collection_pass"]
    assert not payload["pass"]
    gate = payload["metrics"]["decode_cuda_graph_gate"]
    assert gate["required"] is True
    assert not gate["pass"]
    assert gate["source"] == "HTTP /metrics only"
    assert gate["log_evidence_used"] is False
    assert any(
        "no decode_cuda_graph counter" in failure for failure in payload["failures"]
    )


def test_graph_mode_requires_eager_json_before_http(tmp_path):
    fake = _FakeSGLang()
    args = _args(tmp_path, mode="graph")

    assert ACCEPTANCE.run(args, opener=fake) == 1
    payload = json.loads(Path(args.output).read_text())
    assert not payload["pass"]
    assert any("--eager-json is required" in item for item in payload["failures"])
    assert fake.metrics_calls == 0
    assert fake.flush_calls == 0
    assert not fake.generate_bodies


def test_eager_graph_comparison_enforces_tokens_and_logprob_tolerance(tmp_path):
    eager, _ = _collect(tmp_path, mode="eager")
    graph, _ = _collect(tmp_path, mode="graph")

    within_tolerance = copy.deepcopy(graph)
    within_tolerance["cases"][0]["phases"][0]["sequences"][0]["steps"][0][
        "top_logprobs"
    ][0]["logprob"] += 0.01
    assert ACCEPTANCE.compare_acceptance_payloads(eager, within_tolerance)["pass"]

    token_mismatch = copy.deepcopy(graph)
    token_mismatch["cases"][0]["phases"][0]["sequences"][0]["generated_ids"][0] += 1
    token_result = ACCEPTANCE.compare_acceptance_payloads(eager, token_mismatch)
    assert not token_result["pass"]
    assert any(
        "generated token mismatch" in failure for failure in token_result["failures"]
    )

    logprob_mismatch = copy.deepcopy(graph)
    logprob_mismatch["cases"][0]["phases"][0]["sequences"][0]["steps"][0][
        "top_logprobs"
    ][0]["logprob"] += 1
    logprob_result = ACCEPTANCE.compare_acceptance_payloads(eager, logprob_mismatch)
    assert not logprob_result["pass"]
    assert any("outside" in failure for failure in logprob_result["failures"])


def test_graph_run_loads_eager_json_and_records_comparison_evidence(tmp_path):
    eager, _ = _collect(tmp_path, mode="eager")
    eager_path = tmp_path / "eager-reference.json"
    ACCEPTANCE._atomic_write_json(eager_path, eager)
    args = _args(tmp_path, mode="graph", eager_json=eager_path)

    assert ACCEPTANCE.run(args, opener=_FakeSGLang()) == 0

    graph = json.loads(Path(args.output).read_text())
    assert graph["pass"]
    assert graph["comparison"]["pass"]
    assert graph["comparison"]["tokens_exact_required"] is True
    assert graph["comparison"]["atol"] == 0.05
    assert graph["comparison"]["rtol"] == 0.05
    assert graph["comparison"]["eager_json"] == str(eager_path.resolve())
    assert graph["comparison"]["eager_sha256"] == ACCEPTANCE._sha256_file(eager_path)


def test_nonfinite_response_fails_but_complete_strict_json_is_preserved(tmp_path):
    args = _args(tmp_path, mode="eager")
    fake = _FakeSGLang(nonfinite=True)

    assert ACCEPTANCE.run(args, opener=fake) == 1

    serialized = Path(args.output).read_text()
    assert "NaN" in serialized
    payload = json.loads(
        serialized,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    assert not payload["pass"]
    first_phase = payload["cases"][0]["phases"][0]
    assert (
        first_phase["raw_response"][0]["meta_info"]["output_top_logprobs"][0][0][0]
        == "NaN"
    )
    assert any("non-finite logprob" in failure for failure in first_phase["failures"])
