import copy
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "glm5_next_session_c_final_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location(
    "glm5_next_session_c_final_acceptance", SCRIPT
)
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ACCEPTANCE)


def _round(*, round_index=0, layers=None, compute_ms=None):
    layers = list(range(3, 45)) if layers is None else layers
    compute_ms = [1.0] * len(layers) if compute_ms is None else compute_ms
    return {
        "round_index": round_index,
        "layers": layers,
        "compute_ms": compute_ms,
        "expert_update_ms": [None] * len(layers),
        "complete": len(layers) == 42,
    }


def test_layerwise_log_parser_and_progress_gate_accept_exact_rounds():
    text = "\n".join(
        ["noise"]
        + [
            "KT layerwise prefill: "
            f"layer {layer} compute = {round_index + layer / 100:.2f} ms"
            for round_index in range(2)
            for layer in range(3, 45)
        ]
    )
    current = ACCEPTANCE.parse_layerwise_summaries(text)

    assert len(current) == 2
    assert not ACCEPTANCE.validate_layerwise_progress(
        None, current, expected_rounds=2
    )


def test_layerwise_log_parser_accepts_optional_expert_update_timing():
    text = "\n".join(
        "KT layerwise prefill: "
        f"layer {layer} compute = 1.25 ms, expert update = 0.50 ms"
        for layer in range(3, 45)
    )

    summaries = ACCEPTANCE.parse_layerwise_summaries(text)

    assert len(summaries) == 1
    assert summaries[0]["expert_update_ms"] == [0.5] * 42
    assert not ACCEPTANCE.validate_layerwise_progress(
        None, summaries, expected_rounds=1
    )


def test_layerwise_partial_round_is_retained_and_rejected():
    text = "\n".join(
        f"KT layerwise prefill: layer {layer} compute = 1.00 ms"
        for layer in range(3, 44)
    )

    summaries = ACCEPTANCE.parse_layerwise_summaries(text)
    failures = ACCEPTANCE.validate_layerwise_progress(
        None, summaries, expected_rounds=1
    )

    assert len(summaries) == 1
    assert summaries[0]["complete"] is False
    assert any("incomplete" in failure for failure in failures)


def test_layerwise_progress_fails_closed_on_missing_or_out_of_order_round():
    bad = _round(layers=[3, 5, 4, *range(6, 45)])
    failures = ACCEPTANCE.validate_layerwise_progress(
        None, [bad], expected_rounds=2
    )

    assert any("expected 2" in failure for failure in failures)
    assert any("ordered layers" in failure for failure in failures)


def test_expected_layerwise_rounds_keeps_subthreshold_tail_hybrid():
    assert ACCEPTANCE.expected_layerwise_rounds(128, 4096) == 0
    assert ACCEPTANCE.expected_layerwise_rounds(4096, 4096) == 1
    assert ACCEPTANCE.expected_layerwise_rounds(4097, 4096) == 1
    assert ACCEPTANCE.expected_layerwise_rounds(8193, 4096) == 2
    assert ACCEPTANCE.expected_layerwise_rounds(500_000, 4096) == 122
    assert ACCEPTANCE.expected_layerwise_rounds(4096, 512) == 0


def test_prometheus_graph_counter_sums_matching_modes_only():
    metrics = """
# HELP sglang:cuda_graph_passes_total graph passes
sglang:cuda_graph_passes_total{tp_rank="0",mode="decode_cuda_graph"} 7
sglang:cuda_graph_passes_total{tp_rank="1",mode="decode_cuda_graph"} 5
sglang:cuda_graph_passes_total{tp_rank="0",mode="decode_none"} 99
sglang:cuda_graph_passes_created{tp_rank="0",mode="decode_cuda_graph"} 1
"""

    assert ACCEPTANCE.decode_graph_counter(metrics) == 12
    assert ACCEPTANCE.decode_graph_counter("# no samples\n") is None


def test_per_rank_graph_parser_rejects_unofficial_rank_alias():
    metrics = "\n".join(
        [
            'sglang:cuda_graph_passes_total{rank="0",mode="decode_cuda_graph"} 7',
            'sglang:cuda_graph_passes_total{rank="0",mode="decode_none"} 0',
        ]
    )

    counters = ACCEPTANCE.cuda_graph_counters_by_rank(metrics)
    result = ACCEPTANCE.validate_graph_counter_progress(
        counters, counters, tp_size=1, minimum_graph_delta_per_rank=0
    )

    assert not result["pass"]
    assert any("exact TP ranks" in failure for failure in result["failures"])


def test_per_rank_graph_gate_rejects_decode_none_growth():
    before = {
        "decode_cuda_graph": {str(rank): 10.0 for rank in range(2)},
        "decode_none": {str(rank): 4.0 for rank in range(2)},
    }
    after = {
        "decode_cuda_graph": {str(rank): 17.0 for rank in range(2)},
        "decode_none": {str(rank): 4.0 for rank in range(2)},
    }
    assert ACCEPTANCE.validate_graph_counter_progress(
        before, after, tp_size=2, minimum_graph_delta_per_rank=7
    )["pass"]

    after["decode_none"]["1"] += 1
    result = ACCEPTANCE.validate_graph_counter_progress(
        before, after, tp_size=2, minimum_graph_delta_per_rank=7
    )
    assert not result["pass"]
    assert any("decode_none" in item for item in result["failures"])


def test_per_rank_graph_gate_treats_unpublished_decode_none_as_zero():
    before = {
        "decode_cuda_graph": {str(rank): 10.0 for rank in range(2)},
        "decode_none": {},
    }
    after = {
        "decode_cuda_graph": {str(rank): 17.0 for rank in range(2)},
        "decode_none": {},
    }

    result = ACCEPTANCE.validate_graph_counter_progress(
        before, after, tp_size=2, minimum_graph_delta_per_rank=7
    )

    assert result["pass"]
    assert all(
        item["decode_none_delta"] == 0
        for item in result["per_rank"].values()
    )


def test_per_rank_graph_gate_rejects_lazy_decode_none_growth_and_disappearance():
    graph_before = {"0": 10.0, "1": 10.0}
    graph_after = {"0": 17.0, "1": 17.0}
    appeared = ACCEPTANCE.validate_graph_counter_progress(
        {"decode_cuda_graph": graph_before, "decode_none": {}},
        {"decode_cuda_graph": graph_after, "decode_none": {"0": 1.0}},
        tp_size=2,
        minimum_graph_delta_per_rank=7,
    )
    disappeared = ACCEPTANCE.validate_graph_counter_progress(
        {"decode_cuda_graph": graph_before, "decode_none": {"0": 1.0}},
        {"decode_cuda_graph": graph_after, "decode_none": {}},
        tp_size=2,
        minimum_graph_delta_per_rank=7,
    )

    assert not appeared["pass"]
    assert not disappeared["pass"]
    assert appeared["per_rank"]["0"]["decode_none_delta"] == 1
    assert disappeared["per_rank"]["0"]["decode_none_delta"] == -1


def test_per_rank_graph_gate_rejects_unexpected_decode_none_rank():
    graph = {"0": 10.0, "1": 10.0}
    result = ACCEPTANCE.validate_graph_counter_progress(
        {"decode_cuda_graph": graph, "decode_none": {}},
        {"decode_cuda_graph": graph, "decode_none": {"2": 0.0}},
        tp_size=2,
        minimum_graph_delta_per_rank=0,
    )

    assert not result["pass"]
    assert any("unexpected TP ranks" in item for item in result["failures"])


def _bucket_payload(label, outputs):
    expanded = {
        batch_size: [[token_ids[0] + step for step in range(8)] for token_ids in rows]
        for batch_size, rows in outputs.items()
    }
    return {
        "schema_version": ACCEPTANCE.SCHEMA_VERSION,
        "kind": "session_c_exact_cuda_graph_buckets",
        "label": label,
        "pass": True,
        "contract": {
            "batch_sizes": [1, 2, 4],
            "input_tokens": 128,
            "output_tokens": 8,
            "top_k": 64,
            "vocab_size": ACCEPTANCE.DEFAULT_VOCAB_SIZE,
            "fixture_seed": ACCEPTANCE.FIXTURE_SEED,
            "require_decode_cuda_graph": True,
            "compliant": True,
        },
        "cases": [
            {
                "batch_size": batch_size,
                "input_rows_sha256": f"fixture-{batch_size}",
                "output_rows": expanded[batch_size],
                "top_k": 64,
                "top_logprobs": [
                    [
                        [
                            {"token_id": token_id, "logprob": -0.25 - rank}
                            for rank, token_id in enumerate(range(64))
                        ]
                        for _ in range(8)
                    ]
                    for _ in rows
                ],
                "pass": True,
            }
            for batch_size, rows in outputs.items()
        ],
    }


def test_bucket_comparison_requires_all_buckets_and_exact_outputs():
    outputs = {1: [[1]], 2: [[2], [3]], 4: [[4], [5], [6], [7]]}
    baseline = _bucket_payload("baseline", outputs)
    candidate = _bucket_payload("candidate", copy.deepcopy(outputs))
    assert ACCEPTANCE.compare_bucket_payloads(baseline, candidate)["pass"]

    candidate["cases"][1]["output_rows"][0][0] = 99
    result = ACCEPTANCE.compare_bucket_payloads(baseline, candidate)
    assert not result["pass"]
    assert any("batch 2" in failure for failure in result["failures"])


def test_bucket_comparison_rejects_non_argmax_logprob_drift():
    outputs = {1: [[1]], 2: [[2], [3]], 4: [[4], [5], [6], [7]]}
    baseline = _bucket_payload("baseline", copy.deepcopy(outputs))
    candidate = _bucket_payload("candidate", copy.deepcopy(outputs))
    candidate["cases"][2]["top_logprobs"][0][0][0]["logprob"] = -0.5

    result = ACCEPTANCE.compare_bucket_payloads(baseline, candidate)

    assert not result["pass"]
    assert any("top logprobs differ" in failure for failure in result["failures"])


def test_final_contract_defaults_are_500k_plus_1024():
    args = ACCEPTANCE.build_parser().parse_args(
        [
            "long",
            "--label",
            "final",
            "--output",
            "/tmp/final.json",
            "--server-log",
            "/tmp/server.log",
        ]
    )

    assert args.input_length == 500_000
    assert args.output_length == 1024
    assert args.chunk_size == 4096


def test_bucket_contract_defaults_are_exact():
    args = ACCEPTANCE.build_parser().parse_args(
        ["buckets", "--label", "candidate", "--output", "/tmp/buckets.json"]
    )

    assert args.vocab_size == ACCEPTANCE.DEFAULT_VOCAB_SIZE
    assert args.fixture_seed == ACCEPTANCE.FIXTURE_SEED
    assert args.input_length == 128
    assert args.output_length == 8
    assert args.top_k == 64
    assert args.tp_size == 4


def test_final_long_contract_is_exact_not_a_minimum(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ACCEPTANCE,
        "_capture_request_evidence",
        lambda **kwargs: {"failures": [], "pass": True},
    )
    output = tmp_path / "long.json"
    server_log = tmp_path / "server.log"
    server_log.write_text("")
    args = ACCEPTANCE.build_parser().parse_args(
        [
            "long",
            "--label",
            "too-long",
            "--output",
            str(output),
            "--server-log",
            str(server_log),
            "--input-length",
            "500001",
            "--output-length",
            "1025",
        ]
    )

    assert ACCEPTANCE.run_long(args) == 1
    payload = json.loads(output.read_text())
    assert not payload["contract"]["compliant"]
    assert payload["contract"]["input_tokens"] == 500_000
    assert payload["contract"]["output_tokens"] == 1024


def test_fatal_log_scan_ignores_info_but_rejects_runtime_failures():
    assert not ACCEPTANCE.fatal_log_lines("INFO initialized fallback_count=0")
    failures = ACCEPTANCE.fatal_log_lines(
        "CUDA out of memory\nvalue is NaN\nTraceback (most recent call last)"
    )
    assert len(failures) == 3


def _startup_log(*, threshold=1024, lazy=True, include_allocation=True):
    server_args = (
        "server_args=ServerArgs(tp_size=4, chunked_prefill_size=4096, "
        "kt_method='FP8', kt_gpu_prefill_token_threshold="
        f"{threshold})"
    )
    allocation = (
        "KT GLM-5-Next FP8 layerwise prefill lazily allocated generic "
        "SharedFullContext: gpu_full_layer_slots=1 host_expert_slots=2 "
        "gpu_slot_bytes=1812381696 host_buffer_bytes=12582912 device=cuda:0"
    )
    memory_pool = "Memory pool end. avail mem=12.34 GB"
    ready = "The server is fired up and ready to roll!"
    ordered = [memory_pool, ready]
    if include_allocation:
        ordered = [*ordered, allocation] if lazy else [allocation, *ordered]
    return "\n".join([server_args, *ordered])


def test_layerwise_startup_requires_one_lazy_single_gpu_two_host_allocation():
    result = ACCEPTANCE.validate_layerwise_startup(_startup_log())

    assert result["pass"]
    assert result["allocation"]["gpu_full_layer_slots"] == 1
    assert result["allocation"]["host_expert_slots"] == 2
    assert result["allocation"]["gpu_slot_bytes"] == 1812381696

    eager = ACCEPTANCE.validate_layerwise_startup(_startup_log(lazy=False))
    duplicate = ACCEPTANCE.validate_layerwise_startup(
        _startup_log() + "\n" + _startup_log().splitlines()[-1]
    )
    missing = ACCEPTANCE.validate_layerwise_startup(
        _startup_log(include_allocation=False)
    )
    pending = ACCEPTANCE.validate_layerwise_startup(
        _startup_log(include_allocation=False), require_allocation=False
    )
    assert not eager["pass"]
    assert any("not lazy" in failure for failure in eager["failures"])
    assert not duplicate["pass"]
    assert any("exactly one" in failure for failure in duplicate["failures"])
    assert not missing["pass"]
    assert pending["pass"]


def test_layerwise_startup_rejects_wrong_slots_and_legacy_protocol():
    wrong_slots = _startup_log().replace(
        "gpu_full_layer_slots=1 host_expert_slots=2",
        "gpu_full_layer_slots=2 host_expert_slots=1",
    )
    legacy = _startup_log() + "\n" + (
        "KT GLM-5-Next FP8 layerwise prefill eagerly allocated two raw "
        "E4M3+FP32 full-layer slots"
    )

    slots_result = ACCEPTANCE.validate_layerwise_startup(wrong_slots)
    legacy_result = ACCEPTANCE.validate_layerwise_startup(legacy)

    assert not slots_result["pass"]
    assert any("gpu_full_layer_slots" in item for item in slots_result["failures"])
    assert any("host_expert_slots" in item for item in slots_result["failures"])
    assert not legacy_result["pass"]
    assert any("legacy" in item for item in legacy_result["failures"])


def test_prefill_server_log_binds_explicit_threshold_and_tp4():
    threshold_1024 = ACCEPTANCE.validate_prefill_parity_server_log(
        _startup_log(threshold=1024), threshold=1024
    )
    threshold_0_log = "\n".join(
        [
            (
                "server_args=ServerArgs(tp_size=4, chunked_prefill_size=4096, "
                "kt_method='FP8', kt_gpu_prefill_token_threshold=0)"
            ),
            "The server is fired up and ready to roll!",
        ]
    )
    threshold_0 = ACCEPTANCE.validate_prefill_parity_server_log(
        threshold_0_log, threshold=0
    )

    assert threshold_1024["pass"]
    assert threshold_0["pass"]
    wrong_threshold = ACCEPTANCE.validate_prefill_parity_server_log(
        threshold_0_log, threshold=1024
    )
    assert not wrong_threshold["pass"]
    assert any(
        "kt_gpu_prefill_token_threshold" in failure
        for failure in wrong_threshold["failures"]
    )


def test_threshold_zero_rejects_nonfinal_layerwise_event():
    threshold_0_log = "\n".join(
        [
            (
                "server_args=ServerArgs(tp_size=4, chunked_prefill_size=4096, "
                "kt_method='FP8', kt_gpu_prefill_token_threshold=0)"
            ),
            "The server is fired up and ready to roll!",
            (
                "KT layerwise prefill: layer 3 compute = 1.25 ms"
            ),
        ]
    )

    result = ACCEPTANCE.validate_prefill_parity_server_log(
        threshold_0_log, threshold=0
    )

    assert not result["pass"]
    assert any("executed" in failure for failure in result["failures"])


def test_completed_request_requires_length_finish_and_in_range_tokens():
    good = {
        "success": True,
        "finish_reason": {"type": "length", "length": 16},
        "output_ids": list(range(16)),
    }
    assert not ACCEPTANCE.validate_completed_request(
        good, output_tokens=16, vocab_size=154880
    )

    bad = copy.deepcopy(good)
    bad["finish_reason"] = {"type": "stop"}
    bad["output_ids"][-1] = 154880
    failures = ACCEPTANCE.validate_completed_request(
        bad, output_tokens=16, vocab_size=154880
    )
    assert any("finish_reason" in failure for failure in failures)
    assert any("out-of-range" in failure for failure in failures)


def _prefill_parity_payload(threshold):
    cases = []
    for length in ACCEPTANCE.LAYERWISE_LENGTHS:
        fixture = ACCEPTANCE.build_fixture(
            length,
            ACCEPTANCE.DEFAULT_VOCAB_SIZE,
            salt=89,
            seed=ACCEPTANCE.LAYERWISE_FIXTURE_SEED,
        )
        fixture.pop("input_ids")
        output_ids = [1000 + step for step in range(16)]
        top_logprobs = []
        for generated_id in output_ids:
            top_logprobs.append(
                [
                    {
                        "token_id": generated_id + rank,
                        "logprob": -0.1 - rank,
                    }
                    for rank in range(ACCEPTANCE.PREFILL_PARITY_COLLECTION_TOP_K)
                ]
            )
        summaries = []
        expected_rounds = ACCEPTANCE.expected_layerwise_rounds(length, 4096)
        if threshold == ACCEPTANCE.LAYERWISE_TOKEN_THRESHOLD:
            summaries = [_round(round_index=index) for index in range(expected_rounds)]
        cases.append(
            {
                "input_tokens": length,
                "fixture": fixture,
                "output_ids": output_ids,
                "output_sha256": ACCEPTANCE._sha256_token_rows([output_ids]),
                "finish_reason": {"type": "length"},
                "prompt_tokens": length,
                "completion_tokens": 16,
                "top_logprobs": top_logprobs,
                "expected_layerwise_rounds": (
                    expected_rounds
                    if threshold == ACCEPTANCE.LAYERWISE_TOKEN_THRESHOLD
                    else 0
                ),
                "previous_layerwise_summary": None,
                "layerwise_summaries": summaries,
                "server_log_window": {
                    "path": "/tmp/server.log",
                    "start_offset": 10,
                    "end_offset": 20,
                    "sha256": "b" * 64,
                    "fatal_lines": [],
                },
                "failures": [],
                "pass": True,
            }
        )
    server_args = {
        "tp_size": "4",
        "chunked_prefill_size": "4096",
        "kt_method": "'FP8'",
        "kt_gpu_prefill_token_threshold": str(threshold),
    }
    return {
        "schema_version": ACCEPTANCE.SCHEMA_VERSION,
        "kind": "glm5_next_session_c_prefill_threshold_collection",
        "pass": True,
        "contract": {
            "threshold": threshold,
            "lengths": list(ACCEPTANCE.LAYERWISE_LENGTHS),
            "output_tokens": 16,
            "chunk_size": 4096,
            "fixture_seed": 20260811,
            "tp_size": 4,
            "vocab_size": 154880,
            "collection_top_k": 256,
            "comparison_top_k": 64,
            "temperature": 0,
            "ignore_eos": True,
            "require_finish_reason_length": True,
            "require_exact_greedy_tokens": True,
            "compliant": True,
        },
        "server_evidence": {
            "threshold": threshold,
            "server_args": server_args,
            "layerwise_startup": (
                ACCEPTANCE.validate_layerwise_startup(_startup_log())
                if threshold == ACCEPTANCE.LAYERWISE_TOKEN_THRESHOLD
                else None
            ),
            "path": "/tmp/server.log",
            "initial_size": 1024,
            "initial_sha256": "a" * 64,
            "pass": True,
        },
        "cases": cases,
        "failures": [],
    }


def test_prefill_parity_comparison_accepts_exact_tokens_and_bounded_toplogprobs():
    baseline = _prefill_parity_payload(0)
    candidate = _prefill_parity_payload(1024)
    for case in candidate["cases"]:
        for step in case["top_logprobs"]:
            for item in step:
                item["logprob"] += 0.001

    result = ACCEPTANCE.compare_prefill_parity_payloads(baseline, candidate)

    assert result["pass"]
    assert len(result["diagnostics"]) == 4 * 16


def test_prefill_parity_comparison_rejects_token_or_toplogprob_drift():
    baseline = _prefill_parity_payload(0)
    token_candidate = _prefill_parity_payload(1024)
    token_case = token_candidate["cases"][0]
    token_case["output_ids"][0] = 150000
    token_case["top_logprobs"][0][0]["token_id"] = 150000
    token_case["output_sha256"] = ACCEPTANCE._sha256_token_rows(
        [token_case["output_ids"]]
    )
    token_result = ACCEPTANCE.compare_prefill_parity_payloads(baseline, token_candidate)
    assert not token_result["pass"]
    assert any("greedy output token" in failure for failure in token_result["failures"])

    logprob_candidate = _prefill_parity_payload(1024)
    logprob_candidate["cases"][0]["top_logprobs"][0][0]["logprob"] = -0.5
    logprob_result = ACCEPTANCE.compare_prefill_parity_payloads(
        baseline, logprob_candidate
    )
    assert not logprob_result["pass"]
    assert any(
        "top logprobs exceed" in failure for failure in logprob_result["failures"]
    )


def test_prefill_parity_comparison_fails_closed_on_missing_top_evidence():
    baseline = _prefill_parity_payload(0)
    candidate = _prefill_parity_payload(1024)
    candidate["cases"][0]["top_logprobs"][0].pop()

    result = ACCEPTANCE.compare_prefill_parity_payloads(baseline, candidate)

    assert not result["pass"]
    assert any(
        "top-logprob evidence is invalid" in failure for failure in result["failures"]
    )


def test_prefill_parity_parser_defaults_freeze_matrix_seed_and_tp():
    args = ACCEPTANCE.build_parser().parse_args(
        [
            "prefill-parity",
            "--label",
            "threshold-1024",
            "--output",
            "/tmp/prefill.json",
            "--server-log",
            "/tmp/server.log",
            "--threshold",
            "1024",
        ]
    )

    assert args.lengths == [128, 4096, 4097, 8193]
    assert args.output_length == 16
    assert args.fixture_seed == 20260811
    assert args.tp_size == 4
    assert args.vocab_size == 154880
    assert args.top_k == 256
