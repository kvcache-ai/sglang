import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).parents[3] / "scripts" / "glm5_next_session_c_oracle.py"
SPEC = importlib.util.spec_from_file_location("glm5_next_session_c_oracle", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORACLE)


def _kda_conv_correction(source_dtype="torch.bfloat16"):
    layer_ids = list(ORACLE.EXPECTED_HF_KDA_LAYER_IDS)
    source_dtype_by_layer = {str(layer_id): source_dtype for layer_id in layer_ids}
    return {
        "glm5_next_kda_conv_fp32": {
            "contract": ORACLE.HF_KDA_CONV_FP32_CONTRACT,
            "source_dtype_by_layer": source_dtype_by_layer,
            "converted_layer_ids": (
                layer_ids if source_dtype == "torch.bfloat16" else []
            ),
            "already_fp32_layer_ids": (
                layer_ids if source_dtype == "torch.float32" else []
            ),
            "runtime_dtype_by_layer": {
                str(layer_id): "torch.float32" for layer_id in layer_ids
            },
            "verified_layer_ids": layer_ids,
        }
    }


def _cache_audit(prompt_length, max_new_tokens):
    steps = []
    for step_index in range(max_new_tokens):
        before = 0 if step_index == 0 else prompt_length + step_index - 1
        input_length = prompt_length if step_index == 0 else 1
        after = before + input_length
        positions = list(range(before, after))
        steps.append(
            {
                "step": step_index,
                "input_length": input_length,
                "attention_mask_length": after,
                "cache_position": positions,
                "position_ids": positions,
                "before": {"sequence_length": before, "seen_tokens": before},
                "after": {"sequence_length": after, "seen_tokens": after},
            }
        )
    return {
        "cache_class": "transformers.cache_utils.DynamicCache",
        "use_cache": True,
        "initial": {"sequence_length": 0, "seen_tokens": 0},
        "steps": steps,
    }


def _payload(delta=0.0, token_id=7, *, candidate=False):
    payload = {
        "schema_version": ORACLE.SCHEMA_VERSION,
        "kind": "sglang_http_result" if candidate else "patched_transformers_oracle",
        "pass": True,
        "fixture_seed": ORACLE.FIXTURE_SEED,
        "max_new_tokens": 1,
        "top_k": 2,
        "cases": [
            {
                "name": "tokens_16",
                "prompt_length": 16,
                "input_sha256": "same",
                "generated_ids": [token_id],
                "steps": [
                    {
                        "token_id": token_id,
                        "top_logprobs": [
                            {"token_id": token_id, "logprob": -0.1 + delta},
                            {"token_id": 8, "logprob": -1.0 + delta},
                        ],
                    }
                ],
            }
        ],
    }
    if not candidate:
        payload["repetitions"] = 2
        payload["repeat_envelope"] = {"tokens_exact": True}
        payload["cache_contract"] = ORACLE.HF_CACHE_CONTRACT
        payload["oracle_corrections"] = _kda_conv_correction()
        payload["cases"][0]["cache_audit"] = _cache_audit(16, 1)
    return payload


def _rollout_payload(generated_ids, deltas=None, *, candidate=False):
    if deltas is None:
        deltas = [0.0] * len(generated_ids)
    assert len(generated_ids) == len(deltas)
    payload = {
        "schema_version": ORACLE.SCHEMA_VERSION,
        "kind": "sglang_http_result" if candidate else "patched_transformers_oracle",
        "pass": True,
        "fixture_seed": ORACLE.FIXTURE_SEED,
        "max_new_tokens": len(generated_ids),
        "top_k": 2,
        "cases": [
            {
                "name": "tokens_16",
                "prompt_length": 16,
                "input_sha256": "same",
                "generated_ids": generated_ids,
                "steps": [
                    {
                        "token_id": token_id,
                        "top_logprobs": [
                            {"token_id": 100 + step, "logprob": -0.1 + delta},
                            {"token_id": 200 + step, "logprob": -1.0 + delta},
                        ],
                    }
                    for step, (token_id, delta) in enumerate(zip(generated_ids, deltas))
                ],
            }
        ],
    }
    if not candidate:
        payload["repetitions"] = 2
        payload["repeat_envelope"] = {"tokens_exact": True}
        payload["cache_contract"] = ORACLE.HF_CACHE_CONTRACT
        payload["oracle_corrections"] = _kda_conv_correction()
        payload["cases"][0]["cache_audit"] = _cache_audit(16, len(generated_ids))
    return payload


class _FakeCache:
    def __init__(self):
        self.length = 0
        self.seen_tokens = 0

    def get_seq_length(self):
        return self.length


class _RolloutModel:
    def __init__(self, cache, *, return_none=False, grow=False):
        self.cache = cache
        self.return_none = return_none
        self.grow = grow
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.grow:
            increment = int(kwargs["input_ids"].shape[1])
            self.cache.length += increment
            self.cache.seen_tokens += increment
        returned_cache = None if self.return_none else self.cache
        logits = torch.tensor([[[0.0, 3.0, -2.0, 1.0]]])
        return SimpleNamespace(logits=logits, past_key_values=returned_cache)


def _small_fixture():
    return {
        "name": "tokens_3",
        "prompt_length": 3,
        "input_ids": [1, 2, 3],
        "input_sha256": "test",
    }


def _fake_hf_model(conv_dtype=torch.bfloat16):
    layer_types = [
        (
            "linear_attention"
            if layer_id in ORACLE.EXPECTED_HF_KDA_LAYER_IDS
            else "deepseek_sparse_attention"
        )
        for layer_id in range(ORACLE.EXPECTED_HF_LAYER_COUNT)
    ]
    layers = []
    channels = ORACLE.EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE[0]
    for layer_id, layer_type in enumerate(layer_types):
        self_attn = SimpleNamespace(layer_idx=layer_id)
        if layer_type == "linear_attention":
            self_attn.activation = "silu"
            self_attn.conv1d = torch.nn.Conv1d(
                channels,
                channels,
                kernel_size=4,
                padding=3,
                groups=channels,
                bias=False,
                device="meta",
                dtype=conv_dtype,
            )
        layers.append(SimpleNamespace(self_attn=self_attn))
    text_config = SimpleNamespace(
        layer_types=layer_types,
        linear_num_heads=64,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    return SimpleNamespace(
        config=SimpleNamespace(text_config=text_config),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )


def test_hf_kda_conv_repair_converts_and_attests_all_34_layers():
    model = _fake_hf_model()

    correction = ORACLE._repair_hf_kda_conv_weights(model)

    assert correction == _kda_conv_correction()["glm5_next_kda_conv_fp32"]
    assert not ORACLE._validate_serialized_hf_kda_conv_correction(
        {"oracle_corrections": {"glm5_next_kda_conv_fp32": correction}}
    )
    for layer_id in ORACLE.EXPECTED_HF_KDA_LAYER_IDS:
        conv1d = model.model.language_model.layers[layer_id].self_attn.conv1d
        assert conv1d.weight.dtype == torch.float32
        assert tuple(conv1d.weight.shape) == ORACLE.EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE


def test_hf_kda_conv_repair_accepts_already_fp32_weights():
    model = _fake_hf_model(torch.float32)

    correction = ORACLE._repair_hf_kda_conv_weights(model)

    assert (
        correction == _kda_conv_correction("torch.float32")["glm5_next_kda_conv_fp32"]
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("count", "KDA layer IDs differ"),
        ("shape", "kernel_size"),
        ("dtype", "neither BF16 nor FP32"),
        ("activation", "does not use fused SiLU"),
    ],
)
def test_hf_kda_conv_repair_fails_closed(mutation, error):
    model = _fake_hf_model()
    first_kda = model.model.language_model.layers[0].self_attn
    channels = ORACLE.EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE[0]
    if mutation == "count":
        model.config.text_config.layer_types[44] = "deepseek_sparse_attention"
    elif mutation == "shape":
        first_kda.conv1d = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            groups=channels,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
    elif mutation == "dtype":
        first_kda.conv1d.weight.data = first_kda.conv1d.weight.data.to(torch.float16)
    else:
        first_kda.activation = "gelu"

    with pytest.raises(RuntimeError, match=error):
        ORACLE._repair_hf_kda_conv_weights(model)


def test_hf_model_repairs_kda_conv_before_cpu_offload():
    source = SCRIPT.read_text(encoding="utf-8")
    load_model = source[
        source.index("def _load_hf_model") : source.index("def _cache_counter")
    ]

    assert load_model.index("_repair_hf_kda_conv_weights(model)") < load_model.index(
        "cpu_offload(model"
    )


def test_hf_rollout_rejects_none_cache(monkeypatch):
    cache = _FakeCache()
    model = _RolloutModel(cache, return_none=True)
    monkeypatch.setattr(ORACLE, "_new_hf_dynamic_cache", lambda _model: cache)

    with pytest.raises(RuntimeError, match="past_key_values=None"):
        ORACLE._run_hf_case(
            model, _small_fixture(), device="cpu", max_new_tokens=1, top_k=2
        )


def test_hf_rollout_rejects_stale_cache(monkeypatch):
    cache = _FakeCache()
    model = _RolloutModel(cache)
    monkeypatch.setattr(ORACLE, "_new_hf_dynamic_cache", lambda _model: cache)

    with pytest.raises(RuntimeError, match="expected exactly 3"):
        ORACLE._run_hf_case(
            model, _small_fixture(), device="cpu", max_new_tokens=1, top_k=2
        )


def test_hf_rollout_accepts_exactly_growing_cache_and_positions(monkeypatch):
    cache = _FakeCache()
    model = _RolloutModel(cache, grow=True)
    monkeypatch.setattr(ORACLE, "_new_hf_dynamic_cache", lambda _model: cache)

    case, logits = ORACLE._run_hf_case(
        model, _small_fixture(), device="cpu", max_new_tokens=3, top_k=2
    )

    assert tuple(logits.shape) == (3, 4)
    assert [call["input_ids"].shape[1] for call in model.calls] == [3, 1, 1]
    assert [call["attention_mask"].shape[1] for call in model.calls] == [3, 4, 5]
    assert [call["cache_position"].tolist() for call in model.calls] == [
        [0, 1, 2],
        [3],
        [4],
    ]
    assert [call["position_ids"].tolist() for call in model.calls] == [
        [[0, 1, 2]],
        [[3]],
        [[4]],
    ]
    assert [
        step["after"]["sequence_length"] for step in case["cache_audit"]["steps"]
    ] == [
        3,
        4,
        5,
    ]
    assert not ORACLE._validate_serialized_cache_audit(case, max_new_tokens=3)


def test_fixtures_are_stable_and_distinct():
    first = ORACLE.build_fixtures(154856)
    second = ORACLE.build_fixtures(154856)
    assert first == second
    assert [case["prompt_length"] for case in first] == [16, 128]
    assert first[0]["input_sha256"] != first[1]["input_sha256"]
    assert min(first[0]["input_ids"]) >= 1024
    assert max(first[1]["input_ids"]) < 154856 - 1024


def test_compare_accepts_bounded_logprob_delta():
    result = ORACLE.compare_payloads(
        _payload(),
        _payload(delta=0.01, candidate=True),
        atol=0.02,
        rtol=0,
        comparison_top_k=2,
    )
    assert result["pass"]
    assert not result["failures"]


def test_compare_rejects_token_mismatch_and_large_delta():
    token_result = ORACLE.compare_payloads(
        _payload(),
        _payload(token_id=9, candidate=True),
        atol=0.02,
        rtol=0,
        comparison_top_k=2,
    )
    assert not token_result["pass"]
    assert any(
        "generated token mismatch" in failure for failure in token_result["failures"]
    )

    delta_result = ORACLE.compare_payloads(
        _payload(),
        _payload(delta=0.2, candidate=True),
        atol=0.02,
        rtol=0,
        comparison_top_k=2,
    )
    assert not delta_result["pass"]
    assert any("outside" in failure for failure in delta_result["failures"])


def test_compare_skips_logits_after_greedy_prefix_diverges():
    oracle = _rollout_payload([7, 8, 9, 10])
    # Step 1 is the first token mismatch.  Its logits still share the [7]
    # prefix and are compared.  The intentionally huge later deltas must not
    # be interpreted as model errors because those requests consumed [7, 8]
    # and [7, 12], respectively.
    candidate = _rollout_payload(
        [7, 12, 13, 14], [0.0, 0.0, 100.0, 100.0], candidate=True
    )

    result = ORACLE.compare_payloads(
        oracle, candidate, atol=0.02, rtol=0, comparison_top_k=2
    )

    assert not result["pass"]
    assert result["failures"] == [
        "tokens_16: generated token mismatch: [7, 8, 9, 10] != [7, 12, 13, 14]"
    ]
    assert [diagnostic["status"] for diagnostic in result["diagnostics"]] == [
        "compared",
        "compared",
        "skipped_due_to_prefix_divergence",
        "skipped_due_to_prefix_divergence",
    ]
    assert result["diagnostics"][2]["first_divergent_step"] == 1
    assert result["diagnostics"][3]["first_divergent_step"] == 1


def test_compare_checks_all_steps_when_rollout_tokens_match():
    oracle = _rollout_payload([7, 8, 9])
    candidate = _rollout_payload([7, 8, 9], [0.0, 0.0, 0.2], candidate=True)

    result = ORACLE.compare_payloads(
        oracle, candidate, atol=0.02, rtol=0, comparison_top_k=2
    )

    assert not result["pass"]
    assert [diagnostic["status"] for diagnostic in result["diagnostics"]] == [
        "compared",
        "compared",
        "compared",
    ]
    assert any("tokens_16 step 2" in failure for failure in result["failures"])


def test_compare_rejects_wrong_kind_failed_or_mismatched_contract():
    oracle = _payload()
    candidate = _payload(candidate=True)
    candidate["kind"] = "wrong"
    candidate["pass"] = False
    candidate["fixture_seed"] += 1

    result = ORACLE.compare_payloads(
        oracle, candidate, atol=0.02, rtol=0, comparison_top_k=2
    )

    assert not result["pass"]
    assert any("kind" in failure for failure in result["failures"])
    assert any("did not pass" in failure for failure in result["failures"])
    assert any("fixture seed" in failure for failure in result["failures"])


def test_compare_rejects_oracle_without_kda_conv_correction_provenance():
    oracle = _payload()
    del oracle["oracle_corrections"]

    result = ORACLE.compare_payloads(
        oracle,
        _payload(candidate=True),
        atol=0.02,
        rtol=0,
        comparison_top_k=2,
    )

    assert not result["pass"]
    assert "missing oracle_corrections" in result["failures"]
