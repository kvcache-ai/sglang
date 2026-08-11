"""CPU-only guards for GLM-5-Next's fail-closed checkpoint contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE3_TEST_PATH = REPO_ROOT / "test/registered/unit/test_glm5_next_stage3.py"


def _load_stage3_support():
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_loader_contract_stage3_support",
        STAGE3_TEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLoaderModel:
    def __init__(
        self,
        model_module,
        parameter_names,
        *,
        num_experts=2,
        start_layer=0,
        end_layer=45,
        is_first_rank=True,
        is_last_rank=True,
    ):
        self.config = SimpleNamespace(
            n_routed_experts=num_experts,
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
        )
        self.model = SimpleNamespace(
            start_layer=start_layer,
            end_layer=end_layer,
        )
        self.pp_group = SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=is_last_rank,
        )
        self.packed_modules_mapping = (
            model_module.Glm5NextForConditionalGeneration.packed_modules_mapping
        )
        self._parameters_for_test = {
            name: torch.nn.Parameter(torch.empty(1)) for name in parameter_names
        }
        self.skipped_phase7_visual_weights = ()
        self.skipped_session_ab_mtp_weights = ()
        self.skipped_pipeline_parallel_weight_count = 0
        self.checkpoint_runtime_default_parameters = ()

    def named_parameters(self):
        return self._parameters_for_test.items()

    @staticmethod
    def _load_kda_stacked_weight(name, loaded_weight, params_dict):
        del name, loaded_weight, params_dict
        return False


@pytest.fixture(scope="module")
def model_module():
    support = _load_stage3_support()
    config_module = support._load_config_module()
    return support._load_model_module(config_module)


def _consume(model_module, fake, names, *, require_complete=True):
    weights = [(name, torch.empty(1)) for name in names]
    return list(
        model_module.Glm5NextForConditionalGeneration._normalized_text_weights(
            fake,
            weights,
            require_complete=require_complete,
        )
    )


def test_reverse_contract_covers_packed_experts_fp8_scales_and_runtime_defaults(
    model_module,
):
    parameters = {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight_scale_inv",
        "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv",
        "model.layers.3.mlp.experts.w13_weight",
        "model.layers.3.mlp.experts.w13_weight_scale_inv",
        "model.layers.3.mlp.experts.w2_weight",
        "model.layers.3.mlp.experts.w2_weight_scale_inv",
        "model.layers.3.self_attn.attn_mha.k_scale",
        "model.layers.3.self_attn.attn_mha.v_scale",
        "model.layers.3.self_attn.attn_mqa.k_scale",
        "model.layers.3.self_attn.attn_mqa.v_scale",
        "model.norm.weight",
    }

    expected, runtime_defaults = model_module._glm5_next_checkpoint_source_contract(
        parameters,
        num_experts=2,
        packed_modules_mapping=(
            model_module.Glm5NextForConditionalGeneration.packed_modules_mapping
        ),
    )

    assert len(expected) == 20
    assert len(runtime_defaults) == 4
    assert "model.layers.0.self_attn.q_proj.weight" in expected
    assert "model.layers.0.self_attn.k_proj.weight" in expected
    assert "model.layers.0.self_attn.v_proj.weight" in expected
    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" in expected
    assert "model.layers.0.mlp.up_proj.weight_scale_inv" in expected
    assert "model.layers.3.self_attn.q_a_proj.weight_scale_inv" in expected
    assert "model.layers.3.self_attn.kv_a_proj_with_mqa.weight_scale_inv" in expected
    assert "model.layers.3.mlp.experts.1.up_proj.weight" in expected
    assert "model.layers.3.mlp.experts.1.down_proj.weight_scale_inv" in expected

    fake = _FakeLoaderModel(model_module, parameters)
    outputs = _consume(model_module, fake, sorted(expected))
    assert {name for name, _ in outputs} == expected
    assert set(fake.checkpoint_runtime_default_parameters) == runtime_defaults


def test_missing_direct_and_partial_packed_shards_fail_closed(model_module):
    direct = _FakeLoaderModel(
        model_module,
        {"model.layers.3.self_attn.q_b_proj.weight"},
    )
    with pytest.raises(RuntimeError, match="missing 1 required"):
        _consume(model_module, direct, [])

    packed = _FakeLoaderModel(
        model_module,
        {"model.layers.0.self_attn.qkv_proj.weight"},
    )
    with pytest.raises(RuntimeError, match="v_proj.weight"):
        _consume(
            model_module,
            packed,
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
            ],
        )


def test_missing_one_expert_fp8_scale_shard_fails_closed(model_module):
    fake = _FakeLoaderModel(
        model_module,
        {"model.layers.3.mlp.experts.w13_weight_scale_inv"},
        num_experts=2,
    )
    with pytest.raises(
        RuntimeError,
        match=r"experts\.1\.up_proj\.weight_scale_inv",
    ):
        _consume(
            model_module,
            fake,
            [
                "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
                "model.layers.3.mlp.experts.0.up_proj.weight_scale_inv",
                "model.layers.3.mlp.experts.1.gate_proj.weight_scale_inv",
            ],
        )


def test_unknown_local_text_duplicate_alias_and_layer46_are_rejected(model_module):
    fake = _FakeLoaderModel(
        model_module,
        {"model.layers.3.self_attn.q_b_proj.weight"},
    )
    with pytest.raises(RuntimeError, match="unknown text tensor"):
        _consume(
            model_module,
            fake,
            ["model.layers.3.self_attn.q_typo.weight"],
        )

    top_level = _FakeLoaderModel(model_module, {"model.norm.weight"})
    with pytest.raises(RuntimeError, match="duplicate normalized text tensor"):
        _consume(
            model_module,
            top_level,
            [
                "model.language_model.norm.weight",
                "model.norm.weight",
            ],
        )

    with pytest.raises(RuntimeError, match="unknown text tensor"):
        _consume(
            model_module,
            _FakeLoaderModel(model_module, set()),
            ["model.layers.46.input_layernorm.weight"],
            require_complete=False,
        )


def test_pp_nonowner_exact_mtp45_and_raw_visual_prefixes_are_the_only_skips(
    model_module,
):
    fake = _FakeLoaderModel(
        model_module,
        {"model.layers.20.input_layernorm.weight"},
        start_layer=20,
        end_layer=25,
        is_first_rank=False,
        is_last_rank=False,
    )
    outputs = _consume(
        model_module,
        fake,
        [
            "model.layers.3.input_layernorm.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
            "model.language_model.layers.45.eh_proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "visual.patch_embed.proj.weight",
            "model.language_model.layers.20.input_layernorm.weight",
        ],
    )

    assert [name for name, _ in outputs] == ["model.layers.20.input_layernorm.weight"]
    assert fake.skipped_pipeline_parallel_weight_count == 4
    assert fake.skipped_session_ab_mtp_weights == (
        "model.language_model.layers.45.eh_proj.weight",
    )
    assert fake.skipped_phase7_visual_weights == (
        "model.visual.blocks.0.attn.qkv.weight",
        "visual.patch_embed.proj.weight",
    )


def test_visual_classification_checks_the_raw_prefix_only(model_module):
    normalize = model_module.normalize_glm5_next_weight_name

    assert normalize("model.visual.blocks.0.attn.qkv.weight") is None
    assert normalize("visual.patch_embed.proj.weight") is None
    assert normalize("model.language_model.visual.fake.weight") == (
        "model.visual.fake.weight"
    )
    assert normalize("language_model.visual.fake.weight") == (
        "model.visual.fake.weight"
    )

    with pytest.raises(RuntimeError, match="unknown text tensor"):
        _consume(
            model_module,
            _FakeLoaderModel(model_module, set()),
            ["model.language_model.visual.fake.weight"],
            require_complete=False,
        )


def test_nextn_loader_entry_is_rejected_for_session_ab(model_module):
    fake = _FakeLoaderModel(model_module, set())
    with pytest.raises(RuntimeError, match="MTP/NextN"):
        model_module.Glm5NextForConditionalGeneration.load_weights(
            fake,
            [],
            is_nextn=True,
        )
