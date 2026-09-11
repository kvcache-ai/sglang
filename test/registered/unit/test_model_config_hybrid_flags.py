"""Hybrid KV-pool flags must be defined even when hybrid memory is disabled."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


def derive(architecture, *, disabled=False):
    # Execute the actual dependency-light config methods without importing CUDA.
    source = ROOT / "python/sglang/srt/configs/model_config.py"
    tree = ast.parse(source.read_bytes())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelConfig"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_derive_hybrid_model"
    )
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("is_hybrid_swa_model", "get_hybrid_layer_ids")
    ]
    module = ast.Module(body=helpers + [method], type_ignores=[])
    ns = {
        "List": List,
        "PretrainedConfig": object,
        "logger": logging.getLogger(__name__),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
        hf_text_config=SimpleNamespace(
            num_hidden_layers=4, hybrid_layer_pattern=[1, 0, 1, 0]
        ),
        disable_hybrid_swa_memory=disabled,
    )
    ns["_derive_hybrid_model"](config)
    return config


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiK25ForConditionalGeneration",
        "DeepseekV3ForCausalLM",
        "Glm5NextForCausalLM",
        "Qwen3MoeForCausalLM",
    ],
)
def test_non_hybrid_models_have_explicit_false_flags(architecture):
    config = derive(architecture)
    assert config.is_hybrid_swa is False
    assert config.is_swa_with_compressed_attention is False
    assert config.is_hybrid_swa_compress is False


@pytest.mark.parametrize(
    "architecture", ["DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN"]
)
def test_deepseek_compressed_pool_flags_preserved(architecture):
    config = derive(architecture)
    assert config.is_hybrid_swa is True
    assert config.is_swa_with_compressed_attention is True
    assert config.is_hybrid_swa_compress is False


@pytest.mark.parametrize("architecture", ["MiMoV2FlashForCausalLM", "MiMoV2MTP"])
def test_mimo_compressed_swa_flags_preserved(architecture):
    config = derive(architecture)
    assert config.is_hybrid_swa is True
    assert config.is_swa_with_compressed_attention is False
    assert config.is_hybrid_swa_compress is True
    assert config.swa_attention_layer_ids == (
        [0, 2] if architecture == "MiMoV2FlashForCausalLM" else [0]
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "DeepseekV4ForCausalLM",
        "MiMoV2FlashForCausalLM",
        "KimiK25ForConditionalGeneration",
    ],
)
def test_disabled_hybrid_memory_has_false_flags(architecture):
    config = derive(architecture, disabled=True)
    assert config.is_hybrid_swa is False
    assert config.is_swa_with_compressed_attention is False
    assert config.is_hybrid_swa_compress is False


def test_ordinary_hybrid_layer_mapping_preserved():
    config = derive("Llama4ForConditionalGeneration")
    assert config.is_hybrid_swa is True
    assert config.is_swa_with_compressed_attention is False
    assert config.is_hybrid_swa_compress is False
    assert config.swa_attention_layer_ids == [0, 1, 2]
    assert config.full_attention_layer_ids == [3]


@pytest.fixture(scope="module")
def init_pool_capacity():
    # Execute the real capacity-selection prefix, stopping before GPU allocation.
    # Like derive(), this keeps the regression test independent of CUDA imports.
    source = ROOT / "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
    tree = ast.parse(source.read_bytes())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerKVCacheMixin"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "init_memory_pool"
    )
    first_branch = next(
        i for i, node in enumerate(method.body) if isinstance(node, ast.If)
    )
    method.body = method.body[: first_branch + 1]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "profile_max_num_token"
        for node in ast.walk(method)
    ), "The prefix must include the real KV capacity selection"
    method.args.args[0].annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["init_memory_pool"]


def capacity_runner(config, max_total_tokens):
    return SimpleNamespace(
        model_config=config,
        server_args=SimpleNamespace(
            max_running_requests=4, max_total_tokens=max_total_tokens
        ),
        profile_max_num_token=Mock(return_value=8192),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiK25ForConditionalGeneration",
        "DeepseekV3ForCausalLM",
        "Glm5NextForCausalLM",
        "Qwen3MoeForCausalLM",
    ],
)
@pytest.mark.parametrize("max_total_tokens", [None, 4096])
@pytest.mark.parametrize("legacy_missing_flag", [False, True])
def test_non_hybrid_kv_capacity_uses_profile(
    init_pool_capacity, architecture, max_total_tokens, legacy_missing_flag
):
    config = derive(architecture)
    if legacy_missing_flag:
        # Before the default-flag fix, non-SWA initialization returned here.
        # The entry-point guard must work independently of that later fix.
        del config.is_swa_with_compressed_attention
    runner = capacity_runner(config, max_total_tokens)
    init_pool_capacity(runner)
    assert runner.max_total_num_tokens == 8192
    runner.profile_max_num_token.assert_called_once_with()


@pytest.mark.parametrize(
    "architecture", ["DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN"]
)
@pytest.mark.parametrize("max_total_tokens", [None, 4096])
def test_dsv4_kv_capacity_keeps_configured_ceiling(
    init_pool_capacity, architecture, max_total_tokens
):
    runner = capacity_runner(derive(architecture), max_total_tokens)
    init_pool_capacity(runner)
    if max_total_tokens is None:
        assert runner.max_total_num_tokens == 8192
        runner.profile_max_num_token.assert_called_once_with()
    else:
        assert runner.max_total_num_tokens == max_total_tokens
        runner.profile_max_num_token.assert_not_called()


@pytest.mark.parametrize(
    "architecture",
    ["MiMoV2FlashForCausalLM", "MiMoV2MTP", "Llama4ForConditionalGeneration"],
)
@pytest.mark.parametrize("max_total_tokens", [None, 4096])
def test_other_hybrid_kv_capacity_uses_profile(
    init_pool_capacity, architecture, max_total_tokens
):
    runner = capacity_runner(derive(architecture), max_total_tokens)
    init_pool_capacity(runner)
    assert runner.max_total_num_tokens == 8192
    runner.profile_max_num_token.assert_called_once_with()


@pytest.mark.parametrize(
    "architecture", ["DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN"]
)
@pytest.mark.parametrize("max_total_tokens", [None, 4096])
def test_disabled_hybrid_kv_capacity_uses_profile(
    init_pool_capacity, architecture, max_total_tokens
):
    config = derive(architecture, disabled=True)
    del config.is_swa_with_compressed_attention
    runner = capacity_runner(config, max_total_tokens)
    init_pool_capacity(runner)
    assert runner.max_total_num_tokens == 8192
    runner.profile_max_num_token.assert_called_once_with()
