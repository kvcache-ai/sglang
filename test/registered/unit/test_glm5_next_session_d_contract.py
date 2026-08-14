"""CPU-only contracts for the frozen GLM-5-Next Session D boundary."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE3_TEST_PATH = REPO_ROOT / "test/registered/unit/test_glm5_next_stage3.py"
MODEL_CONFIG_PATH = REPO_ROOT / "python/sglang/srt/configs/model_config.py"
PROCESSOR_PATH = REPO_ROOT / "python/sglang/srt/multimodal/processors/glm5_next.py"
GLM_OCR_MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm_ocr.py"
GLM5_MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next.py"


def _load_stage3_support():
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_session_d_stage3_support", STAGE3_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _model_config_tree() -> ast.Module:
    return ast.parse(
        MODEL_CONFIG_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_CONFIG_PATH),
    )


def _literal_assignment(tree: ast.Module, name: str):
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )
    return ast.literal_eval(node.value)


def _resolve_model_config_mm(architecture, requested, *, language_only=False):
    """Execute the three MM-resolution statements copied from ModelConfig."""

    tree = _model_config_tree()
    model_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelConfig"
    )
    init = next(
        node
        for node in model_config.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    default_resolution = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "enable_multimodal is None"
    )
    effective_assignment = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "is_multimodal"
            for target in node.targets
        )
    )
    exact_glm_gate = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.is_glm5_next"
        and "_glm5_next_multimodal_active" in ast.unparse(node)
    )

    multimodal_architectures = set(_literal_assignment(tree, "multimodal_model_archs"))
    holder = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[architecture], model_type="test-model"
        ),
        is_glm5_next=architecture == "Glm5NextForConditionalGeneration",
    )

    class _Logger:
        @staticmethod
        def info(*args, **kwargs):
            del args, kwargs

    namespace = {
        "self": holder,
        "enable_multimodal": requested,
        "language_only": language_only,
        "logger": _Logger(),
        "is_multimodal_model": lambda architectures: bool(
            set(architectures) & multimodal_architectures
        ),
    }
    selected = ast.fix_missing_locations(
        ast.Module(
            body=[
                copy.deepcopy(default_resolution),
                copy.deepcopy(effective_assignment),
                copy.deepcopy(exact_glm_gate),
            ],
            type_ignores=[],
        )
    )
    exec(compile(selected, str(MODEL_CONFIG_PATH), "exec"), namespace)
    return namespace["enable_multimodal"], holder


def _compile_max_pixels_resolver():
    tree = ast.parse(
        PROCESSOR_PATH.read_text(encoding="utf-8"), filename=str(PROCESSOR_PATH)
    )
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextSGLangProcessor"
    )
    method = copy.deepcopy(
        next(
            node
            for node in processor.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_max_pixels"
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "GLM5_NEXT_MIN_PIXELS": 12_544,
        "GLM5_NEXT_DEFAULT_MAX_PIXELS": 1_254_400,
        "GLM5_NEXT_CHECKPOINT_MAX_PIXELS": 9_633_792,
    }
    exec(compile(module, str(PROCESSOR_PATH), "exec"), namespace)
    return namespace["_resolve_max_pixels"]


def _compile_checkpoint_image_config_parser():
    tree = ast.parse(
        PROCESSOR_PATH.read_text(encoding="utf-8"), filename=str(PROCESSOR_PATH)
    )
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextImageProcessor"
    )
    method = copy.deepcopy(
        next(
            node
            for node in processor.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "from_checkpoint_config"
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    expected_values = {
        "do_rescale": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "GlmgaImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "merge_size": 2,
        "patch_size": 14,
        "patch_expand_factor": 2,
        "temporal_patch_size": 2,
    }
    namespace = {
        "Any": object,
        "GLM5_NEXT_MIN_PIXELS": 12_544,
        "GLM5_NEXT_DEFAULT_MAX_PIXELS": 1_254_400,
        "GLM5_NEXT_CHECKPOINT_MAX_PIXELS": 9_633_792,
        "_GLM5_NEXT_IMAGE_CONFIG_KEYS": frozenset({*expected_values, "size"}),
        "_GLM5_NEXT_IMAGE_CONFIG_VALUES": expected_values,
    }
    exec(compile(module, str(PROCESSOR_PATH), "exec"), namespace)
    return namespace["from_checkpoint_config"], expected_values


def test_exact_checkpoint_token_ids_and_vision_defaults():
    config_module = _load_stage3_support()._load_config_module()
    config = config_module.Glm5NextConfig()

    assert (
        config.image_token_id,
        config.video_token_id,
        config.image_start_token_id,
        config.image_end_token_id,
        config.video_start_token_id,
        config.video_end_token_id,
    ) == (154854, 154855, 154830, 154831, 154832, 154833)

    expected_vision = {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 4096,
        "projection_intermediate_size": 10240,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "image_size": 448,
        "initializer_range": 0.02,
        "attention_dropout": 0.0,
        "attention_bias": True,
    }
    assert {
        key: getattr(config.vision_config, key) for key in expected_vision
    } == expected_vision

    overridden = config_module.Glm5NextConfig(
        vision_config={"depth": 12, "future_checkpoint_field": 17}
    )
    assert overridden.vision_config.depth == 12
    assert overridden.vision_config.future_checkpoint_field == 17


def test_multimodal_default_explicit_off_and_non_glm_isolation():
    effective, glm = _resolve_model_config_mm("Glm5NextForConditionalGeneration", None)
    assert effective is True
    assert glm.is_multimodal is True
    assert glm.hf_config._glm5_next_multimodal_active is True

    effective, glm = _resolve_model_config_mm("Glm5NextForConditionalGeneration", False)
    assert effective is False
    assert glm.is_multimodal is False
    assert glm.hf_config._glm5_next_multimodal_active is False

    _, language_stage = _resolve_model_config_mm(
        "Glm5NextForConditionalGeneration", True, language_only=True
    )
    assert language_stage.is_multimodal is True
    assert language_stage.hf_config._glm5_next_multimodal_active is False

    comparisons = {
        "Qwen3ForCausalLM": (True, False),
        "Qwen3VLMoeForConditionalGeneration": (True, True),
        "Glm4vForConditionalGeneration": (True, True),
        "Gemma3ForConditionalGeneration": (False, False),
    }
    for architecture, expected in comparisons.items():
        effective, holder = _resolve_model_config_mm(architecture, None)
        assert (effective, holder.is_multimodal) == expected
        assert not hasattr(holder.hf_config, "_glm5_next_multimodal_active")


def test_glm5_registry_is_exact_and_no_longer_default_disabled():
    tree = _model_config_tree()
    architectures = _literal_assignment(tree, "multimodal_model_archs")
    assert architectures.count("Glm5NextForConditionalGeneration") == 1

    source = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    init = next(
        node
        for node in next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelConfig"
        ).body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    disabled_branch = next(
        node
        for node in init.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "enable_multimodal is None"
    )
    assert "Glm5NextForConditionalGeneration" not in ast.unparse(disabled_branch)
    assert source.count('"Glm5NextForConditionalGeneration"') >= 1


def test_pixel_cap_contract_is_fail_closed():
    resolve = _compile_max_pixels_resolver()
    assert resolve(None) == 1_254_400
    assert resolve({}) == 1_254_400
    assert resolve({"image": {"max_pixels": 9_633_792}}) == 9_633_792
    assert resolve({"image": {"max_pixels": 12_544}}) == 12_544

    invalid = (
        [],
        {"video": {}},
        {"image": []},
        {"image": {"min_pixels": 12_544}},
        {"image": {"max_pixels": True}},
        {"image": {"max_pixels": 12_543}},
        {"image": {"max_pixels": 9_633_793}},
    )
    for value in invalid:
        with pytest.raises(ValueError):
            resolve(value)


def test_pinned_checkpoint_image_metadata_is_exact_and_complete():
    parse, expected_values = _compile_checkpoint_image_config_parser()

    class _Constructed:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    exact = {
        **expected_values,
        "size": {"shortest_edge": 12_544, "longest_edge": 9_633_792},
    }
    parsed = parse(_Constructed, exact)
    assert parsed.kwargs["size"] == {
        "shortest_edge": 12_544,
        "longest_edge": 1_254_400,
    }
    assert parsed.kwargs["patch_expand_factor"] == 2
    assert "image_processor_type" not in parsed.kwargs

    malformed = []
    missing = dict(exact)
    missing.pop("patch_expand_factor")
    malformed.append(missing)
    unknown = dict(exact)
    unknown["future_field"] = 1
    malformed.append(unknown)
    wrong_type = dict(exact)
    wrong_type["image_processor_type"] = "Glm46VImageProcessor"
    malformed.append(wrong_type)
    wrong_size = dict(exact)
    wrong_size["size"] = {"shortest_edge": 12_544}
    malformed.append(wrong_size)

    for value in malformed:
        with pytest.raises(ValueError):
            parse(_Constructed, value)


def test_processor_and_shared_glm_ocr_changes_remain_isolated():
    processor_source = PROCESSOR_PATH.read_text(encoding="utf-8")
    processor_tree = ast.parse(processor_source, filename=str(PROCESSOR_PATH))
    processor = next(
        node
        for node in processor_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextSGLangProcessor"
    )
    process_method = next(
        node
        for node in processor.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "process_mm_data_async"
    )
    load_calls = [
        node
        for node in ast.walk(process_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_mm_data"
    ]
    assert len(load_calls) == 1
    assert not any(
        isinstance(node, ast.Await) and node.value is load_calls[0]
        for node in ast.walk(process_method)
    )
    return_node = next(
        node for node in ast.walk(process_method) if isinstance(node, ast.Return)
    )
    assert isinstance(return_node.value, ast.Dict)
    return_source = ast.unparse(return_node.value)
    assert "self.IMAGE_START_TOKEN_ID" in return_source
    assert "self.IMAGE_END_TOKEN_ID" in return_source

    glm_ocr_source = GLM_OCR_MODEL_PATH.read_text(encoding="utf-8")
    glm5_source = GLM5_MODEL_PATH.read_text(encoding="utf-8")
    assert "num_dummy_heads=num_dummy_heads" in glm_ocr_source
    assert 'getattr(vision_config, "num_dummy_heads", 0)' not in glm_ocr_source
    assert "if merger_context_dim is None:" in glm_ocr_source
    assert "num_dummy_heads=getattr(self.vision_config" in glm5_source
    assert "merger_context_dim=self.vision_config.projection_intermediate_size" in (
        glm5_source
    )
