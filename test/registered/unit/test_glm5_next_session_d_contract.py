"""CPU-only contracts for the GLM-5-Next image/video boundary."""

from __future__ import annotations

import ast
import base64
import binascii
import copy
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSOR_PATH = REPO_ROOT / "python/sglang/srt/multimodal/processors/glm5_next.py"
TOKENIZER_MANAGER_PATH = REPO_ROOT / "python/sglang/srt/managers/tokenizer_manager.py"
MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next.py"
SCHEDULE_BATCH_PATH = REPO_ROOT / "python/sglang/srt/managers/schedule_batch.py"
FORWARD_BATCH_PATH = REPO_ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
HF_UTILS_PATH = REPO_ROOT / "python/sglang/srt/utils/hf_transformers_utils.py"
CHAT_TEMPLATE_PATH = REPO_ROOT / "examples/chat_template/glm5_next_multimodal.jinja"


def _class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return copy.deepcopy(
        next(
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
    )


def _compile_method(path, class_name, method_name, namespace):
    method = _class_method(path, class_name, method_name)
    method.decorator_list = [
        decorator
        for decorator in method.decorator_list
        if not isinstance(decorator, ast.Name) or decorator.id != "staticmethod"
    ]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def test_processor_uses_transformers_kt_as_the_only_patchifier():
    source = PROCESSOR_PATH.read_text(encoding="utf-8")
    assert "transformers.models.glm5_next.image_processing_glm5_next" in source
    assert "transformers.models.glm5_next.video_processing_glm5_next" in source
    assert "class Glm5NextImageProcessor" not in source
    assert "GLM5_NEXT_PATCH_EXPAND_FACTOR = 1" in source
    assert "video_token_id=self.IM_TOKEN_ID" in source


def test_image_pixel_override_is_fail_closed_at_factor_one():
    resolve = _compile_method(
        PROCESSOR_PATH,
        "Glm5NextSGLangProcessor",
        "_resolve_max_pixels",
        {
            "GLM5_NEXT_MIN_PIXELS": 12_544,
            "GLM5_NEXT_DEFAULT_MAX_PIXELS": 6_272_000,
            "GLM5_NEXT_CHECKPOINT_MAX_PIXELS": 6_272_000,
        },
    )
    assert resolve(None) == 6_272_000
    assert resolve({"image": {"max_pixels": 12_544}}) == 12_544
    assert resolve({"image": {"max_pixels": 6_272_000}}) == 6_272_000
    for invalid in (
        [],
        {"video": {}},
        {"image": []},
        {"image": {"min_pixels": 12_544}},
        {"image": {"max_pixels": True}},
        {"image": {"max_pixels": 12_543}},
        {"image": {"max_pixels": 6_272_001}},
    ):
        with pytest.raises(ValueError):
            resolve(invalid)


def test_request_boundary_rejects_mixed_over_limit_and_audio_before_processor():
    validate = _compile_method(
        TOKENIZER_MANAGER_PATH,
        "TokenizerManager",
        "_validate_glm5_next_mm_boundary",
        {},
    )

    class Harness:
        model_config = SimpleNamespace(is_glm5_next=True)

    valid = (
        SimpleNamespace(image_data=[object()] * 8, video_data=None, audio_data=None),
        SimpleNamespace(image_data=None, video_data=[object()], audio_data=None),
    )
    for request in valid:
        validate(Harness(), request)

    invalid = (
        (SimpleNamespace(image_data=[1], video_data=[2], audio_data=None), "mix"),
        (SimpleNamespace(image_data=list(range(9)), video_data=None, audio_data=None), "8 images"),
        (SimpleNamespace(image_data=None, video_data=[1, 2], audio_data=None), "one video"),
        (SimpleNamespace(image_data=None, video_data=None, audio_data=[1]), "audio"),
    )
    for request, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate(Harness(), request)

    non_glm = SimpleNamespace(model_config=SimpleNamespace(is_glm5_next=False))
    validate(
        non_glm,
        SimpleNamespace(image_data=[1], video_data=[2], audio_data=[3]),
    )


def test_video_materialization_is_seekable_bounded_and_cleaned(monkeypatch, tmp_path):
    materialize = _compile_method(
        PROCESSOR_PATH,
        "Glm5NextSGLangProcessor",
        "_materialize_video",
        {
            "base64": base64,
            "binascii": binascii,
            "contextmanager": contextmanager,
            "os": os,
            "tempfile": tempfile,
            "unquote": unquote,
            "urlparse": urlparse,
            "requests": SimpleNamespace(),
        },
    )
    payload = b"\x00\x00\x00\x18ftypmp42test-video"
    with materialize(payload) as path:
        generated = Path(path)
        assert generated.is_file()
        assert generated.read_bytes() == payload
    assert not generated.exists()

    local = tmp_path / "local.mp4"
    local.write_bytes(payload)
    with materialize(f"file://{local}") as path:
        assert path == str(local)
    assert local.exists()

    monkeypatch.setenv("SGLANG_GLM5_NEXT_MAX_VIDEO_BYTES", "4")
    with pytest.raises(ValueError, match="byte limit"):
        with materialize(payload):
            pass


def test_video_validation_and_vision_microbatch_contracts_are_explicit():
    processor_source = PROCESSOR_PATH.read_text(encoding="utf-8")
    model_source = MODEL_PATH.read_text(encoding="utf-8")
    assert "one image-token block per temporal" in processor_source
    assert "timestamped video token expansion changed unexpectedly" in processor_source
    assert "metadata.timestamps[::2]" in processor_source
    assert "isinstance(video_metadata, (list, tuple))" in processor_source
    assert "max_patch_rows = 32_768" in model_source
    assert "frame_grids.extend([(1, grid_h, grid_w)] * grid_t)" in model_source
    assert "def get_video_feature" in model_source


def test_multimodal_hybrid_prefill_marker_survives_chunked_prefill():
    schedule_source = SCHEDULE_BATCH_PATH.read_text(encoding="utf-8")
    forward_source = FORWARD_BATCH_PATH.read_text(encoding="utf-8")
    assert schedule_source.count("glm5_next_force_hybrid_prefill") >= 4
    assert "batch.forward_mode.is_extend()" in forward_source
    assert "batch.multimodal_inputs or []" in forward_source


def test_official_template_covers_media_tools_and_observations():
    template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    for token in (
        "<|begin_of_image|><|image|><|end_of_image|>",
        "<|begin_of_video|><|video|><|end_of_video|>",
        "<tool_call>",
        "<arg_key>",
        "<arg_value>",
        "<tool_response>",
        "<|observation|>",
    ):
        assert token in template


def test_processor_loader_requires_the_new_checkpoint_classes():
    source = HF_UTILS_PATH.read_text(encoding="utf-8")
    assert "Glm5NextProcessor.from_pretrained" in source
    assert "Glm5NextImageProcessor" in source
    assert "Glm5NextVideoProcessor" in source
    assert "Glm46VVideoProcessor()" not in source
