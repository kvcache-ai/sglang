"""Pinned-wheel CPU smoke for the GLM-5-Next Session D processor.

The production processor imports the full SGLang model and multimodal stack.
This test intentionally runs in a child process and replaces only those SGLang
imports with small type stubs.  The Transformers classes, the production
``glm5_next.py`` file, and the production ``get_processor`` implementation are
executed unchanged.  Keeping the harness in a subprocess prevents its module
stubs and temporary AutoConfig registrations from leaking into other tests.
"""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PYTHON = REPO_ROOT / "python"
PROCESSOR_PATH = REPO_PYTHON / "sglang/srt/multimodal/processors/glm5_next.py"
HF_UTILS_PATH = REPO_PYTHON / "sglang/srt/utils/hf_transformers_utils.py"

PINNED_TRANSFORMERS_DIST = "transformers-kt"
PINNED_TRANSFORMERS_VERSION = "5.6.0.post1"
CHILD_FLAG = "--processor-smoke-child"
SKIP_EXIT_CODE = 77
SUCCESS_MARKER = "GLM5_SESSION_D_PROCESSOR_SMOKE_OK"

CHECKPOINT_IMAGE_CONFIG = {
    "do_rescale": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_processor_type": "GlmgaImageProcessor",
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "merge_size": 2,
    "patch_size": 14,
    "patch_expand_factor": 2,
    "size": {"shortest_edge": 12_544, "longest_edge": 9_633_792},
    "temporal_patch_size": 2,
}

CHECKPOINT_VIDEO_CONFIG = {
    "do_rescale": True,
    "fps": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "merge_size": 2,
    "patch_expand_factor": 2,
    "patch_size": 14,
    "size": {"shortest_edge": 12_544, "longest_edge": 100_352_000},
    "temporal_patch_size": 2,
    "video_processor_type": "Glm5NextVideoProcessor",
}

TOKEN_STRINGS = (
    "<unk>",
    "<|image|>",
    "<|video|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "describe",
)


def _skip_child(reason: str) -> int:
    print(f"GLM5_SESSION_D_PROCESSOR_SMOKE_SKIP: {reason}")
    return SKIP_EXIT_CODE


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _package(name: str, path: Path | None = None):
    module = _module(name)
    module.__path__ = [] if path is None else [str(path)]
    return module


def _load_production_processor():
    """Load the production file without importing top-level ``sglang``."""

    _package("sglang", REPO_PYTHON / "sglang")
    _package("sglang.srt", REPO_PYTHON / "sglang/srt")
    for package_name in (
        "sglang.srt.layers",
        "sglang.srt.models",
        "sglang.srt.multimodal",
        "sglang.srt.multimodal.processors",
    ):
        _package(package_name)

    class _Stub:
        pass

    _module(
        "sglang.srt.layers.rotary_embedding",
        MRotaryEmbedding=_Stub,
    )
    _module(
        "sglang.srt.models.glm5_next",
        Glm5NextForConditionalGeneration=_Stub,
    )
    _module(
        "sglang.srt.multimodal.processors.base_processor",
        MultimodalSpecialTokens=_Stub,
    )
    _module(
        "sglang.srt.multimodal.processors.glm4v",
        Glm4vImageProcessor=_Stub,
    )

    module_name = "sglang.srt.multimodal.processors.glm5_next"
    spec = importlib.util.spec_from_file_location(module_name, PROCESSOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load production processor: {PROCESSOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocabulary = {token: index for index, token in enumerate(TOKEN_STRINGS)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=list(TOKEN_STRINGS[1:7]),
    )


def _assert_raises(error_type, callback, message_fragment: str):
    try:
        callback()
    except error_type as error:
        if message_fragment not in str(error):
            raise AssertionError(
                f"Expected {message_fragment!r} in {error_type.__name__}: {error}"
            ) from error
    else:
        raise AssertionError(f"Expected {error_type.__name__} to be raised")


def _assert_image_output(output, tokenizer, torch):
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert output.image_grid_thw.tolist() == [[1, 8, 20]]
    assert tuple(output.pixel_values.shape) == (160, 1176)
    assert output.pixel_values.dtype == torch.float32
    assert bool(output.pixel_values.isfinite().all())
    assert int((output.input_ids == image_token_id).sum().item()) == 40
    assert int((output.mm_token_type_ids == 1).sum().item()) == 40


def _assert_multi_image_output(output, tokenizer, torch, image_count):
    image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert tuple(output.image_grid_thw.shape) == (image_count, 3)
    patch_counts = [int(grid.prod().item()) for grid in output.image_grid_thw]
    expected_tokens = sum(patch_count // 4 for patch_count in patch_counts)
    assert tuple(output.pixel_values.shape) == (sum(patch_counts), 1176)
    assert output.pixel_values.dtype == torch.float32
    assert bool(output.pixel_values.isfinite().all())
    assert int((output.input_ids == image_token_id).sum().item()) == expected_tokens
    assert int((output.mm_token_type_ids == 1).sum().item()) == expected_tokens


def _run_image_and_processor_parity(glm5_module, dependencies):
    np, torch, Image, hf_image_module, Glm46VVideoProcessor = dependencies

    image_processor = glm5_module.Glm5NextImageProcessor.from_checkpoint_config(
        CHECKPOINT_IMAGE_CONFIG
    )
    assert image_processor.patch_expand_factor == 2
    assert image_processor.size.shortest_edge == 12_544
    assert image_processor.size.longest_edge == 1_254_400
    assert image_processor.get_number_of_image_patches(123, 257) == 160
    assert glm5_module.smart_resize(
        num_frames=2,
        height=123,
        width=257,
        temporal_factor=2,
        factor=56,
        min_pixels=12_544,
        max_pixels=1_254_400,
    ) == (112, 280)

    tokenizer = _make_tokenizer()
    processor = glm5_module.Glm5NextProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Glm46VVideoProcessor(),
    )
    image = Image.fromarray(
        np.arange(123 * 257 * 3, dtype=np.uint8).reshape(123, 257, 3),
        "RGB",
    )
    output = processor(
        images=[image],
        text=["<|image|> describe"],
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )
    _assert_image_output(output, tokenizer, torch)
    second_image = Image.fromarray(
        np.arange(211 * 97 * 3, dtype=np.uint8).reshape(211, 97, 3),
        "RGB",
    )
    multi_output = processor(
        images=[image, second_image],
        text=["<|image|> describe <|image|>"],
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )
    _assert_multi_image_output(multi_output, tokenizer, torch, 2)

    reference_config = copy.deepcopy(CHECKPOINT_IMAGE_CONFIG)
    reference_config.pop("image_processor_type")
    reference_config.pop("patch_expand_factor")
    reference_config["size"] = {
        "shortest_edge": 12_544,
        "longest_edge": 1_254_400,
    }
    reference = hf_image_module.Glm46VImageProcessor(**reference_config)
    original_smart_resize = hf_image_module.smart_resize

    def _expanded_smart_resize(
        num_frames,
        height,
        width,
        temporal_factor=2,
        factor=28,
        min_pixels=112 * 112,
        max_pixels=14 * 14 * 2 * 2 * 2 * 6144,
    ):
        return original_smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_factor,
            factor=factor * 2,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    hf_image_module.smart_resize = _expanded_smart_resize
    try:
        reference_output = reference(images=[image], return_tensors="pt")
        reference_multi_output = reference(
            images=[image, second_image], return_tensors="pt"
        )
    finally:
        hf_image_module.smart_resize = original_smart_resize

    assert torch.equal(output.image_grid_thw, reference_output.image_grid_thw)
    assert torch.equal(output.pixel_values, reference_output.pixel_values)
    assert torch.equal(
        multi_output.image_grid_thw, reference_multi_output.image_grid_thw
    )
    assert torch.equal(multi_output.pixel_values, reference_multi_output.pixel_values)

    _assert_raises(
        ValueError,
        lambda: processor(
            images=[image],
            text=["<|image|> describe"],
            videos=[image],
        ),
        "does not support video input",
    )
    return [image, second_image], tokenizer, output


def _install_hf_utils_stubs(PretrainedConfig):
    class _EnvironmentValue:
        @staticmethod
        def get():
            return "none"

    _module(
        "sglang.srt.environ",
        envs=SimpleNamespace(SGLANG_APPLY_CONFIG_BACKUP=_EnvironmentValue()),
    )
    utils_module = _module(
        "sglang.srt.utils",
        get_bool_env_var=lambda _name: False,
        is_remote_url=lambda _path: False,
        logger=logging.getLogger("glm5-processor-smoke"),
        lru_cache_frozenset=lambda maxsize=128: lambda function: function,
        mistral_utils=SimpleNamespace(),
    )
    utils_module.__path__ = [str(REPO_PYTHON / "sglang/srt/utils")]

    config_names = (
        "AfmoeConfig",
        "BailingHybridConfig",
        "ChatGLMConfig",
        "DbrxConfig",
        "DeepseekVL2Config",
        "DotsOCRConfig",
        "DotsVLMConfig",
        "ExaoneConfig",
        "FalconH1Config",
        "Glm5NextConfig",
        "Glm5NextTextConfig",
        "GraniteMoeHybridConfig",
        "JetNemotronConfig",
        "JetVLMConfig",
        "KimiK25Config",
        "KimiLinearConfig",
        "KimiVLConfig",
        "LongcatFlashConfig",
        "MultiModalityConfig",
        "NemotronH_Nano_VL_V2_Config",
        "NemotronHConfig",
        "Olmo3Config",
        "Qwen3_5Config",
        "Qwen3_5MoeConfig",
        "Qwen3NextConfig",
        "Step3p5Config",
        "Step3VLConfig",
    )
    config_module = _package("sglang.srt.configs")
    for index, name in enumerate(config_names):
        setattr(
            config_module,
            name,
            type(
                name,
                (PretrainedConfig,),
                {"model_type": f"_glm5_processor_smoke_{index}"},
            ),
        )

    specialized_configs = (
        (
            "sglang.srt.configs.deepseek_ocr",
            "DeepseekVLV2Config",
            "_glm5_processor_smoke_deepseek_ocr",
        ),
        (
            "sglang.srt.configs.deepseek_v4",
            "DeepSeekV4Config",
            "_glm5_processor_smoke_deepseek_v4",
        ),
        (
            "sglang.srt.configs.internvl",
            "InternVLChatConfig",
            "_glm5_processor_smoke_internvl",
        ),
    )
    for module_name, class_name, model_type in specialized_configs:
        config_class = type(
            class_name,
            (PretrainedConfig,),
            {"model_type": model_type},
        )
        _module(module_name, **{class_name: config_class})

    _module(
        "sglang.srt.connector",
        create_remote_connector=lambda _path: None,
    )
    _module(
        "sglang.srt.multimodal.customized_mm_processor_utils",
        _CUSTOMIZED_MM_PROCESSOR={},
    )


def _load_production_hf_utils(PretrainedConfig):
    _install_hf_utils_stubs(PretrainedConfig)
    module_name = "sglang.srt.utils.hf_transformers_utils"
    spec = importlib.util.spec_from_file_location(module_name, HF_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load production HF helpers: {HF_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # This smoke targets the exact get_processor branch. Config parsing has
    # separate Session D coverage and would otherwise require importing all of
    # SGLang's production config dependencies in this isolated child process.
    module.AutoConfig = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: SimpleNamespace(
            model_type="glm5_next"
        )
    )
    return module


def _write_processor_config(path: Path, processor_config: dict):
    (path / "processor_config.json").write_text(
        json.dumps(processor_config), encoding="utf-8"
    )


def _run_get_processor_smoke(
    glm5_module,
    images,
    tokenizer,
    torch,
    PretrainedConfig,
    Glm46VImageProcessor,
    Glm46VProcessor,
    Glm46VVideoProcessor,
):
    with tempfile.TemporaryDirectory(prefix="glm5-session-d-processor-") as tmp:
        fixture_path = Path(tmp)
        seed_processor = Glm46VProcessor(
            image_processor=Glm46VImageProcessor(),
            tokenizer=tokenizer,
            video_processor=Glm46VVideoProcessor(),
        )
        seed_processor.save_pretrained(fixture_path)

        processor_config = json.loads(
            (fixture_path / "processor_config.json").read_text(encoding="utf-8")
        )
        processor_config["processor_class"] = "Glm46VProcessor"
        processor_config["image_processor"] = copy.deepcopy(CHECKPOINT_IMAGE_CONFIG)
        processor_config["video_processor"] = copy.deepcopy(CHECKPOINT_VIDEO_CONFIG)
        _write_processor_config(fixture_path, processor_config)

        _assert_raises(
            ValueError,
            lambda: Glm46VProcessor.from_pretrained(
                fixture_path, local_files_only=True
            ),
            "Unrecognized image processor",
        )

        hf_utils = _load_production_hf_utils(PretrainedConfig)

        wrong_class = copy.deepcopy(processor_config)
        wrong_class["processor_class"] = "UnexpectedProcessor"
        _write_processor_config(fixture_path, wrong_class)
        _assert_raises(
            ValueError,
            lambda: hf_utils.get_processor(
                str(fixture_path), local_files_only=True, revision="main"
            ),
            "expected processor_class='Glm46VProcessor'",
        )

        wrong_video = copy.deepcopy(processor_config)
        wrong_video["video_processor"]["video_processor_type"] = (
            "UnexpectedVideoProcessor"
        )
        _write_processor_config(fixture_path, wrong_video)
        _assert_raises(
            ValueError,
            lambda: hf_utils.get_processor(
                str(fixture_path), local_files_only=True, revision="main"
            ),
            "must retain its Glm5NextVideoProcessor declaration",
        )

        _write_processor_config(fixture_path, processor_config)
        processor = hf_utils.get_processor(
            str(fixture_path), local_files_only=True, revision="main"
        )
        assert isinstance(processor, glm5_module.Glm5NextProcessor)
        assert isinstance(processor.image_processor, glm5_module.Glm5NextImageProcessor)
        assert isinstance(processor.video_processor, Glm46VVideoProcessor)
        assert processor.image_processor.size.shortest_edge == 12_544
        assert processor.image_processor.size.longest_edge == 1_254_400
        assert processor.image_processor.patch_expand_factor == 2

        output = processor(
            images=[images[0]],
            text=["<|image|> describe"],
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        _assert_image_output(output, processor.tokenizer, torch)
        multi_output = processor(
            images=images,
            text=["<|image|> describe <|image|>"],
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
        _assert_multi_image_output(multi_output, processor.tokenizer, torch, 2)
        _assert_raises(
            ValueError,
            lambda: processor(
                images=[images[0]],
                text=["<|image|> describe"],
                videos=[images[0]],
            ),
            "does not support video input",
        )


def _child_main() -> int:
    try:
        installed_version = importlib.metadata.version(PINNED_TRANSFORMERS_DIST)
    except importlib.metadata.PackageNotFoundError:
        return _skip_child(f"{PINNED_TRANSFORMERS_DIST} is not installed")
    if installed_version != PINNED_TRANSFORMERS_VERSION:
        return _skip_child(
            f"requires {PINNED_TRANSFORMERS_DIST}=={PINNED_TRANSFORMERS_VERSION}; "
            f"found {installed_version}"
        )

    try:
        import numpy as np
        import torch
        import torchvision  # noqa: F401
        from PIL import Image
        from transformers import PretrainedConfig
        from transformers.models.glm46v import (
            image_processing_glm46v as hf_image_module,
        )
        from transformers.models.glm46v.image_processing_glm46v import (
            Glm46VImageProcessor,
        )
        from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
        from transformers.models.glm46v.video_processing_glm46v import (
            Glm46VVideoProcessor,
        )
    except (ImportError, ModuleNotFoundError, OSError) as error:
        return _skip_child(f"optional CPU processor dependency unavailable: {error}")

    glm5_module = _load_production_processor()
    dependencies = (
        np,
        torch,
        Image,
        hf_image_module,
        Glm46VVideoProcessor,
    )
    images, tokenizer, _output = _run_image_and_processor_parity(
        glm5_module, dependencies
    )
    _run_get_processor_smoke(
        glm5_module,
        images,
        tokenizer,
        torch,
        PretrainedConfig,
        Glm46VImageProcessor,
        Glm46VProcessor,
        Glm46VVideoProcessor,
    )
    print(
        f"{SUCCESS_MARKER}: transformers-kt={installed_version}; "
        "single_and_multi_image_parity=pass; resize=112x280; "
        "grid=[1,8,20]; patches=160; tokens=40"
    )
    return 0


def test_pinned_wheel_processor_smoke_in_isolated_subprocess():
    import pytest

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    details = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )
    if result.returncode == SKIP_EXIT_CODE:
        pytest.skip(details)
    assert result.returncode == 0, details
    assert SUCCESS_MARKER in result.stdout, details


if __name__ == "__main__":
    if sys.argv[1:] != [CHILD_FLAG]:
        raise SystemExit(f"usage: {Path(__file__).name} {CHILD_FLAG}")
    raise SystemExit(_child_main())
