"""Real transformers-kt config smoke for the pinned GLM-5-Next layer names."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "python/sglang/srt/configs/glm5_next.py"
PINNED_DIST = "transformers-kt"
PINNED_VERSION = "5.6.0.post1"
CHILD_FLAG = "--runtime-config-child"
SKIP_EXIT_CODE = 77
SUCCESS_MARKER = "GLM5_NEXT_RUNTIME_CONFIG_STRICT_OK"


def _load_config_module():
    for name, path in (
        ("sglang", REPO_ROOT / "python/sglang"),
        ("sglang.srt", REPO_ROOT / "python/sglang/srt"),
        ("sglang.srt.configs", REPO_ROOT / "python/sglang/srt/configs"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    mamba_utils = types.ModuleType("sglang.srt.configs.mamba_utils")
    mamba_utils.KimiLinearCacheParams = type("KimiLinearCacheParams", (), {})
    mamba_utils.KimiLinearStateShape = type("KimiLinearStateShape", (), {})
    sys.modules[mamba_utils.__name__] = mamba_utils

    module_name = "sglang.srt.configs.glm5_next"
    spec = importlib.util.spec_from_file_location(module_name, CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load production config: {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _child_main() -> int:
    try:
        version = importlib.metadata.version(PINNED_DIST)
    except importlib.metadata.PackageNotFoundError:
        print(f"skip: {PINNED_DIST} is not installed")
        return SKIP_EXIT_CODE
    if version != PINNED_VERSION:
        print(f"skip: expected {PINNED_DIST}=={PINNED_VERSION}; got {version}")
        return SKIP_EXIT_CODE

    module = _load_config_module()
    checkpoint_layer_types = [
        "linear_attention"
        if layer_idx % 4 != 3
        else "deepseek_sparse_attention"
        for layer_idx in range(45)
    ]
    kda_layers = [
        index
        for index, layer_type in enumerate(checkpoint_layer_types)
        if layer_type == "linear_attention"
    ]
    full_attn_layers = [
        index
        for index, layer_type in enumerate(checkpoint_layer_types)
        if layer_type == "deepseek_sparse_attention"
    ]
    linear_attn_config = {
        "full_attn_layers": full_attn_layers,
        "head_dim": 128,
        "kda_layers": kda_layers,
        "num_heads": 64,
        "short_conv_kernel_size": 4,
    }
    text_config = {
        "model_type": "glm5_next_text",
        "num_hidden_layers": 45,
        "layer_types": checkpoint_layer_types,
        "linear_attn_config": linear_attn_config,
    }
    config = module.Glm5NextConfig.from_dict(
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "num_hidden_layers": 45,
            "layer_types": checkpoint_layer_types,
            "linear_attn_config": linear_attn_config,
            "text_config": text_config,
            "vision_config": {"projection_intermediate_size": 10240},
        }
    )

    generic_layer_types = [
        "linear_attention" if layer_type == "linear_attention" else "sparse"
        for layer_type in checkpoint_layer_types
    ]
    assert config.layer_types == generic_layer_types
    assert config.text_config.layer_types == generic_layer_types
    assert (
        config.text_config._glm5_next_checkpoint_layer_types
        == checkpoint_layer_types
    )
    assert config.text_config.linear_layer_ids == kda_layers
    assert config.text_config.full_attention_layer_ids == full_attn_layers
    assert len(kda_layers) == 34
    assert len(full_attn_layers) == 11

    # The strict validator must remain callable after construction, not merely
    # be bypassed during ``from_dict``.
    config.validate()
    config.text_config.validate()
    print(f"{SUCCESS_MARKER}: transformers-kt={version}; kda=34; dsa=11")
    return 0


def test_real_transformers_kt_accepts_checkpoint_layer_types():
    import pytest

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
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
