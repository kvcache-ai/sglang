"""The native Kimi encoder uses Conv2d, not the affected cuDNN Conv3d path."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_glm5_next_pypi_runtime import _compile_server_args_method

ROOT = Path(__file__).resolve().parents[3]


def guard(monkeypatch, *, architecture="KimiK25ForConditionalGeneration", backend="auto", cudnn=91002):
    torch = SimpleNamespace(__version__="2.9.1+cu128", backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: cudnn)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    check = _compile_server_args_method("check_torch_2_9_1_cudnn_compatibility", {
        "get_bool_env_var": lambda _: False, "torch_release": (2, 9, 1),
    })
    config = SimpleNamespace(is_glm5_next=False, is_multimodal=True,
                             hf_config=SimpleNamespace(architectures=[architecture]))
    args = SimpleNamespace(model_impl=backend, get_model_config=lambda: config)
    return lambda: check(args)


@pytest.mark.parametrize("backend", ["auto", "sglang"])
def test_native_kimi_does_not_require_unused_conv3d_dependency(monkeypatch, backend):
    assert guard(monkeypatch, backend=backend)() is None


@pytest.mark.parametrize("architecture,backend", [
    ("Qwen3VLForConditionalGeneration", "auto"),
    ("UnknownVisionModel", "sglang"),
    ("KimiK25ForConditionalGeneration", "transformers"),
])
def test_guard_is_retained_for_other_models_and_backends(monkeypatch, architecture, backend):
    with pytest.raises(RuntimeError, match="CuDNN Compatibility"):
        guard(monkeypatch, architecture=architecture, backend=backend)()


def test_supported_cudnn_still_passes_other_vlm(monkeypatch):
    assert guard(monkeypatch, architecture="OtherVisionModel", cudnn=91600)() is None


def test_native_encoder_contract_is_conv2d_not_conv3d():
    tree = ast.parse((ROOT / "python/sglang/srt/models/kimi_k25.py").read_bytes())
    patch_embed = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoonVision3dPatchEmbed")
    calls = [n.func for n in ast.walk(patch_embed) if isinstance(n, ast.Call)]
    assert any(isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "nn" and f.attr == "Conv2d" for f in calls)
    # Changing the encoder to Conv3d requires revisiting its startup exemption.
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("Conv3d", "conv3d") for n in ast.walk(tree))
