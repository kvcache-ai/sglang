"""CPU-only contracts for the GLM-5.3-flash PyPI runtime hotfix."""

from __future__ import annotations

import ast
import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_ARGS_PATH = REPO_ROOT / "python/sglang/srt/server_args.py"
GLM4V_PATH = REPO_ROOT / "python/sglang/srt/models/glm4v.py"
PYPROJECT_PATH = REPO_ROOT / "python/pyproject.toml"


def _compile_server_args_method(name: str, globals_: dict):
    tree = ast.parse(
        SERVER_ARGS_PATH.read_text(encoding="utf-8"), filename=str(SERVER_ARGS_PATH)
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    method = copy.deepcopy(
        next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"__builtins__": __builtins__, **globals_}
    exec(compile(module, str(SERVER_ARGS_PATH), "exec"), namespace)
    return namespace[name]


def test_default_grammar_backend_does_not_install_upstream_transformers():
    handle = _compile_server_args_method("_handle_grammar_backend", {})
    args = SimpleNamespace(grammar_backend=None)
    handle(args)
    assert args.grammar_backend == "llguidance"

    dependencies = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    normalized = {
        dependency.lower().split(";")[0].strip() for dependency in dependencies
    }
    assert not any(item.startswith("transformers==") for item in normalized)
    assert not any(item.startswith("xgrammar") for item in normalized)
    assert not any(item.startswith("compressed-tensors") for item in normalized)
    assert any(item.startswith("transformers-kt==5.6.0.post4") for item in normalized)


def test_glm5_next_bypasses_only_the_unused_cudnn_conv3d_guard():
    check = _compile_server_args_method(
        "check_torch_2_9_1_cudnn_compatibility",
        {
            "get_bool_env_var": lambda _name: False,
            "torch_release": (2, 9, 1),
        },
    )

    class Harness:
        @staticmethod
        def get_model_config():
            return SimpleNamespace(is_glm5_next=True, is_multimodal=True)

    # This returns before importing torch or inspecting cuDNN.
    assert check(Harness()) is None


def test_glm_vision_patch_projection_uses_the_conv_weights_without_cudnn():
    source = GLM4V_PATH.read_text(encoding="utf-8")
    assert "self.proj.weight.flatten(start_dim=1)" in source
    assert "return F.linear(" in source
    assert "x = self.proj(x).view(-1, self.hidden_size)" not in source
