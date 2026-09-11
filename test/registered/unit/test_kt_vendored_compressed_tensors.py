"""Offline source/metadata checks; GPU and real import checks run on release hosts."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "python/sglang/_vendor/compressed_tensors"


def test_vendored_files_match_recorded_hashes():
    lock = json.loads((VENDOR / "vendor-lock.json").read_text())
    assert lock["version"] == "0.15.0.1"
    assert lock["namespace"] == "sglang._vendor.compressed_tensors"
    actual = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    assert actual == set(lock["files"]) | {"LICENSE", "vendor-lock.json"}
    for name, hashes in lock["files"].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == hashes["vendored_sha256"]
    assert "Apache License" in (VENDOR / "LICENSE").read_text()


def test_runtime_does_not_import_the_upstream_distribution():
    for path in (ROOT / "python/sglang").rglob("*.py"):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not any(n == "compressed_tensors" or n.startswith("compressed_tensors.") for n in names), path


def test_resolver_uses_kt_and_vendor_runtime_dependencies():
    config = tomllib.loads((ROOT / "python/pyproject.toml").read_text())
    deps = config["project"]["dependencies"]
    assert "transformers-kt==5.6.0.post5" in deps
    assert "accelerate-kt==1.14.0.post3" in deps
    assert "pydantic>=2.0" in deps
    assert "loguru>=0.7,<1" in deps
    assert not any(d == "transformers" or d.startswith(("transformers=", "transformers>", "compressed-tensors")) for d in deps)
    assert "_vendor/**/*" in config["tool"]["setuptools"]["package-data"]["sglang"]


@pytest.mark.skipif(not os.environ.get("KT_COMPRESSED_TENSORS_WHEEL"), reason="Set locked wheel path to reproduce vendoring offline")
def test_locked_wheel_reproduces_the_entire_vendor_tree():
    spec = importlib.util.spec_from_file_location("vendor_recipe", ROOT / "scripts/vendor_compressed_tensors.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.sync(Path(os.environ["KT_COMPRESSED_TENSORS_WHEEL"]), VENDOR, check=True)
