"""Resolve new wheel-relative payloads while retaining legacy manifests."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mapped", [False, True])
def test_payload_binary_layout(tmp_path, monkeypatch, mapped):
    source = (
        Path(__file__).resolve().parents[3]
        / "sgl-kernel/python/sgl_kernel/payload_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("payload_layout_test", source)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    name = "sm100/common_ops.abi3.so"
    relative = "sgl_kernel/" + name if mapped else name
    manifest = SimpleNamespace(FILES={relative: "checksum"})
    if mapped:
        manifest.BINARIES = {name: relative}
    monkeypatch.setattr(loader, "_manifest", lambda: manifest)
    monkeypatch.setattr(loader, "_materialize_all", lambda: tmp_path)
    assert loader.materialize_binary(name) == tmp_path / relative
    with pytest.raises(FileNotFoundError, match="Unknown"):
        loader.materialize_binary("unknown.so")
