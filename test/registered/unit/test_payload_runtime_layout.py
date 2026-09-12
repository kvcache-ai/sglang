"""CPU-only loader checks, including an actual ELF $ORIGIN dependency."""

import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def loader():
    path = (
        Path(__file__).resolve().parents[3]
        / "sgl-kernel/python/sgl_kernel/payload_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("_payload_runtime_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_fixture(loader, tmp_path, monkeypatch, files, binaries=None):
    part = tmp_path / "payload.part"
    with tarfile.open(part, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    manifest = SimpleNamespace(
        VERSION="test",
        ARCHIVE_SHA256=hashlib.sha256(part.read_bytes()).hexdigest(),
        PAYLOAD_MODULES=("test_payload",),
        FILES={
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        },
    )
    if binaries is not None:
        manifest.BINARIES = binaries
    cache = tmp_path / "cache"
    monkeypatch.setattr(loader, "_manifest", lambda: manifest)
    monkeypatch.setattr(loader, "_cache_root", lambda *args: cache)
    monkeypatch.setattr(loader, "_payload_part", lambda name: part)
    return cache, manifest


@pytest.mark.parametrize("mapped", [False, True])
def test_legacy_and_wheel_relative_payloads(loader, tmp_path, monkeypatch, mapped):
    name = "sm100/common_ops.abi3.so"
    relative = "sgl_kernel/" + name if mapped else name
    cache, _ = install_fixture(
        loader,
        tmp_path,
        monkeypatch,
        {relative: b"fixture"},
        {name: relative} if mapped else None,
    )
    binary = loader.materialize_binary(name)
    assert binary == cache / relative
    assert binary.read_bytes() == b"fixture"
    assert loader.materialize_binary(name) == binary
    with pytest.raises(FileNotFoundError, match="Unknown"):
        loader.materialize_binary("not-present.so")


def test_bad_payload_checksum_fails(loader, tmp_path, monkeypatch):
    _, manifest = install_fixture(
        loader, tmp_path, monkeypatch, {"flash_ops.abi3.so": b"fixture"}
    )
    manifest.ARCHIVE_SHA256 = "0" * 64
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        loader.materialize_binary("flash_ops.abi3.so")


def test_materialized_elf_resolves_sibling_library(loader, tmp_path, monkeypatch):
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("ELF linkage probe requires gcc")
    library = tmp_path / "libkt_payload_probe.so"
    binary = tmp_path / "common_ops.abi3.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-Wl,-soname,libkt_payload_probe.so",
            "-o",
            str(library),
        ],
        input="int payload_probe(void) { return 42; }\n",
        text=True,
        check=True,
    )
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-x",
            "c",
            "-",
            "-L" + str(tmp_path),
            "-lkt_payload_probe",
            "-Wl,-rpath,$ORIGIN/../../sgl_kernel_kt.libs",
            "-o",
            str(binary),
        ],
        input="extern int payload_probe(void); int result(void) { return payload_probe(); }\n",
        text=True,
        check=True,
    )
    name = "sm100/common_ops.abi3.so"
    files = {
        "sgl_kernel/" + name: binary.read_bytes(),
        "sgl_kernel_kt.libs/" + library.name: library.read_bytes(),
    }
    cache, _ = install_fixture(
        loader, tmp_path, monkeypatch, files, {name: "sgl_kernel/" + name}
    )
    probe = "import ctypes, sys; print(ctypes.CDLL(sys.argv[1]).result())"
    loaded = subprocess.check_output(
        [sys.executable, "-c", probe, str(loader.materialize_binary(name))], text=True
    )
    assert loaded.strip() == "42"
    flat = tmp_path / "legacy-flat"
    (flat / "sm100").mkdir(parents=True)
    (flat / "sgl_kernel_kt.libs").mkdir()
    shutil.copyfile(binary, flat / name)
    shutil.copyfile(library, flat / "sgl_kernel_kt.libs" / library.name)
    broken = subprocess.run(
        [sys.executable, "-c", probe, str(flat / name)], capture_output=True, text=True
    )
    assert broken.returncode != 0
    assert library.name in broken.stderr
    # Corrupted dependencies must be re-extracted on the next materialization.
    (cache / "sgl_kernel_kt.libs" / library.name).write_bytes(b"corrupted")
    loader.materialize_binary(name)
    assert (cache / "sgl_kernel_kt.libs" / library.name).read_bytes() == files[
        "sgl_kernel_kt.libs/" + library.name
    ]
