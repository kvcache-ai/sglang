"""Reproduce the private compressed-tensors copy from a SHA256-locked wheel.

Download the wheel URL from vendor-lock.json, then run this script with --wheel.
Only import relocation and logger namespace isolation are applied. No package
is installed, no dependency resolver is invoked, and no top-level distribution
metadata is copied into SGLang. --check verifies an existing vendored tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

VERSION = "0.15.0.1"
WHEEL_SHA256 = "e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9"
WHEEL_URL = "https://files.pythonhosted.org/packages/a8/52/93833dc1610e017ac5b7dcd59b8304d8ef67d1114c2d124e728a2cbbea12/compressed_tensors-0.15.0.1-py3-none-any.whl"
PRIVATE = "sglang._vendor.compressed_tensors"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relocate(source: str, relative: str) -> str:
    result = re.sub(r"(?m)^(\s*from )compressed_tensors(?=[.\s])", rf"\g<1>{PRIVATE}", source)
    result = re.sub(r"(?m)^(\s*)import compressed_tensors\s*$", r"\1from sglang._vendor import compressed_tensors", result)
    if relative == "logger.py":
        # Loguru filters by the actual Python module name, not the old namespace.
        for method in ("enable", "disable"):
            result = result.replace(f'logger.{method}("compressed_tensors")', f'logger.{method}("{PRIVATE}")')
    for node in ast.walk(ast.parse(result)):
        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if any(name == "compressed_tensors" or name.startswith("compressed_tensors.") for name in names):
            raise ValueError(f"Unrelocated upstream import in {relative}")
    return result


def expected_files(wheel: Path) -> dict[str, bytes]:
    if digest(wheel.read_bytes()) != WHEEL_SHA256:
        raise ValueError("Upstream wheel SHA256 mismatch")
    output = {}
    provenance = {}
    with zipfile.ZipFile(wheel) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate wheel paths")
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unsafe upstream path")
            if name.startswith("compressed_tensors/") and not name.endswith("/"):
                relative = name.removeprefix("compressed_tensors/")
                data = archive.read(name)
                if relative.endswith(".py"):
                    rewritten = relocate(data.decode("utf-8"), relative).encode("utf-8")
                elif relative in ("py.typed", "transform/utils/hadamards.safetensors"):
                    rewritten = data
                else:
                    raise ValueError(f"Unexpected upstream runtime file: {name}")
                output[relative] = rewritten
                provenance[relative] = {"upstream_sha256": digest(data), "vendored_sha256": digest(rewritten)}
        output["LICENSE"] = archive.read(f"compressed_tensors-{VERSION}.dist-info/licenses/LICENSE")
    lock = {
        "version": VERSION, "upstream_wheel_url": WHEEL_URL,
        "upstream_wheel_sha256": WHEEL_SHA256, "namespace": PRIVATE,
        "changes": ["Relocate absolute internal imports into the private namespace", "Use the private module name in Loguru enable/disable filters"],
        "dependencies": {"torch": "==2.9.1", "transformers": "provided by transformers-kt", "pydantic": ">=2.0", "loguru": ">=0.7,<1"},
        "files": provenance,
    }
    output["vendor-lock.json"] = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    return output


def sync(wheel: Path, target: Path, check: bool = False) -> None:
    files = expected_files(wheel)
    existing = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if existing - files.keys():
        raise ValueError("Unexpected files in vendor tree; refusing to remove them")
    if check:
        if existing != files.keys() or any((target / name).read_bytes() != data for name, data in files.items()):
            raise ValueError("Vendor tree differs from the locked upstream and deterministic transforms")
    else:
        for name, data in files.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(f"{'Verified' if check else 'Generated'} {len(files)} files from compressed-tensors {VERSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sync(args.wheel, Path(__file__).resolve().parents[1] / "python/sglang/_vendor/compressed_tensors", args.check)
