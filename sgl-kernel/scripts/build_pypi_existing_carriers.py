#!/usr/bin/env python3
"""Carry the KT SGL runtime entirely in existing PyPI projects.

``ktransformers[sglang]`` always installs kt-kernel, sglang-kt, and
transformers-kt.  This builder distributes the compressed CUDA archive across
those existing projects and places the remaining ``sgl_kernel`` runtime in
kt-kernel.  No shard-only or sgl-kernel-kt project needs to be created.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


KT_KERNEL_VERSION = "0.7.0.post2"
SGL_KERNEL_VERSION = "0.3.21.post2"
PLATFORM_TAG = "manylinux_2_35_x86_64"
# PyPI's default limit is 100 MiB. Keep a small metadata/upload margin.
MAX_PYPI_WHEEL_SIZE = 104_000_000
LARGE_BINARIES = (
    "sgl_kernel/flash_ops.abi3.so",
    "sgl_kernel/sm100/common_ops.abi3.so",
)

# Sizes account for the existing contents of each carrier. The final carrier
# receives the remainder of the deterministic gzip archive.
CARRIERS = (
    ("kt_kernel", "sgl_kernel_kt_payload_core", 70_000_000),
    ("transformers", "transformers_kt_sgl_kernel_payload", 90_000_000),
    ("sglang", "sglang_kt_sgl_kernel_payload", 96_000_000),
    ("ktransformers", "ktransformers_sgl_kernel_payload", None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256=" + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode()


def write_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != record:
            rows.append(
                (path.relative_to(root).as_posix(), record_hash(path), str(path.stat().st_size))
            )
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def pack_wheel(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                wheel.write(path, path.relative_to(root).as_posix())
    if output.stat().st_size >= MAX_PYPI_WHEEL_SIZE:
        raise RuntimeError(f"PyPI carrier is too large: {output} ({output.stat().st_size} bytes)")


def make_archive(unpacked: Path, archive: Path) -> dict[str, str]:
    hashes = {}
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as bundle:
                for wheel_name in LARGE_BINARIES:
                    source = unpacked / wheel_name
                    archive_name = wheel_name.removeprefix("sgl_kernel/")
                    hashes[archive_name] = sha256(source)
                    info = bundle.gettarinfo(str(source), arcname=archive_name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        bundle.addfile(info, stream)
    return hashes


def split_archive(archive: Path, temp: Path) -> dict[str, Path]:
    parts = {}
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for carrier, module, size in CARRIERS:
            block = stream.read() if size is None else stream.read(size)
            if not block:
                raise RuntimeError(f"Empty payload part for {carrier}")
            part = temp / f"{module}.part"
            part.write_bytes(block)
            digest.update(block)
            parts[carrier] = part
        if stream.read(1):
            raise RuntimeError("Payload split did not consume the complete archive")
    if digest.hexdigest() != sha256(archive):
        raise RuntimeError("Payload split changed the CUDA archive")
    return parts


def add_payload(root: Path, module: str, part: Path) -> None:
    package = root / module
    package.mkdir()
    (package / "__init__.py").write_text("# Binary payload for the KT SGL runtime.\n")
    shutil.copy2(part, package / "payload.part")


def retag_wheel_metadata(wheel_metadata: Path, python_tag: str, abi_tag: str) -> None:
    lines = []
    for line in wheel_metadata.read_text().splitlines():
        if not line.strip() or line.startswith("Tag:"):
            continue
        if line.startswith("Root-Is-Purelib:"):
            line = "Root-Is-Purelib: false"
        lines.append(line)
    lines.append(f"Tag: {python_tag}-{abi_tag}-{PLATFORM_TAG}")
    wheel_metadata.write_text("\n".join(lines) + "\n")


def platform_filename(input_wheel: Path, *, version: str | None = None) -> str:
    components = input_wheel.stem.split("-")
    if len(components) < 5:
        raise RuntimeError(f"Unexpected wheel filename: {input_wheel.name}")
    if version is not None:
        components[1] = version
    return "-".join((*components[:-3], "py3", "none", PLATFORM_TAG)) + ".whl"


def build_python_carrier(
    input_wheel: Path,
    module: str,
    part: Path,
    output_dir: Path,
    replacements: tuple[tuple[str, str], ...] = (),
    overlays: tuple[tuple[Path, str], ...] = (),
) -> Path:
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        with zipfile.ZipFile(input_wheel) as wheel:
            wheel.extractall(root)
        dist_info = next(root.glob("*.dist-info"))
        add_payload(root, module, part)
        for source, relative_destination in overlays:
            destination = root / relative_destination
            if not destination.is_file():
                raise RuntimeError(f"Missing carrier overlay destination: {destination}")
            shutil.copy2(source, destination)
        metadata = dist_info / "METADATA"
        text = metadata.read_text()
        for old, new in replacements:
            if old not in text:
                raise RuntimeError(f"Missing carrier metadata dependency: {old}")
            text = text.replace(old, new, 1)
        metadata.write_text(text)
        retag_wheel_metadata(dist_info / "WHEEL", "py3", "none")
        write_record(root, dist_info)
        output = output_dir / platform_filename(input_wheel)
        pack_wheel(root, output)
    return output


def prepare_sgl_runtime(
    root: Path,
    module: str,
    part: Path,
    archive_sha256: str,
    file_hashes: dict[str, str],
) -> None:
    for wheel_name in LARGE_BINARIES:
        (root / wheel_name).unlink()
    shutil.rmtree(root / "sgl_kernel/sm90", ignore_errors=True)

    source_root = Path(__file__).resolve().parents[1] / "python/sgl_kernel"
    for name in ("load_utils.py", "flash_attn.py", "payload_runtime.py"):
        shutil.copy2(source_root / name, root / "sgl_kernel" / name)
    add_payload(root, module, part)
    manifest = root / "sgl_kernel/_payload_manifest.py"
    manifest.write_text(
        f'VERSION = "{SGL_KERNEL_VERSION}"\n'
        f'ARCHIVE_SHA256 = "{archive_sha256}"\n'
        f"PAYLOAD_MODULES = {tuple(item[1] for item in CARRIERS)!r}\n"
        f"FILES = {file_hashes!r}\n"
    )
    version_py = root / "sgl_kernel/version.py"
    if version_py.exists():
        version_py.write_text(version_py.read_text().replace("0.3.21.post1", SGL_KERNEL_VERSION))


def merge_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        if destination.exists():
            raise RuntimeError(f"Carrier path collision: {destination}")
        shutil.copy2(source, destination)


def build_kt_kernel_carrier(
    kt_kernel_wheel: Path,
    sgl_kernel_wheel: Path,
    module: str,
    part: Path,
    archive_sha256: str,
    file_hashes: dict[str, str],
    output_dir: Path,
) -> Path:
    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        root = temp / "kt"
        sgl_root = temp / "sgl"
        with zipfile.ZipFile(kt_kernel_wheel) as wheel:
            wheel.extractall(root)
        with zipfile.ZipFile(sgl_kernel_wheel) as wheel:
            wheel.extractall(sgl_root)
        prepare_sgl_runtime(
            sgl_root, module, part, archive_sha256, file_hashes
        )

        for source in sorted(sgl_root.iterdir()):
            if source.name.endswith(".dist-info"):
                continue
            merge_tree(source, root / source.name)

        old_dist_info = next(root.glob("kt_kernel-*.dist-info"))
        new_dist_info = root / f"kt_kernel-{KT_KERNEL_VERSION}.dist-info"
        old_dist_info.rename(new_dist_info)
        metadata = new_dist_info / "METADATA"
        text = metadata.read_text()
        old_version = next(
            line.removeprefix("Version: ")
            for line in text.splitlines()
            if line.startswith("Version: ")
        )
        text = text.replace(f"Version: {old_version}", f"Version: {KT_KERNEL_VERSION}", 1)
        metadata.write_text(text)
        retag_wheel_metadata(new_dist_info / "WHEEL", "cp312", "cp312")
        write_record(root, new_dist_info)
        output = output_dir / (
            f"kt_kernel-{KT_KERNEL_VERSION}-cp312-cp312-{PLATFORM_TAG}.whl"
        )
        pack_wheel(root, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgl-kernel-wheel", required=True, type=Path)
    parser.add_argument("--kt-kernel-wheel", required=True, type=Path)
    parser.add_argument("--transformers-wheel", required=True, type=Path)
    parser.add_argument("--sglang-wheel", required=True, type=Path)
    parser.add_argument("--ktransformers-wheel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        unpacked = temp / "sgl-input"
        with zipfile.ZipFile(args.sgl_kernel_wheel) as wheel:
            wheel.extractall(unpacked)
        archive = temp / "payload.tar.gz"
        file_hashes = make_archive(unpacked, archive)
        archive_hash = sha256(archive)
        parts = split_archive(archive, temp)
        modules = {carrier: module for carrier, module, _ in CARRIERS}

        outputs = [
            build_kt_kernel_carrier(
                args.kt_kernel_wheel,
                args.sgl_kernel_wheel,
                modules["kt_kernel"],
                parts["kt_kernel"],
                archive_hash,
                file_hashes,
                args.output_dir,
            ),
            build_python_carrier(
                args.transformers_wheel,
                modules["transformers"],
                parts["transformers"],
                args.output_dir,
            ),
            build_python_carrier(
                args.sglang_wheel,
                modules["sglang"],
                parts["sglang"],
                args.output_dir,
                (("sgl-kernel-kt==0.3.21.post2", "kt-kernel==0.7.0.post2"),),
                (
                    (
                        Path(__file__).resolve().parents[2]
                        / "python/sglang/srt/entrypoints/engine.py",
                        "sglang/srt/entrypoints/engine.py",
                    ),
                ),
            ),
            build_python_carrier(
                args.ktransformers_wheel,
                modules["ktransformers"],
                parts["ktransformers"],
                args.output_dir,
                (("kt-kernel==0.7.0.post1", "kt-kernel==0.7.0.post2"),),
            ),
        ]

    for wheel in outputs:
        print(f"{sha256(wheel)}  {wheel.name}  {wheel.stat().st_size}")


if __name__ == "__main__":
    main()
