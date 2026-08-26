#!/usr/bin/env python3
"""Split an already validated KT kernel wheel into PyPI-sized payload wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


VERSION = "0.3.21.post2"
CHUNK_SIZE = 88_000_000
LARGE_BINARIES = ("sgl_kernel/flash_ops.abi3.so", "sgl_kernel/sm100/common_ops.abi3.so")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def write_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != record:
            rows.append((path.relative_to(root).as_posix(), record_hash(path), str(path.stat().st_size)))
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def pack_wheel(root: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                wheel.write(path, path.relative_to(root).as_posix())


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


def payload_wheel(part: Path, index: int, output_dir: Path) -> tuple[str, str]:
    dist = f"sgl_kernel_kt_payload_{index}"
    project = f"sgl-kernel-kt-payload-{index}"
    module = dist
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        package = root / module
        package.mkdir()
        (package / "__init__.py").write_text("# Binary payload for sgl-kernel-kt.\n")
        shutil.copy2(part, package / "payload.part")
        dist_info = root / f"{dist}-{VERSION}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {project}\nVersion: {VERSION}\n"
            "Summary: Binary payload for sgl-kernel-kt\nRequires-Python: >=3.10\n\n"
        )
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: kt-split-wheel\nRoot-Is-Purelib: false\n"
            "Tag: py3-none-manylinux_2_35_x86_64\n"
        )
        write_record(root, dist_info)
        output = output_dir / f"{dist}-{VERSION}-py3-none-manylinux_2_35_x86_64.whl"
        pack_wheel(root, output)
    return project, module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wheel", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_name:
        temp = Path(temp_name)
        unpacked = temp / "base"
        with zipfile.ZipFile(args.input_wheel) as wheel:
            wheel.extractall(unpacked)

        archive = temp / "payload.tar.gz"
        file_hashes = make_archive(unpacked, archive)
        archive_hash = sha256(archive)

        parts = []
        with archive.open("rb") as stream:
            index = 1
            while block := stream.read(CHUNK_SIZE):
                part = temp / f"payload-{index}.part"
                part.write_bytes(block)
                parts.append(part)
                index += 1

        payloads = [payload_wheel(part, index, args.output_dir) for index, part in enumerate(parts, 1)]

        for wheel_name in LARGE_BINARIES:
            (unpacked / wheel_name).unlink()
        shutil.rmtree(unpacked / "sgl_kernel/sm90", ignore_errors=True)

        source_root = Path(__file__).resolve().parents[1] / "python/sgl_kernel"
        for name in ("load_utils.py", "flash_attn.py", "payload_runtime.py"):
            shutil.copy2(source_root / name, unpacked / "sgl_kernel" / name)

        manifest = unpacked / "sgl_kernel/_payload_manifest.py"
        manifest.write_text(
            f'VERSION = "{VERSION}"\n'
            f'ARCHIVE_SHA256 = "{archive_hash}"\n'
            f"PAYLOAD_MODULES = {tuple(module for _, module in payloads)!r}\n"
            f"FILES = {file_hashes!r}\n"
        )

        old_dist_info = next(unpacked.glob("sgl_kernel_kt-*.dist-info"))
        new_dist_info = unpacked / f"sgl_kernel_kt-{VERSION}.dist-info"
        old_dist_info.rename(new_dist_info)
        metadata = new_dist_info / "METADATA"
        text = metadata.read_text()
        text = text.replace("Version: 0.3.21.post1", f"Version: {VERSION}", 1)
        marker = text.find("\n\n")
        requirements = "".join(
            f"Requires-Dist: {project}=={VERSION}\n" for project, _ in payloads
        )
        text = text[: marker + 1] + requirements + text[marker + 1 :]
        metadata.write_text(text)
        version_py = unpacked / "sgl_kernel/version.py"
        if version_py.exists():
            version_py.write_text(version_py.read_text().replace("0.3.21.post1", VERSION))
        wheel_metadata = new_dist_info / "WHEEL"
        wheel_metadata.write_text(
            wheel_metadata.read_text().replace(
                "cp310-abi3-linux_x86_64", "cp310-abi3-manylinux_2_35_x86_64"
            )
        )
        write_record(unpacked, new_dist_info)
        base_output = (
            args.output_dir
            / f"sgl_kernel_kt-{VERSION}-cp310-abi3-manylinux_2_35_x86_64.whl"
        )
        pack_wheel(unpacked, base_output)

    for wheel in sorted(args.output_dir.glob("*.whl")):
        print(f"{sha256(wheel)}  {wheel.name}  {wheel.stat().st_size}")


if __name__ == "__main__":
    main()
