"""Materialize large KT CUDA extensions from checksummed payload wheels."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest():
    try:
        from sgl_kernel import _payload_manifest
    except ImportError as error:
        raise FileNotFoundError(
            "The sgl-kernel-kt CUDA payload wheels are not installed."
        ) from error
    return _payload_manifest


def _payload_part(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.submodule_search_locations is None:
        raise FileNotFoundError(f"Missing KT CUDA payload package: {module_name}")
    return Path(next(iter(spec.submodule_search_locations))) / "payload.part"


def _cache_root(version: str, archive_sha256: str) -> Path:
    override = os.environ.get("SGL_KERNEL_KT_CACHE_DIR")
    if override:
        candidates = [Path(override).expanduser()]
    else:
        # Prefer the environment that was large enough to install the wheel.
        # System site-packages can be read-only, so retain user-cache and /tmp
        # fallbacks.  Materialization peaks below 4 GB for the current payload.
        candidates = [
            Path(__file__).resolve().parent / "_payload_cache",
            Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")),
            Path(tempfile.gettempdir()),
        ]

    failures = []
    for base in candidates:
        root = base / "sgl-kernel-kt" / version / archive_sha256[:16]
        try:
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < 4_000_000_000:
                failures.append(f"{root}: insufficient free space")
                continue
            probe = root / f".write-probe-{os.getpid()}"
            probe.touch(exist_ok=False)
            probe.unlink()
            return root
        except OSError as error:
            failures.append(f"{root}: {error}")
    raise OSError("No writable sgl-kernel-kt payload cache: " + "; ".join(failures))


def _all_valid(root: Path, files: dict[str, str]) -> bool:
    return all((root / name).is_file() and _sha256(root / name) == digest for name, digest in files.items())


def _materialize_all() -> Path:
    manifest = _manifest()
    root = _cache_root(manifest.VERSION, manifest.ARCHIVE_SHA256)
    if _all_valid(root, manifest.FILES):
        return root

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".materialize.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _all_valid(root, manifest.FILES):
            return root

        with tempfile.TemporaryDirectory(prefix="materialize-", dir=root) as temp_name:
            temp = Path(temp_name)
            archive = temp / "payload.tar.gz"
            digest = hashlib.sha256()
            with archive.open("wb") as output:
                for module_name in manifest.PAYLOAD_MODULES:
                    part = _payload_part(module_name)
                    with part.open("rb") as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            output.write(block)
                            digest.update(block)
            if digest.hexdigest() != manifest.ARCHIVE_SHA256:
                raise RuntimeError("sgl-kernel-kt CUDA payload archive checksum mismatch")

            extract_root = temp / "extract"
            extract_root.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                members = {member.name: member for member in bundle.getmembers()}
                for name, expected in manifest.FILES.items():
                    member = members.get(name)
                    if member is None or not member.isfile():
                        raise RuntimeError(f"Missing {name} in sgl-kernel-kt CUDA payload")
                    destination = extract_root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Cannot extract {name} from CUDA payload")
                    with source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                    if _sha256(destination) != expected:
                        raise RuntimeError(f"Checksum mismatch for {name}")

            for name in manifest.FILES:
                source = extract_root / name
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)

        if not _all_valid(root, manifest.FILES):
            raise RuntimeError("sgl-kernel-kt CUDA payload verification failed")
    return root


def materialize_binary(name: str) -> Path:
    manifest = _manifest()
    if name not in manifest.FILES:
        raise FileNotFoundError(f"Unknown sgl-kernel-kt payload binary: {name}")
    return _materialize_all() / name
