"""Launch the inference server."""

import asyncio
import os
import sys


def _sweep_stale_torch_extension_locks():
    """Remove stale ninja locks under ~/.cache/torch_extensions before any
    torch.utils.cpp_extension build runs.

    Background: torch's cpp_extension uses ninja to build C++ / CUDA modules
    JIT. Ninja takes a `lock` / `.ninja_lock` file in the build dir and
    blocks while it's held. If a previous sglang run was killed mid-build
    (SIGKILL, OOM, scheduler crash), the lock survives on disk. Subsequent
    runs hang forever waiting on the orphaned lock with zero CPU/GPU
    activity — appearing identical to a deadlock.

    Sweeping locks older than `SGLANG_STALE_LOCK_AGE_MINUTES` (default 30
    minutes; long enough to never interrupt a real build, short enough to
    auto-recover same-day reruns) eliminates this class of hang at startup.
    Origin: sglang 本身 (bare-metal kt_1 deployment recovery).
    """
    try:
        import time

        cache_dir = os.path.expanduser(
            os.environ.get("TORCH_EXTENSIONS_DIR", "~/.cache/torch_extensions")
        )
        if not os.path.isdir(cache_dir):
            return
        max_age_min = int(os.environ.get("SGLANG_STALE_LOCK_AGE_MINUTES", "30"))
        if max_age_min <= 0:
            return
        cutoff = time.time() - max_age_min * 60
        swept = 0
        for root, _dirs, files in os.walk(cache_dir):
            for name in files:
                if name not in ("lock", ".ninja_lock"):
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.unlink(path)
                        swept += 1
                except OSError:
                    pass
        if swept:
            print(
                f"[sglang] swept {swept} stale ninja locks under {cache_dir} "
                f"(older than {max_age_min}m)",
                file=sys.stderr,
            )
    except Exception:
        # Best-effort cleanup. Failures here must not block startup.
        pass


_sweep_stale_torch_extension_locks()

# Auto-inject CUDA arch list for flashinfer / torch JIT before sglang imports.
# flashinfer's fp4 modules and torch C++ extensions JIT-compile cubins at
# import time using these env vars; on CUDA 12.8 with consumer Blackwell
# (SM_120) the toolchain default arch list is empty, causing
# `check_cuda_arch` in flashinfer to falsely report "sm75 or higher".
# `setdefault` preserves any explicit value the user set (escape hatch for
# multi-arch builds, etc.). Origin: sglang 本身.
try:
    import torch as _torch_for_arch_inject  # noqa: F401

    if _torch_for_arch_inject.cuda.is_available():
        _cap = _torch_for_arch_inject.cuda.get_device_capability()
        _arch = f"{_cap[0]}.{_cap[1]}"
        # `a` (architecture-specific) variants only exist for Hopper
        # (SM_90a) and Blackwell (SM_100a / SM_103a / SM_120a). Ada (SM_89)
        # and Ampere (SM_8x) have only generic SASS / PTX.
        _arch_letter = "a" if _cap[0] in (9, 10, 11, 12) else ""
        os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", f"{_arch}{_arch_letter}")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_arch}+PTX")
    del _torch_for_arch_inject
except Exception:
    pass

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.encoder_only:
        from sglang.srt.disaggregation.encode_server import launch_server

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
