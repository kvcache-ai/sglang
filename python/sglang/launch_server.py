"""Launch the inference server."""

import asyncio
import os
import sys

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
