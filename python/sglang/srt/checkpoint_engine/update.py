"""
Usage:
1) Launch the server with wait-for-initial-weights option in one terminal:
   python -m sglang.launch_server --model-path /workspace/Qwen/Qwen3-4B/ --tensor-parallel-size 2 --port 19730 --load-format dummy --checkpoint-engine-wait-weights-before-ready --mem-fraction-static 0.7

2) Torchrun this script in another terminal:
    torchrun --nproc-per-node 2 update.py --update-method broadcast --checkpoint-path /workspace/Qwen/Qwen3-4B/  --inference-parallel-size 2

Or use the integrated entry point:
    python -m sglang.srt.checkpoint_engine.update --update-method broadcast --checkpoint-path /workspace/Qwen/Qwen3-4B/  --inference-parallel-size 2
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal

import httpx
import torch
import torch.distributed as dist
from safetensors import safe_open

try:
    from checkpoint_engine.ps import ParameterServer
    from loguru import logger
except ImportError:
    # Fallback for when checkpoint_engine is not available
    ParameterServer = None
    import logging

    logger = logging.getLogger(__name__)


UPDATE_METHODS = ("broadcast", "p2p", "all")


@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{msg} duration: {end - start:.2f} seconds")


def _build_http_transport(uds: str | None) -> httpx.BaseTransport | None:
    if uds is None:
        return None
    return httpx.HTTPTransport(uds=uds)


def build_torchrun_command(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inference-parallel-size", type=int, default=8)
    parsed_args, _ = parser.parse_known_args(argv)
    return [
        "torchrun",
        f"--nproc-per-node={parsed_args.inference_parallel_size}",
        "-m",
        "sglang.srt.checkpoint_engine.update",
        *argv,
    ]


def validate_args(args: argparse.Namespace, world_size: int | None = None):
    if args.inference_parallel_size <= 0:
        raise ValueError("--inference-parallel-size must be greater than 0")
    if args.checkpoint_path and args.load_metas_file:
        raise ValueError(
            "--checkpoint-path and --load-metas-file cannot be provided together"
        )
    if not args.checkpoint_path and not args.load_metas_file:
        raise ValueError(
            "One of --checkpoint-path or --load-metas-file must be provided"
        )
    if args.load_metas_file and not os.path.isfile(args.load_metas_file):
        raise ValueError(f"Meta file does not exist: {args.load_metas_file}")
    if args.checkpoint_path and not os.path.isdir(args.checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {args.checkpoint_path}")
    if world_size is not None:
        if world_size <= 0:
            raise ValueError("WORLD_SIZE must be greater than 0")
        if world_size % args.inference_parallel_size != 0:
            raise ValueError(
                "WORLD_SIZE must be divisible by --inference-parallel-size for checkpoint groups"
            )


def check_sglang_ready(
    endpoint: str, inference_parallel_size: int, uds: str | None = None
):
    rank = int(os.getenv("RANK", 0))
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    transport = _build_http_transport(uds)
    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                if retry_num % 10 == 0:
                    logger.warning(
                        f"fail to check sglang ready, retry {retry_num} times, error: {e}"
                    )
                retry_num += 1
                time.sleep(0.1)


def split_checkpoint_files(
    checkpoint_path: str, rank: int, world_size: int
) -> list[str]:
    checkpoint_files = [
        os.path.join(checkpoint_path, f)
        for f in sorted(
            filter(lambda x: x.endswith(".safetensors"), os.listdir(checkpoint_path))
        )
    ]
    files_per_rank = (len(checkpoint_files) + world_size - 1) // world_size
    return checkpoint_files[rank * files_per_rank : (rank + 1) * files_per_rank]


def split_tensors(
    checkpoint_path: str, rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    index_fn = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_fn) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    fn_tensors: dict[str, list[str]] = defaultdict(list)
    weight_keys = list(weight_map.items())
    for name, file in weight_keys[
        rank * weights_per_rank : (rank + 1) * weights_per_rank
    ]:
        fn_tensors[file].append(name)
    named_tensors = {}
    for file, names in fn_tensors.items():
        with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
            for name in names:
                named_tensors[name] = f.get_tensor(name)
    return named_tensors


def req_inference(
    endpoint: str,
    inference_parallel_size: int,
    timeout: float = 300.0,
    uds: str | None = None,
    weight_version: str | None = None,
) -> Callable[[list[tuple[str, str]]], None]:
    rank = int(os.getenv("RANK", 0))
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            with httpx.Client(transport=_build_http_transport(uds)) as client:
                resp = client.post(
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()

    return req_func


def update_weights(
    ps,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    save_metas_file: str | None = None,
    update_method: Literal["broadcast", "p2p", "all"] = "broadcast",
    uds: str | None = None,
):
    ps.register_checkpoint(
        checkpoint_name, files=checkpoint_files, named_tensors=named_tensors
    )
    ps.init_process_group()
    check_sglang_ready(endpoint, inference_parallel_size, uds)
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)

    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            # sleep 2s to wait destroy process group
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(
                checkpoint_name, req_func, ranks=list(range(inference_parallel_size))
            )


def join(
    ps: ParameterServer,
    checkpoint_name: str,
    load_metas_file: str,
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    uds: str | None = None,
):
    assert load_metas_file, "load_metas_file is required"
    with open(load_metas_file, "rb") as f:
        metas = pickle.load(f)
    ps.init_process_group()
    check_sglang_ready(endpoint, inference_parallel_size, uds)
    dist.barrier()
    with timer("Gather metas before join"):
        ps.gather_metas(checkpoint_name)
    ps.load_metas(metas)
    with timer(
        f"Update weights with setting ranks as range(0, {inference_parallel_size}) by using p2p"
    ):
        ps.update(checkpoint_name, req_func, ranks=list(range(inference_parallel_size)))


def run_with_torchrun():
    """Run the update script with torchrun automatically."""
    args = sys.argv[1:]
    cmd = build_torchrun_command(args)

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    # Execute torchrun with the original script
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            "Error: torchrun command not found. Please ensure PyTorch is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


def main():
    # Check if we're running under torchrun or need to invoke it
    if os.getenv("RANK") is None:
        # Not running under torchrun, so invoke it
        run_with_torchrun()
        return

    # Running under torchrun, proceed with normal execution
    parser = argparse.ArgumentParser(description="Update weights example")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--save-metas-file", type=str, default=None)
    parser.add_argument("--load-metas-file", type=str, default=None)
    parser.add_argument("--sleep-time", type=int, default=0)
    parser.add_argument("--endpoint", type=str, default="http://localhost:19730")
    parser.add_argument("--inference-parallel-size", type=int, default=8)
    parser.add_argument("--checkpoint-name", type=str, default="my-checkpoint-iter-0")
    parser.add_argument("--update-method", choices=UPDATE_METHODS, default="broadcast")
    parser.add_argument("--uds", type=str, default=None)
    parser.add_argument("--weight-version", type=str, default=None)
    args = parser.parse_args()
    validate_args(args)

    # Get rank and world_size from environment (set by torchrun)
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    validate_args(args, world_size=world_size)

    req_func = req_inference(
        args.endpoint,
        args.inference_parallel_size,
        uds=args.uds,
        weight_version=args.weight_version,
    )

    if ParameterServer is None:
        print("Error: checkpoint_engine package not available", file=sys.stderr)
        sys.exit(1)

    ps = ParameterServer(auto_pg=True)
    ps._p2p_store = None
    if args.load_metas_file:
        join(
            ps,
            args.checkpoint_name,
            args.load_metas_file,
            req_func,
            args.inference_parallel_size,
            args.endpoint,
            args.uds,
        )
    else:
        if args.checkpoint_path and os.path.exists(
            os.path.join(args.checkpoint_path, "model.safetensors.index.json")
        ):
            named_tensors = split_tensors(args.checkpoint_path, rank, world_size)
            checkpoint_files = []
        else:
            checkpoint_files = (
                split_checkpoint_files(args.checkpoint_path, rank, world_size)
                if args.checkpoint_path
                else []
            )
            named_tensors = {}
        update_weights(
            ps,
            args.checkpoint_name,
            checkpoint_files,
            named_tensors,
            req_func,
            args.inference_parallel_size,
            args.endpoint,
            args.save_metas_file,
            args.update_method,
            args.uds,
        )
    time.sleep(args.sleep_time)


if __name__ == "__main__":
    main()
