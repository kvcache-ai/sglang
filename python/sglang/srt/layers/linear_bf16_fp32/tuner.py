
from __future__ import annotations

import datetime as _dt
import functools
import json
import logging
from pathlib import Path
from typing import Annotated, Callable, Dict, List, Optional, Tuple

import torch
import triton.testing
import typer

from sglang.jit_kernel.deepseek_v4 import _dispatch_bf16_fp32_backend

logger = logging.getLogger(__name__)

DEFAULT_M_BUCKETS: Tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
)

_BACKEND_NAMES: Tuple[str, ...] = ("cublas", "deep_gemm")
_BACKENDS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    name: functools.partial(_dispatch_bf16_fp32_backend, algo=name)
    for name in _BACKEND_NAMES
}


def main(
    nk_pairs: Annotated[
        str,
        typer.Option(help="Space-separated N,K pairs, e.g. '4096,7168 256,7168'"),
    ],
    output: Annotated[
        Path,
        typer.Option(
            help="Output JSON path, typically configs/device_name=<DEVICE>.json"
        ),
    ],
    m_buckets: Annotated[
        str,
        typer.Option(help="Comma-separated M values to sweep"),
    ] = ",".join(str(m) for m in DEFAULT_M_BUCKETS),
    rep_ms: Annotated[
        int,
        typer.Option(
            help="triton do_bench_cudagraph `rep` in ms (timing budget per backend/shape)"
        ),
    ] = 50,
) -> None:
    assert torch.cuda.is_available(), "CUDA device required"

    device_name = torch.cuda.get_device_name(0)
    m_list = [int(s) for s in m_buckets.split(",") if s.strip()]
    nk_list: List[Tuple[int, int]] = []
    for pair in nk_pairs.split():
        n_str, k_str = pair.split(",")
        nk_list.append((int(n_str), int(k_str)))

    logger.info(
        "tuning on device=%s  m_buckets=%s  nk_pairs=%s  rep_ms=%d",
        device_name,
        m_list,
        nk_list,
        rep_ms,
    )

    entries: Dict[str, Dict] = {}
    for n, k in nk_list:
        for m in m_list:
            key = f"{m},{n},{k}"
            logger.info("tune %s", key)
            entry = _tune_one(m=m, n=n, k=k, rep_ms=rep_ms)
            logger.info("  -> %s", entry)
            entries[key] = entry

    payload = {
        "metadata": {
            "device_name": device_name,
            "tuned_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "rep_ms": rep_ms,
            "m_buckets": m_list,
            "nk_pairs": [[n, k] for n, k in nk_list],
        },
        "entries": entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("wrote %s with %d entries", output, len(entries))


def _tune_one(*, m: int, n: int, k: int, rep_ms: int) -> Dict:
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    timings: Dict[str, float] = {}
    for name, fn in _BACKENDS.items():
        try:
            timings[name] = _bench(fn=fn, x=x, y=y, rep_ms=rep_ms)
        except Exception:
            logger.warning(
                "backend=%s shape=(M=%d,N=%d,K=%d) raised; marking as +inf",
                name,
                m,
                n,
                k,
                exc_info=True,
            )
            timings[name] = float("inf")

    chosen: Optional[str] = min(timings, key=timings.__getitem__)
    if timings[chosen] == float("inf"):
        chosen = None

    entry: Dict = {"chosen": chosen}
    for name, t in timings.items():
        entry[f"{name}_us"] = None if t == float("inf") else round(t, 3)
    return entry


def _bench(*, fn: Callable, x: torch.Tensor, y: torch.Tensor, rep_ms: int) -> float:
    for _ in range(3):
        fn(x, y)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench_cudagraph(lambda: fn(x, y), rep=rep_ms)
    return ms * 1000.0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    typer.run(main)
