
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from sglang.srt.utils import get_device_name, next_power_of_2

logger = logging.getLogger(__name__)

Backend = Literal["cublas", "deep_gemm"]

_FALLBACK: Backend = "cublas"
_CONFIG_DIR = Path(__file__).parent / "configs"


def pick_backend(*, m: int, n: int, k: int) -> Backend:
    m_bucket = next_power_of_2(m)
    device_name = _cached_device_name()
    entries = _load_config(device_name)

    key = f"{m_bucket},{n},{k}"
    entry = entries.get(key)
    if entry is None:
        logger.debug(
            "linear_bf16_fp32 config miss key=%s (real M=%d) device=%s; falling back to %s",
            key,
            m,
            device_name,
            _FALLBACK,
        )
        return _FALLBACK
    return entry["chosen"]


@lru_cache(maxsize=1)
def _cached_device_name() -> str:
    return get_device_name(0).replace(" ", "_")


@lru_cache(maxsize=None)
def _load_config(device_name: str) -> dict:
    path = _CONFIG_DIR / f"device_name={device_name}.json"
    if not path.exists():
        logger.warning(
            "linear_bf16_fp32 tuned config not found at %s; selector will always fall back to %s",
            path,
            _FALLBACK,
        )
        return {}
    with path.open() as f:
        payload = json.load(f)
    return payload.get("entries", {})
