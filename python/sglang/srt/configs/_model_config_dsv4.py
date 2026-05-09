"""DSV4-specific helpers extracted from configs/model_config.py.

The big DSV4 additions to ModelConfig.__init__ (probe-FP4-vs-FP8 logic +
the safetensors header probe) live here. Imported lazily by
ModelConfig._maybe_auto_set_dsv4_fp4_experts so non-DSV4 models don't
load this code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


# Matches routed-expert weight keys in both HF-style layouts
# (``...mlp.experts.<N>.{gate,up,down}_proj.weight``) and DeepseekV4 2604-style
# layouts (``...ffn.experts.<N>.w{1,2,3}.weight``). ``shared_experts`` is
# excluded because the index segment requires a digit after ``.experts.``.
_ROUTED_EXPERT_KEY_RE = re.compile(
    r"\.experts\.\d+\.(?:w[123]|down_proj|up_proj|gate_proj)\.weight$"
)


def probe_routed_expert_weight_dtype(model_path: str) -> Optional[str]:
    """Return the safetensors dtype string (e.g. ``F8_E4M3``, ``U8``) of one
    routed-expert weight tensor, or ``None`` if the checkpoint is remote or has
    no matching key. Reads only the safetensors header of the relevant shard.
    """
    if not os.path.isdir(model_path):
        return None

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    target_key = None
    target_shard_path = None

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {}) or {}
        for k, shard in weight_map.items():
            if _ROUTED_EXPERT_KEY_RE.search(k):
                target_key = k
                target_shard_path = os.path.join(model_path, shard)
                break
        if target_key is None:
            return None
    else:
        shards = sorted(Path(model_path).glob("*.safetensors"))
        if not shards:
            return None
        target_shard_path = str(shards[0])

    with open(target_shard_path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))

    if target_key is not None:
        meta = header.get(target_key)
        return meta.get("dtype") if meta else None

    for k, meta in header.items():
        if k == "__metadata__" or not isinstance(meta, dict):
            continue
        if _ROUTED_EXPERT_KEY_RE.search(k):
            return meta.get("dtype")
    return None


def maybe_auto_set_dsv4_fp4_experts(model_config: "ModelConfig") -> None:
    """Auto-set SGLANG_DSV4_FP4_EXPERTS based on the checkpoint's routed-expert
    weight dtype for DeepseekV4 in 2604 mode. See ModelConfig wrapper docstring
    for semantics.
    """
    from sglang.srt.configs.model_config import is_deepseek_compressed
    from sglang.srt.environ import envs

    if envs.SGLANG_DSV4_FP4_EXPERTS.is_set():
        return
    if not is_deepseek_compressed(model_config.hf_config):
        return
    if envs.SGLANG_DSV4_MODE.get() != "2604":
        return
    try:
        dtype = probe_routed_expert_weight_dtype(model_config.model_path)
    except Exception as e:
        logger.warning(
            "Failed to probe routed-expert dtype for %s; keeping "
            "SGLANG_DSV4_FP4_EXPERTS default. Reason: %s",
            model_config.model_path,
            e,
        )
        return
    if dtype is None:
        return
    if dtype in ("U8", "I8", "F4"):
        is_fp4_experts = True
    elif dtype == "F8_E4M3":
        is_fp4_experts = False
    else:
        logger.warning(
            "Unexpected routed-expert safetensors dtype=%s for 2604 mode; "
            "keeping SGLANG_DSV4_FP4_EXPERTS default.",
            dtype,
        )
        return
    envs.SGLANG_DSV4_FP4_EXPERTS.set(is_fp4_experts)
    logger.info(
        "Auto-detected routed-expert safetensors dtype=%s; "
        "SGLANG_DSV4_FP4_EXPERTS=%s",
        dtype,
        is_fp4_experts,
    )
