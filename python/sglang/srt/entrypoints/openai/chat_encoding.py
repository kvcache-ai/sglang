"""Helpers for model-specific chat encoding configuration.

This module intentionally contains only the DeepSeek-V4 reasoning-effort
profile logic needed by the KTransformers release branch.  Encoder dispatch
remains in ``serving_chat.py`` so that unrelated model paths are unchanged.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

from sglang.srt.entrypoints.openai import encoding_dsv4

logger = logging.getLogger(__name__)

DSV4_REASONING_EFFORT_PROFILE_OVERRIDE = "dsv4_reasoning_effort_profile"
_DSV4_REASONING_EFFORT_ENCODER = "encoding/encoding_dsv4.py"
_MAX_DSV4_ENCODER_BYTES = 1 << 20


def _detect_dsv4_reasoning_effort_profile(
    model_path: str, revision: Optional[str] = None
) -> Optional[str]:
    """Detect the preview or official profile without importing model code."""

    encoder_path = Path(model_path) / _DSV4_REASONING_EFFORT_ENCODER
    try:
        if not encoder_path.is_file():
            from huggingface_hub import hf_hub_download

            encoder_path = Path(
                hf_hub_download(
                    model_path,
                    _DSV4_REASONING_EFFORT_ENCODER,
                    revision=revision,
                )
            )
        if encoder_path.stat().st_size > _MAX_DSV4_ENCODER_BYTES:
            return None
        tree = ast.parse(encoder_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.debug(
            "Could not inspect DeepSeek-V4 checkpoint encoder at %s: %s",
            encoder_path,
            error,
        )
        return None

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                assignments[target.id] = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue

    prompts = assignments.get("REASONING_EFFORT_PROMPTS")
    if (
        assignments.get("DEFAULT_REASONING_EFFORT") == "low"
        and isinstance(prompts, dict)
        and {"low", "high", "max"} <= prompts.keys()
    ):
        return "official"
    if "REASONING_EFFORT_MAX" in assignments:
        return "preview"
    return None


def _validate_dsv4_reasoning_effort_profile(profile: str) -> str:
    if profile not in encoding_dsv4.REASONING_EFFORT_PROFILES:
        raise ValueError(
            f"Invalid {DSV4_REASONING_EFFORT_PROFILE_OVERRIDE}: {profile!r}; "
            f"expected one of {list(encoding_dsv4.REASONING_EFFORT_PROFILES)}"
        )
    return profile


def resolve_dsv4_reasoning_effort_profile(
    *,
    model_path: str,
    revision: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    if override is not None:
        return _validate_dsv4_reasoning_effort_profile(override)

    return (
        _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        or "preview"
    )
