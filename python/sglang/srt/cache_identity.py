"""Stable encoding for radix-cache namespace identities."""

from __future__ import annotations

import json
from typing import Optional

_CACHE_IDENTITY_VERSION = 1


def encode_cache_identity(domain: str, **components: Optional[str]) -> Optional[str]:
    """Encode named cache-identity components without semantic collisions."""
    normalized = {}
    for name, value in components.items():
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"Value of {name} must be a string, but got {type(value).__name__}"
            )
        normalized[name] = value or None

    if all(value is None for value in normalized.values()):
        return None

    return json.dumps(
        {
            "components": normalized,
            "domain": domain,
            "version": _CACHE_IDENTITY_VERSION,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
