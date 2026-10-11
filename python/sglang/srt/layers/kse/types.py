"""Core data types for the KVCache Sparsity Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch


class Granularity(Enum):
    """Selection granularity of a sparsity policy."""

    TOKEN = "token"
    PAGE = "page"


class Frequency(Enum):
    """How often ``SparsityPolicy.select()`` should be invoked."""

    PER_REQUEST = "per_request"
    PER_STEP = "per_step"
    PER_LAYER = "per_layer"


@dataclass
class SelectionResult:
    """Output of ``SparsityPolicy.select()``.

    All index tensors use *logical* positions (0-based offsets within each
    request's sequence).  ``MetadataAdapter`` translates them to physical
    KV-cache locations via ``req_to_token``.
    """

    granularity: Granularity

    # [batch_size, max_selected] — logical indices, padded with -1
    selected_indices: torch.Tensor

    # [batch_size] — number of valid entries per request
    valid_lengths: torch.Tensor

    # [batch_size] — which requests actually use sparse attention
    sparse_mask: torch.Tensor

    # Optional per-layer override: if set, only these layers are affected
    layer_ids: Optional[List[int]] = None
