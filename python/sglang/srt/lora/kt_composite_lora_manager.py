# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""M1/M2 KT composite multi-LoRA manager: CPU expert pool + activate / grouped metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    KTExpertLoraWeights,
    _load_kt_expert_lora_weights,
    _make_zero_kt_expert_lora_weights,
)
from sglang.srt.lora.lora_registry import LoRARef

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Slot 0 is reserved for base-only (zero delta).
BASE_KT_LORA_SLOT = 0


@dataclass
class KTExpertSlot:
    slot_id: int
    generation: int = 0
    lora_id: Optional[str] = None
    lora_name: Optional[str] = None
    expert_path: Optional[str] = None
    pinned: bool = False
    state: str = "empty"  # empty | loading | ready | active | evicting
    # Per-layer weights keyed by layer_idx (only on owner TP rank).
    layer_weights: Dict[int, KTExpertLoraWeights] = field(default_factory=dict)


class KTExpertLoRAPool:
    """Fixed CPU expert LoRA slots. Slot 0 is always base-only zeros."""

    def __init__(self, max_loaded_loras: int):
        if max_loaded_loras < 1:
            raise ValueError("kt_max_loaded_loras must be >= 1")
        # +1 for base slot 0
        self.max_adapter_slots = max_loaded_loras
        self.slots: List[KTExpertSlot] = [
            KTExpertSlot(slot_id=i) for i in range(max_loaded_loras + 1)
        ]
        self.lora_id_to_slot: Dict[str, int] = {}
        self.slots[BASE_KT_LORA_SLOT].state = "ready"
        self.slots[BASE_KT_LORA_SLOT].lora_name = "__base__"

    def get_slot(self, lora_id: Optional[str]) -> int:
        if lora_id is None:
            return BASE_KT_LORA_SLOT
        if lora_id not in self.lora_id_to_slot:
            raise KeyError(f"KT expert LoRA id {lora_id} is not resident in the pool")
        return self.lora_id_to_slot[lora_id]

    def alloc_slot(self, lora_ref: LoRARef) -> int:
        if lora_ref.lora_id in self.lora_id_to_slot:
            return self.lora_id_to_slot[lora_ref.lora_id]
        for slot in self.slots[1:]:
            if slot.state == "empty":
                slot.lora_id = lora_ref.lora_id
                slot.lora_name = lora_ref.lora_name
                slot.expert_path = lora_ref.kt_expert_lora_path
                slot.pinned = bool(lora_ref.pinned)
                slot.state = "loading"
                self.lora_id_to_slot[lora_ref.lora_id] = slot.slot_id
                return slot.slot_id
        raise RuntimeError(
            f"KT expert LoRA pool is full (max_loaded={self.max_adapter_slots}); "
            f"cannot load {lora_ref.lora_name}"
        )

    def mark_ready(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        slot.state = "ready"
        slot.generation += 1

    def release(self, lora_id: str) -> Optional[int]:
        slot_id = self.lora_id_to_slot.pop(lora_id, None)
        if slot_id is None:
            return None
        if slot_id == BASE_KT_LORA_SLOT:
            raise RuntimeError("Cannot release base KT LoRA slot")
        slot = self.slots[slot_id]
        slot.lora_id = None
        slot.lora_name = None
        slot.expert_path = None
        slot.pinned = False
        slot.state = "empty"
        slot.layer_weights.clear()
        slot.generation += 1
        return slot_id


class KTCompositeLoRAManager:
    """Coordinates KT CPU expert LoRA residency and M1/M2 batch prepare."""

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.server_args = model_runner.server_args
        self.tp_rank = model_runner.tp_rank
        composite_refs = [
            ref
            for ref in (self.server_args.lora_paths or [])
            if getattr(ref, "adapter_kind", "ordinary") == "kt_composite"
        ]
        self.composite_refs: Dict[str, LoRARef] = {
            ref.lora_id: ref for ref in composite_refs
        }
        max_loaded = self.server_args.kt_max_loaded_loras
        if max_loaded is None:
            max_loaded = max(len(composite_refs), 1)
        self.pool = KTExpertLoRAPool(max_loaded)
        self.layers: List[KTEPWrapperMethod] = []
        # Sentinel (not a real slot id): forces the first activate_slot() call
        # to actually run method.activate_kt_lora_slot() instead of short-
        # circuiting because slot_id == BASE_KT_LORA_SLOT (0) already matched
        # this field's old default value.
        self._active_slot: int = -1
        self._initialized = False
        self.dispatch_mode = (
            getattr(self.server_args, "kt_lora_dispatch", None) or "single"
        ).lower()
        self.max_loras_per_batch = int(
            getattr(self.server_args, "kt_max_loras_per_batch", 1) or 1
        )
        if self.dispatch_mode == "single":
            self.max_loras_per_batch = 1
        # Per-forward token slot tensor shared with KTEPWrapperMethod.apply.
        self.current_kt_lora_token_slots: Optional["torch.Tensor"] = None

    @property
    def enabled(self) -> bool:
        return len(self.composite_refs) > 0

    def collect_layers(self) -> None:
        layers: List[KTEPWrapperMethod] = []
        model = self.model_runner.model
        for module in model.modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, KTEPWrapperMethod) and quant_method.kt_expert_lora_enabled:
                layers.append(quant_method)
        # Stable order by layer_idx
        layers.sort(key=lambda m: m.kt_config.layer_idx)
        self.layers = layers
        if self.tp_rank == 0 and not layers and self.enabled:
            logger.warning(
                "KTCompositeLoRAManager: no KTEPWrapperMethod layers with expert "
                "LoRA enabled were found."
            )

    def initialize_from_server_args(self) -> None:
        """Load all startup composite adapters into the CPU pool and activate base."""
        if not self.enabled:
            return
        if self._initialized:
            return
        self.collect_layers()
        for method in self.layers:
            method.kt_lora_dispatch = self.dispatch_mode
        # NOTE: pool bookkeeping (slot ids, ready/active state) must be
        # populated identically on *every* TP rank, even though the actual
        # CPU expert weight residency/compute only happens on tp_rank == 0.
        # `build_token_slots()` / `_ensure_resident()` run per forward batch
        # on all ranks and need a valid `lora_id -> slot` mapping to avoid
        # crashing with "KT expert LoRA id ... is not resident in the pool".
        # `_register_base_slot` / `_load_composite_into_pool` internally
        # skip the per-layer weight staging when this rank has no locally
        # owned KT expert-LoRA layers (or no template yet), so calling them
        # unconditionally is safe.
        self._register_base_slot()
        for lora_ref in self.composite_refs.values():
            self._load_composite_into_pool(lora_ref)
        if self.tp_rank == 0:
            # Default active = base; first real request will activate.
            self.activate_slot(BASE_KT_LORA_SLOT)
        self._initialized = True
        logger.info(
            "KTCompositeLoRAManager ready: %d composite adapter(s), %d KT layer(s), "
            "active_slot=%d, dispatch=%s, max_loras_per_batch=%d",
            len(self.composite_refs),
            len(self.layers),
            self._active_slot,
            self.dispatch_mode,
            self.max_loras_per_batch,
        )

    def _register_base_slot(self) -> None:
        if not self.layers:
            self.pool.slots[BASE_KT_LORA_SLOT].state = "ready"
            return
        template = self.layers[0].kt_expert_lora_weights
        if template is None:
            # This TP rank owns KTEPWrapperMethod objects but they have not
            # (and, on non-zero TP ranks, will never) materialize a real KT
            # expert LoRA template — CPU expert residency/compute is owned
            # exclusively by tp_rank == 0. Still mark the base slot ready so
            # that pool-level bookkeeping (lora_id -> slot lookups used by
            # every rank's build_token_slots()) stays consistent.
            self.pool.slots[BASE_KT_LORA_SLOT].state = "ready"
            return
        for method in self.layers:
            zeros = _make_zero_kt_expert_lora_weights(
                rank=template.rank,
                alpha=template.alpha,
                num_experts=method.kt_expert_lora_weights.gate_lora_a.shape[0]
                if method.kt_expert_lora_weights is not None
                else template.gate_lora_a.shape[0],
                hidden_size=template.gate_lora_a.shape[-1],
                moe_intermediate_size=template.gate_lora_b.shape[1],
            )
            # Prefer layer's own zero template if already created in create_weights.
            if method.kt_expert_lora_weights is not None:
                zeros = method.kt_expert_lora_weights
            method.register_kt_lora_slot(BASE_KT_LORA_SLOT, zeros)
            self.pool.slots[BASE_KT_LORA_SLOT].layer_weights[
                method.kt_config.layer_idx
            ] = zeros
        self.pool.slots[BASE_KT_LORA_SLOT].state = "ready"

    def _load_composite_into_pool(self, lora_ref: LoRARef) -> int:
        if not lora_ref.kt_expert_lora_path:
            raise ValueError(
                f"Composite LoRA {lora_ref.lora_name} missing kt_expert_lora_path"
            )
        slot_id = self.pool.alloc_slot(lora_ref)
        try:
            if not self.layers or self.layers[0].kt_expert_lora_weights is None:
                # No locally-owned KT expert-LoRA layers/templates on this TP
                # rank (CPU expert residency/compute is owned exclusively by
                # tp_rank == 0). Only the pool-level slot bookkeeping is
                # needed here so that build_token_slots()/_ensure_resident()
                # succeed uniformly across all ranks.
                self.pool.mark_ready(slot_id)
                return slot_id
            for method in self.layers:
                template = method.kt_expert_lora_weights
                if template is None:
                    raise RuntimeError(
                        f"Layer {method.kt_config.layer_idx} missing KT expert "
                        "LoRA template weights for managed multi-LoRA load."
                    )
                weights = _load_kt_expert_lora_weights(
                    adapter_path=lora_ref.kt_expert_lora_path,
                    layer_idx=method.kt_config.layer_idx,
                    num_experts=template.gate_lora_a.shape[0],
                    hidden_size=template.gate_lora_a.shape[-1],
                    moe_intermediate_size=template.gate_lora_b.shape[1],
                )
                method.register_kt_lora_slot(slot_id, weights)
                self.pool.slots[slot_id].layer_weights[
                    method.kt_config.layer_idx
                ] = weights
            self.pool.mark_ready(slot_id)
            logger.info(
                "Loaded KT composite expert LoRA %s into slot %d from %s",
                lora_ref.lora_name,
                slot_id,
                lora_ref.kt_expert_lora_path,
            )
            return slot_id
        except Exception:
            # Best-effort rollback of partial registration.
            for method in self.layers:
                if slot_id in method.kt_lora_slot_weights:
                    method.unload_kt_lora_slot(slot_id)
            self.pool.release(lora_ref.lora_id)
            raise

    def activate_slot(self, slot_id: int) -> None:
        if self.tp_rank != 0:
            self._active_slot = slot_id
            return
        if slot_id == self._active_slot:
            return
        for method in self.layers:
            method.activate_kt_lora_slot(slot_id)
        self._active_slot = slot_id
        if slot_id < len(self.pool.slots):
            self.pool.slots[slot_id].state = "active"
        logger.debug("Activated KT expert LoRA slot %d", slot_id)

    def validate_batch(self, lora_ids: Set[Optional[str]]) -> bool:
        """Admit a candidate LoRA set if distinct count fits and all are resident."""
        distinct = {lid for lid in lora_ids if lid is not None}
        if len(distinct) > self.max_loras_per_batch:
            return False
        for lid in distinct:
            if lid not in self.pool.lora_id_to_slot and lid not in self.composite_refs:
                return False
            # Must be ready if already mapped.
            if lid in self.pool.lora_id_to_slot:
                slot = self.pool.slots[self.pool.lora_id_to_slot[lid]]
                if slot.state not in ("ready", "active", "loading"):
                    return False
        return True

    def _ensure_resident(self, lora_id: str) -> int:
        if lora_id in self.pool.lora_id_to_slot:
            return self.pool.get_slot(lora_id)
        ref = self.composite_refs.get(lora_id)
        if ref is None:
            raise KeyError(
                f"KT composite LoRA id {lora_id} is unknown to KTCompositeLoRAManager"
            )
        # _load_composite_into_pool() gracefully degrades to pool-only
        # bookkeeping on ranks with no locally-owned KT expert-LoRA layers,
        # so it is safe (and necessary) to call on every TP rank.
        self._load_composite_into_pool(ref)
        return self.pool.get_slot(lora_id)

    def build_token_slots(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Map per-request lora_ids to a per-token KT slot tensor."""
        lora_ids = forward_batch.lora_ids or []
        req_slots: List[int] = []
        generations: List[int] = []
        for lid in lora_ids:
            if lid is None:
                slot = BASE_KT_LORA_SLOT
            else:
                slot = self._ensure_resident(lid)
            req_slots.append(slot)
            generations.append(self.pool.slots[slot].generation)

        forward_batch.kt_lora_req_slots = req_slots

        device = (
            forward_batch.input_ids.device
            if getattr(forward_batch, "input_ids", None) is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not req_slots:
            token_slots = torch.empty(0, dtype=torch.int32, device=device)
            gen_tensor = torch.empty(0, dtype=torch.int32, device=device)
        elif (
            forward_batch.forward_mode is not None
            and forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            # One generated token per request in standard decode.
            token_slots = torch.tensor(req_slots, dtype=torch.int32, device=device)
            gen_tensor = torch.tensor(generations, dtype=torch.int32, device=device)
        else:
            lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
            if lengths is None and getattr(forward_batch, "extend_seq_lens", None) is not None:
                lengths = forward_batch.extend_seq_lens.tolist()
            if lengths is None:
                # Fallback: treat as one token per request (cuda-graph / odd paths).
                lengths = [1] * len(req_slots)
            if len(lengths) != len(req_slots):
                raise RuntimeError(
                    f"KT LoRA token-slot length mismatch: "
                    f"{len(lengths)} seq lens vs {len(req_slots)} requests"
                )
            pieces = [
                torch.full((int(n),), int(slot), dtype=torch.int32, device=device)
                for slot, n in zip(req_slots, lengths)
                if int(n) > 0
            ]
            token_slots = (
                torch.cat(pieces, dim=0)
                if pieces
                else torch.empty(0, dtype=torch.int32, device=device)
            )
            gen_pieces = [
                torch.full((int(n),), int(gen), dtype=torch.int32, device=device)
                for gen, n in zip(generations, lengths)
                if int(n) > 0
            ]
            gen_tensor = (
                torch.cat(gen_pieces, dim=0)
                if gen_pieces
                else torch.empty(0, dtype=torch.int32, device=device)
            )

        forward_batch.kt_lora_token_slots = token_slots
        forward_batch.kt_lora_slot_generations = gen_tensor
        return token_slots

    def _publish_token_slots_to_layers(self, token_slots: Optional[torch.Tensor]) -> None:
        self.current_kt_lora_token_slots = token_slots
        for method in self.layers:
            method.kt_lora_token_slots_for_batch = token_slots
            method.kt_lora_dispatch = self.dispatch_mode

    def prepare_batch(self, forward_batch: "ForwardBatch") -> None:
        if not self.enabled or not self._initialized:
            return
        lora_ids = forward_batch.lora_ids or []
        distinct = {lid for lid in lora_ids if lid is not None}
        if len(distinct) > self.max_loras_per_batch:
            raise RuntimeError(
                f"KT composite rejects mixed-adapter batch with {len(distinct)} "
                f"adapters (limit={self.max_loras_per_batch}, "
                f"dispatch={self.dispatch_mode}): {distinct}"
            )

        for lid in distinct:
            self._ensure_resident(lid)

        token_slots = self.build_token_slots(forward_batch)
        self._publish_token_slots_to_layers(token_slots)

        if self.dispatch_mode == "grouped":
            unique_slots = (
                torch.unique(token_slots).tolist() if token_slots.numel() else [BASE_KT_LORA_SLOT]
            )
            if len(unique_slots) <= 1:
                # Fast path: single active slot for the whole batch.
                self.activate_slot(
                    int(unique_slots[0]) if unique_slots else BASE_KT_LORA_SLOT
                )
            # Multi-slot: wrapper.apply performs per-group activate; do not pin one slot.
            return

        # M1 / single: activate exactly one slot for the batch.
        if not distinct:
            self.activate_slot(BASE_KT_LORA_SLOT)
            return
        lora_id = next(iter(distinct))
        self.activate_slot(self.pool.get_slot(lora_id))

    def stage(self, lora_ref: LoRARef) -> int:
        """Begin loading a new composite adapter into the KT pool."""
        if getattr(lora_ref, "adapter_kind", "ordinary") != "kt_composite":
            raise ValueError("stage() requires a kt_composite LoRARef")
        self.composite_refs[lora_ref.lora_id] = lora_ref
        if self.tp_rank != 0:
            return -1
        return self._load_composite_into_pool(lora_ref)

    def commit(self, lora_ref: LoRARef, slot_id: int) -> None:
        # Slot is already marked ready in _load_composite_into_pool.
        self.composite_refs[lora_ref.lora_id] = lora_ref
        logger.info(
            "Committed KT composite LoRA %s at slot %d", lora_ref.lora_name, slot_id
        )

    def abort(self, lora_ref: LoRARef, slot_id: Optional[int] = None) -> None:
        if self.tp_rank == 0:
            released = self.pool.release(lora_ref.lora_id)
            if released is not None:
                for method in self.layers:
                    method.unload_kt_lora_slot(released)
        self.composite_refs.pop(lora_ref.lora_id, None)

    def unload(self, lora_ref: LoRARef) -> None:
        self.abort(lora_ref)
