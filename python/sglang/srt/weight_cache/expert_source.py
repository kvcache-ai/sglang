# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral layout of routed-expert tensors owned by a cache daemon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True, order=True)
class ExpertSlotKey:
    """Identifies one physical routed-expert slot in a model replica."""

    layer_id: int
    physical_slot: int
    tensor_index: int


@dataclass(frozen=True)
class ExpertMemoryRegion:
    """One contiguous CUDA allocation owned by the daemon."""

    pointer: int
    byte_size: int


@dataclass(frozen=True)
class ExpertSlotLocation:
    """Location of one slot within a daemon-owned memory region."""

    region_index: int
    byte_offset: int
    byte_size: int


class ExpertSourceDirectory:
    """Transport-neutral physical layout of daemon-owned routed-expert tensors.

    The directory records each complete contiguous tensor as a memory region and
    maps every physical expert slot to its region-relative range. Transports
    later register the regions and encode transport-specific remote locators.
    """

    def __init__(
        self,
        slot_locations: dict[ExpertSlotKey, ExpertSlotLocation],
        memory_regions: list[ExpertMemoryRegion],
    ) -> None:
        self._slot_locations = slot_locations
        self._memory_regions = memory_regions

    @classmethod
    def from_routed_experts(
        cls, routed_experts_weights_of_layer: dict[int, list[torch.Tensor]]
    ) -> "ExpertSourceDirectory":
        slots: dict[ExpertSlotKey, ExpertSlotLocation] = {}
        regions: list[ExpertMemoryRegion] = []
        region_indices: dict[ExpertMemoryRegion, int] = {}
        for layer_id, tensors in sorted(routed_experts_weights_of_layer.items()):
            for tensor_index, tensor in enumerate(tensors):
                if tensor.ndim == 0 or not tensor.is_contiguous():
                    raise ValueError(
                        "Routed-expert tensors must be contiguous and have a "
                        "physical-slot dimension: "
                        f"layer={layer_id}, tensor={tensor_index}"
                    )
                region = ExpertMemoryRegion(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size()
                )
                region_index = region_indices.get(region)
                if region_index is None:
                    region_index = len(regions)
                    regions.append(region)
                    region_indices[region] = region_index
                for physical_slot in range(tensor.shape[0]):
                    slot = tensor[physical_slot]
                    slots[ExpertSlotKey(layer_id, physical_slot, tensor_index)] = (
                        ExpertSlotLocation(
                            region_index=region_index,
                            byte_offset=slot.data_ptr() - region.pointer,
                            byte_size=slot.numel() * slot.element_size(),
                        )
                    )
        return cls(slots, regions)

    @property
    def memory_regions(self) -> list[ExpertMemoryRegion]:
        return list(self._memory_regions)

    def items(self) -> Iterable[tuple[ExpertSlotKey, ExpertSlotLocation]]:
        return self._slot_locations.items()

    def pointer_for(self, location: ExpertSlotLocation) -> int:
        """Return the daemon-local CUDA address for a transport adapter."""
        region = self._memory_regions[location.region_index]
        return region.pointer + location.byte_offset
