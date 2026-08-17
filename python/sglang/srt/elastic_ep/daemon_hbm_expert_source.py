# SPDX-License-Identifier: Apache-2.0
"""Daemon-HBM recovery sources for Elastic EP faults."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang.srt.weight_cache.expert_source import ExpertSlotKey
from sglang.srt.weight_cache.mooncake_expert_source import (
    MooncakeDaemonHBMRestorer,
    MooncakeExpertSourceDescriptor,
    MooncakeTensorRestoreSegment,
    fetch_mooncake_expert_source_descriptor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaemonHBMSourceRegistry:
    """Daemon descriptors indexed by the healthy world's global rank."""

    descriptors: dict[int, MooncakeExpertSourceDescriptor]

    @classmethod
    def from_wire(
        cls, ranks: list[int], descriptors: list[dict[str, Any]]
    ) -> "DaemonHBMSourceRegistry":
        if len(ranks) != len(descriptors):
            raise RuntimeError("Daemon-HBM descriptor collection has mismatched ranks")
        return cls(
            {
                rank: MooncakeExpertSourceDescriptor.from_wire(descriptor)
                for rank, descriptor in zip(ranks, descriptors)
            }
        )

    def descriptor_for_rank(self, rank: int) -> MooncakeExpertSourceDescriptor:
        try:
            return self.descriptors[rank]
        except KeyError as exc:
            raise RuntimeError(f"No daemon-HBM source descriptor for rank {rank}") from exc


def collect_daemon_hbm_source_registry(
    *, socket_path: str, world_group
) -> DaemonHBMSourceRegistry:
    """Collect the local weight-cache descriptor from every live world rank."""
    local_descriptor = fetch_mooncake_expert_source_descriptor(socket_path)
    gathered = world_group.all_gather_object(local_descriptor.to_wire())
    registry = DaemonHBMSourceRegistry.from_wire(list(world_group.ranks), gathered)
    logger.info(
        "[DaemonHBMExpertSource] collected_descriptor_ranks=%s",
        sorted(registry.descriptors),
    )
    return registry


def select_daemon_hbm_source_slot(
    *,
    old_physical_to_logical: list[int],
    logical_expert_id: int,
    num_local_physical_experts: int,
    old_ep_world_ranks: list[int],
    registry: DaemonHBMSourceRegistry,
) -> tuple[int, MooncakeExpertSourceDescriptor, int]:
    """Resolve a missing logical expert against the pre-fault placement."""
    for physical_slot, mapped_expert_id in enumerate(old_physical_to_logical):
        if mapped_expert_id != logical_expert_id:
            continue
        source_ep_rank, source_slot = divmod(
            physical_slot, num_local_physical_experts
        )
        if source_ep_rank >= len(old_ep_world_ranks):
            raise RuntimeError(
                "Pre-fault EPLB placement refers to an EP rank outside its "
                f"membership: ep_rank={source_ep_rank}, ranks={old_ep_world_ranks}"
            )
        source_world_rank = old_ep_world_ranks[source_ep_rank]
        if source_world_rank in registry.descriptors:
            return (
                source_world_rank,
                registry.descriptor_for_rank(source_world_rank),
                source_slot,
            )
    raise RuntimeError(
        f"No retained daemon-HBM source for logical expert {logical_expert_id}"
    )


class DaemonHBMExpertSourceClient:
    """Restores EPLB-missing experts from retained weight-cache daemons."""

    def __init__(
        self,
        *,
        moe_ep_rank: int,
        old_ep_world_ranks: list[int],
        get_model: Callable[[], Any],
        registry: DaemonHBMSourceRegistry,
        restorer: MooncakeDaemonHBMRestorer | None = None,
    ) -> None:
        self.moe_ep_rank = moe_ep_rank
        self.old_ep_world_ranks = old_ep_world_ranks
        self._get_model = get_model
        self.registry = registry
        self._restorer = restorer or MooncakeDaemonHBMRestorer.from_shared_transfer_engine()

    def restore_missing_experts(
        self,
        *,
        missing_logical_experts_by_layers: dict[int, list[int]],
        old_expert_location_metadata,
        new_expert_location_metadata,
    ) -> None:
        """Restore every local destination before the new placement is committed."""
        num_local = old_expert_location_metadata.num_local_physical_experts
        local_start = self.moe_ep_rank * num_local
        local_end = local_start + num_local
        routed_experts = self._get_model().routed_experts_weights_of_layer
        segments_by_source: dict[
            int, list[MooncakeTensorRestoreSegment]
        ] = {}

        for layer_id, logical_experts in missing_logical_experts_by_layers.items():
            old_row = old_expert_location_metadata.physical_to_logical_map_cpu[
                layer_id
            ].tolist()
            for logical_expert_id in logical_experts:
                source_rank, descriptor, source_slot = select_daemon_hbm_source_slot(
                    old_physical_to_logical=old_row,
                    logical_expert_id=logical_expert_id,
                    num_local_physical_experts=num_local,
                    old_ep_world_ranks=self.old_ep_world_ranks,
                    registry=self.registry,
                )
                source_segments = segments_by_source.setdefault(source_rank, [])
                for destination_slot in new_expert_location_metadata.logical_to_all_physical(
                    layer_id, logical_expert_id
                ):
                    if not local_start <= destination_slot < local_end:
                        continue
                    local_slot = destination_slot - local_start
                    for tensor_index, tensor in enumerate(routed_experts[layer_id]):
                        source = descriptor.slot_pointers[
                            ExpertSlotKey(layer_id, source_slot, tensor_index)
                        ]
                        destination = tensor[local_slot]
                        byte_size = destination.numel() * destination.element_size()
                        if source.byte_size != byte_size:
                            raise RuntimeError(
                                "Daemon-HBM source and EPLB destination sizes differ: "
                                f"layer={layer_id}, tensor={tensor_index}, "
                                f"source={source.byte_size}, destination={byte_size}"
                            )
                        source_segments.append(
                            MooncakeTensorRestoreSegment(source, destination)
                        )

        for source_rank, segments in segments_by_source.items():
            descriptor = self.registry.descriptor_for_rank(source_rank)
            self._restorer.restore(descriptor, segments)
            logger.info(
                "[DaemonHBMExpertRecovery] source_rank=%d segments=%d "
                "transfer_complete=true",
                source_rank,
                len(segments),
            )
