# SPDX-License-Identifier: Apache-2.0
"""Mooncake publication and restoration for daemon-owned expert HBM."""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .expert_source import ExpertSlotKey, ExpertSourceDirectory
from .protocol import recv_msg, send_msg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MooncakeRemoteSlot:
    """Mooncake-readable remote CUDA address for one expert slot."""

    pointer: int
    byte_size: int


@dataclass(frozen=True)
class MooncakeExpertSourceDescriptor:
    """Mooncake-specific locator for one daemon's expert HBM source."""

    source_id: str
    session_id: str
    slot_pointers: dict[ExpertSlotKey, MooncakeRemoteSlot]
    generation: int = 0

    def to_wire(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "session_id": self.session_id,
            "generation": self.generation,
            "slots": [
                {
                    "layer_id": key.layer_id,
                    "physical_slot": key.physical_slot,
                    "tensor_index": key.tensor_index,
                    "pointer": value.pointer,
                    "byte_size": value.byte_size,
                }
                for key, value in sorted(self.slot_pointers.items())
            ],
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> "MooncakeExpertSourceDescriptor":
        return cls(
            source_id=value["source_id"],
            session_id=value["session_id"],
            generation=value.get("generation", 0),
            slot_pointers={
                ExpertSlotKey(
                    entry["layer_id"],
                    entry["physical_slot"],
                    entry["tensor_index"],
                ): MooncakeRemoteSlot(entry["pointer"], entry["byte_size"])
                for entry in value["slots"]
            },
        )


def publish_mooncake_expert_source(
    directory: ExpertSourceDirectory,
    *,
    source_id: str,
    session_id: str,
    transfer_engine: Any,
) -> MooncakeExpertSourceDescriptor:
    """Register a generic source layout with Mooncake and publish its locator."""
    for region in directory.memory_regions:
        ret = transfer_engine.register_memory(region.pointer, region.byte_size)
        if ret != 0:
            raise RuntimeError(
                "Failed to register daemon MoE HBM region: "
                f"pointer={region.pointer}, bytes={region.byte_size}, error={ret}"
            )
    return MooncakeExpertSourceDescriptor(
        source_id=source_id,
        session_id=session_id,
        slot_pointers={
            key: MooncakeRemoteSlot(directory.pointer_for(location), location.byte_size)
            for key, location in directory.items()
        },
    )


def initialize_mooncake_expert_source(
    directory: ExpertSourceDirectory,
    *,
    source_id: str,
    gpu_id: int,
    ib_device: str | None,
) -> MooncakeExpertSourceDescriptor:
    """Create the Mooncake adapter and publish a daemon source through it."""
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        init_mooncake_transfer_engine,
    )
    from sglang.srt.utils.network import get_local_ip_auto

    transfer_engine = init_mooncake_transfer_engine(
        hostname=get_local_ip_auto(), gpu_id=gpu_id, ib_device=ib_device
    )
    return publish_mooncake_expert_source(
        directory,
        source_id=source_id,
        session_id=transfer_engine.session_id,
        transfer_engine=transfer_engine.engine,
    )


def fetch_mooncake_expert_source_descriptor(
    socket_path: str, *, timeout_s: float = 30.0
) -> MooncakeExpertSourceDescriptor:
    """Fetch the daemon's Mooncake HBM locator over its local control socket."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(socket_path)
        send_msg(conn, {"type": "fetch_expert_source"})
        response = recv_msg(conn)
    if response.get("status") != "ok":
        raise RuntimeError(
            "Weight-cache daemon did not provide a Mooncake HBM source: "
            f"{response.get('message', response.get('status'))}"
        )
    return MooncakeExpertSourceDescriptor.from_wire(response["descriptor"])


@dataclass(frozen=True)
class MooncakeRestoreSegment:
    source: MooncakeRemoteSlot
    destination_pointer: int
    byte_size: int


@dataclass(frozen=True)
class MooncakeTensorRestoreSegment:
    source: MooncakeRemoteSlot
    destination: torch.Tensor


def pack_mooncake_restore_segments(
    segments: list[MooncakeRestoreSegment],
) -> tuple[list[int], int]:
    offsets: list[int] = []
    total_bytes = 0
    for segment in segments:
        if segment.byte_size <= 0 or segment.source.byte_size != segment.byte_size:
            raise ValueError(
                "Restore segments require matching positive source and destination sizes"
            )
        offsets.append(total_bytes)
        total_bytes += segment.byte_size
    return offsets, total_bytes


class MooncakeExpertRestoreTransport:
    """Moves remote daemon HBM into caller-owned staging HBM via Mooncake."""

    def __init__(self, transfer_engine: Any) -> None:
        self._engine = transfer_engine

    def restore(
        self,
        descriptor: MooncakeExpertSourceDescriptor,
        segments: list[MooncakeRestoreSegment],
        *,
        staging_pointer: int,
        register_staging: Callable[[int, int], int],
        unregister_staging: Callable[[int], int],
        copy_from_staging: Callable[[int, int, int], None],
    ) -> None:
        offsets, total_bytes = pack_mooncake_restore_segments(segments)
        if not segments:
            return
        ret = register_staging(staging_pointer, total_bytes)
        if ret != 0:
            raise RuntimeError(f"Failed to register restore staging HBM: error={ret}")
        try:
            ret = self._engine.batch_transfer_sync_read(
                descriptor.session_id,
                [staging_pointer + offset for offset in offsets],
                [segment.source.pointer for segment in segments],
                [segment.byte_size for segment in segments],
            )
            if ret != 0:
                raise RuntimeError(
                    "Failed to restore Mooncake daemon HBM source "
                    f"{descriptor.source_id}: error={ret}"
                )
            for segment, offset in zip(segments, offsets):
                copy_from_staging(
                    segment.destination_pointer,
                    staging_pointer + offset,
                    segment.byte_size,
                )
        finally:
            unregister_staging(staging_pointer)


class MooncakeDaemonHBMRestorer:
    """Restores expert tensors from a daemon HBM source via Mooncake."""

    def __init__(self, transfer_engine: Any) -> None:
        self._engine = transfer_engine

    @classmethod
    def from_shared_transfer_engine(cls) -> "MooncakeDaemonHBMRestorer":
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        transfer_engine = get_mooncake_transfer_engine()
        if transfer_engine is None:
            raise RuntimeError("Mooncake Transfer Engine is required for HBM restore")
        return cls(transfer_engine.engine)

    def restore(
        self,
        descriptor: MooncakeExpertSourceDescriptor,
        segments: list[MooncakeTensorRestoreSegment],
    ) -> None:
        if not segments:
            return
        device = segments[0].destination.device
        if device.type != "cuda":
            raise ValueError("Mooncake HBM restore destinations must be CUDA tensors")
        restore_segments: list[MooncakeRestoreSegment] = []
        for segment in segments:
            destination = segment.destination
            if destination.device != device or not destination.is_contiguous():
                raise ValueError(
                    "Mooncake HBM restore destinations must be contiguous CUDA tensors "
                    "on one device"
                )
            restore_segments.append(
                MooncakeRestoreSegment(
                    segment.source,
                    destination.data_ptr(),
                    destination.numel() * destination.element_size(),
                )
            )

        _, total_bytes = pack_mooncake_restore_segments(restore_segments)
        staging = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        transport = MooncakeExpertRestoreTransport(self._engine)
        destination_by_pointer = {
            segment.destination.data_ptr(): segment.destination for segment in segments
        }

        def copy_from_staging(destination_pointer: int, staging_pointer: int, size: int):
            destination = destination_by_pointer[destination_pointer]
            offset = staging_pointer - staging.data_ptr()
            destination_bytes = destination.view(torch.uint8)
            destination_bytes.copy_(
                staging[offset : offset + size].view_as(destination_bytes)
            )

        def unregister_staging(pointer: int) -> int:
            ret = self._engine.unregister_memory(pointer)
            if ret != 0:
                logger.warning(
                    "Failed to unregister Mooncake restore staging memory: error=%d", ret
                )
            return ret

        transport.restore(
            descriptor,
            restore_segments,
            staging_pointer=staging.data_ptr(),
            register_staging=self._engine.register_memory,
            unregister_staging=unregister_staging,
            copy_from_staging=copy_from_staging,
        )
