# SPDX-License-Identifier: Apache-2.0

import socket
import tempfile
import threading
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.srt.weight_cache.expert_source import ExpertSlotKey, ExpertSourceDirectory
from sglang.srt.weight_cache.mooncake_expert_source import (
    MooncakeExpertRestoreTransport,
    MooncakeExpertSourceDescriptor,
    MooncakeRemoteSlot,
    MooncakeRestoreSegment,
    fetch_mooncake_expert_source_descriptor,
    pack_mooncake_restore_segments,
    publish_mooncake_expert_source,
)
from sglang.srt.weight_cache.protocol import recv_msg, send_msg

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingTransferEngine:
    def __init__(self, register_result=0, read_result=0):
        self.register_result = register_result
        self.read_result = read_result
        self.registered = []
        self.reads = []

    def register_memory(self, pointer, byte_size):
        self.registered.append((pointer, byte_size))
        return self.register_result

    def batch_transfer_sync_read(self, session_id, destinations, sources, sizes):
        self.reads.append((session_id, destinations, sources, sizes))
        return self.read_result


class TestMooncakeExpertSource(CustomTestCase):
    def test_publishes_transport_specific_descriptor(self):
        tensor = torch.empty(2, 4)
        directory = ExpertSourceDirectory.from_routed_experts({0: [tensor]})
        engine = _RecordingTransferEngine()

        descriptor = publish_mooncake_expert_source(
            directory,
            source_id="pp=0/tp=0/moe_dp=0/moe_ep=0",
            session_id="source:1",
            transfer_engine=engine,
        )

        self.assertEqual(
            engine.registered,
            [(tensor.data_ptr(), tensor.numel() * tensor.element_size())],
        )
        self.assertEqual(
            descriptor.slot_pointers[ExpertSlotKey(0, 1, 0)].pointer,
            tensor[1].data_ptr(),
        )

    def test_descriptor_round_trip_keeps_distinct_dp_replicas(self):
        first = MooncakeExpertSourceDescriptor(
            "pp=0/tp=0/moe_dp=0/moe_ep=0",
            "first:1",
            {ExpertSlotKey(0, 0, 0): MooncakeRemoteSlot(123, 4)},
        )
        second = MooncakeExpertSourceDescriptor(
            "pp=0/tp=0/moe_dp=1/moe_ep=0",
            "second:1",
            {ExpertSlotKey(0, 0, 0): MooncakeRemoteSlot(456, 4)},
        )

        self.assertNotEqual(first.source_id, second.source_id)
        self.assertEqual(MooncakeExpertSourceDescriptor.from_wire(first.to_wire()), first)

    def test_fetches_descriptor_from_daemon_socket(self):
        descriptor = MooncakeExpertSourceDescriptor(
            "pp=0/tp=0/moe_dp=0/moe_ep=0",
            "source:1",
            {ExpertSlotKey(0, 0, 0): MooncakeRemoteSlot(123, 4)},
        )
        with tempfile.TemporaryDirectory() as directory:
            socket_path = f"{directory}/daemon.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(socket_path)
            server.listen(1)

            def serve_once():
                conn, _ = server.accept()
                with conn:
                    self.assertEqual(recv_msg(conn), {"type": "fetch_expert_source"})
                    send_msg(conn, {"status": "ok", "descriptor": descriptor.to_wire()})

            worker = threading.Thread(target=serve_once)
            worker.start()
            try:
                self.assertEqual(
                    fetch_mooncake_expert_source_descriptor(socket_path), descriptor
                )
            finally:
                worker.join()
                server.close()


class TestMooncakeExpertRestoreTransport(CustomTestCase):
    def test_packs_and_restores_through_staging_then_local_copy(self):
        descriptor = MooncakeExpertSourceDescriptor("source", "source:1", {})
        segments = [
            MooncakeRestoreSegment(MooncakeRemoteSlot(101, 4), 201, 4),
            MooncakeRestoreSegment(MooncakeRemoteSlot(105, 8), 205, 8),
        ]
        engine = _RecordingTransferEngine()
        registrations = []
        copies = []
        transport = MooncakeExpertRestoreTransport(engine)

        transport.restore(
            descriptor,
            segments,
            staging_pointer=1000,
            register_staging=lambda pointer, size: registrations.append((pointer, size))
            or 0,
            unregister_staging=lambda pointer: registrations.append(("unregister", pointer))
            or 0,
            copy_from_staging=lambda destination, source, size: copies.append(
                (destination, source, size)
            ),
        )

        self.assertEqual(engine.reads, [("source:1", [1000, 1004], [101, 105], [4, 8])])
        self.assertEqual(copies, [(201, 1000, 4), (205, 1004, 8)])
        self.assertEqual(registrations, [(1000, 12), ("unregister", 1000)])

    def test_unregisters_staging_when_transfer_fails(self):
        transport = MooncakeExpertRestoreTransport(_RecordingTransferEngine(read_result=-1))
        cleaned = []
        with self.assertRaisesRegex(RuntimeError, "Failed to restore"):
            transport.restore(
                MooncakeExpertSourceDescriptor("source", "source:1", {}),
                [MooncakeRestoreSegment(MooncakeRemoteSlot(1, 4), 2, 4)],
                staging_pointer=100,
                register_staging=lambda *_: 0,
                unregister_staging=lambda pointer: cleaned.append(pointer) or 0,
                copy_from_staging=lambda *_: self.fail("must not copy failed transfer"),
            )
        self.assertEqual(cleaned, [100])

    def test_rejects_mismatched_segment_sizes(self):
        with self.assertRaisesRegex(ValueError, "matching positive"):
            pack_mooncake_restore_segments(
                [MooncakeRestoreSegment(MooncakeRemoteSlot(1, 8), 2, 4)]
            )


if __name__ == "__main__":
    unittest.main()
