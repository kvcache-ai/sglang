"""Execute the FP8 transport loop with deliberately delayed, simulated DMA readers.

The real method is compiled in isolation to avoid GPU/native imports. Only its
external CUDA, collective and native-writer boundaries are replaced. Queued
copies retain references to the host slots until an event/device synchronization
completes them, so premature slot reuse is an observable state violation.
"""

import ast
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
)
WEIGHTS = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")


class HostBuffer:
    def __init__(self, rank):
        self.rank = rank

    def __getitem__(self, slot):
        return self.rank, slot

    def numel(self):
        return 2

    def element_size(self):
        return 1


class DeviceBuffer:
    device = "simulated-cuda"

    def __init__(self, runtime):
        self.runtime = runtime

    def __getitem__(self, expert):
        def copy(source, non_blocking):
            assert non_blocking
            self.runtime.local.stream.copies.append((source, expert))

        return SimpleNamespace(copy_=copy)


class Stream:
    def __init__(self, runtime):
        self.runtime = runtime
        self.copies = []
        self.completed = 0

    def complete(self, end):
        for (rank, slot), expected in self.copies[self.completed : end]:
            actual = self.runtime.slots[rank][slot]
            if actual != expected:
                raise AssertionError(
                    f"rank {rank}: copied expert {actual}, expected {expected}"
                )
        self.completed = max(self.completed, end)

    def wait_event(self, event):
        # GPU-side waits enqueue a dependency; they do not block the CPU writer.
        pass

    def wait_stream(self, stream):
        pass


class Event:
    def __init__(self, runtime):
        self.runtime = runtime

    def record(self, stream):
        self.stream = stream
        self.end = len(stream.copies)
        expert = stream.copies[-1][1]
        self.runtime.recorded[self.runtime.local.rank][expert].set()

    def synchronize(self):
        self.stream.complete(self.end)


class Runtime:
    def __init__(self, ranks, experts):
        self.ranks = ranks
        self.experts = experts
        self.local = threading.local()
        self.slots = [[None, None] for _ in range(ranks)]
        self.streams = [[] for _ in range(ranks)]
        self.recorded = [
            [threading.Event() for _ in range(experts)] for _ in range(ranks)
        ]
        self.barrier = threading.Barrier(ranks, timeout=5)
        self.group = object()

    def new_stream(self, device):
        stream = Stream(self)
        self.streams[self.local.rank].append(stream)
        return stream

    @contextmanager
    def use_stream(self, stream):
        self.local.stream = stream
        yield

    def synchronize(self, device):
        for stream in self.streams[self.local.rank]:
            stream.complete(len(stream.copies))

    def collective(self, group):
        assert group is self.group
        self.barrier.wait()

    def write(self, world_size, expert, w13_ptrs, *other_ptrs):
        assert world_size == self.ranks
        slot = w13_ptrs[0]
        if self.ranks > 1 and expert >= 2:
            # Make the negative control deterministic: the peer's earlier copy
            # is queued but not completed when rank zero tries to reuse its slot.
            assert self.recorded[1][expert - 2].wait(timeout=5)
        for rank, streams in enumerate(self.streams):
            for stream in streams:
                if any(
                    src == (rank, slot) for src, _ in stream.copies[stream.completed :]
                ):
                    raise AssertionError(
                        f"writer reused slot {slot} before rank {rank} finished"
                    )
        for rank in range(self.ranks):
            self.slots[rank][slot] = expert

    def method(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SharedFullContext"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_weight_fp8"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        isolated = ast.fix_missing_locations(
            ast.Module(body=[future, method], type_ignores=[])
        )
        namespace = {
            "os": SimpleNamespace(
                cpu_count=lambda: self.ranks, sched_setaffinity=lambda *_: None
            ),
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    Stream=self.new_stream,
                    Event=lambda: Event(self),
                    stream=self.use_stream,
                    synchronize=self.synchronize,
                    current_stream=lambda _: Stream(self),
                )
            ),
            "dist": SimpleNamespace(
                is_initialized=lambda: self.ranks > 1, barrier=self.collective
            ),
            "get_tensor_model_parallel_rank": lambda: self.local.rank,
            "get_tensor_model_parallel_world_size": lambda: self.ranks,
            "get_tp_group": lambda: SimpleNamespace(device_group=self.group),
        }
        exec(compile(isolated, str(SOURCE), "exec"), namespace)
        return namespace["_prepare_weight_fp8"]

    def run(self, layers):
        method = self.method()
        errors = []

        def rank_main(rank):
            self.local.rank = rank
            layer = SimpleNamespace(
                num_experts=self.experts,
                **{name: DeviceBuffer(self) for name in WEIGHTS},
            )
            context = SimpleNamespace(
                gpu_layer=layer,
                WEIGHT_NAMES_FP8=WEIGHTS,
                cpu_buffers={name: HostBuffer(rank) for name in WEIGHTS},
                all_rank_buffer_ptrs={name: [0] * self.ranks for name in WEIGHTS},
            )
            writer = SimpleNamespace(
                submit_write_weight_scale_to_buffer=self.write,
                sync_write_weight_scale_to_buffer=lambda: None,
            )
            try:
                for _ in range(layers):
                    method(context, writer if rank == 0 else None)
                self.synchronize("simulated-cuda")
            except Exception as error:
                errors.append(error)
                self.barrier.abort()

        threads = [
            threading.Thread(target=rank_main, args=(rank,))
            for rank in range(self.ranks)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            self.barrier.abort()
            raise AssertionError("transport schedule deadlocked")
        if errors:
            raise errors[0]


class TestFP8StagingReaders(unittest.TestCase):
    def test_tp2_slot_reuse(self):
        for experts in (3, 4, 5):
            with self.subTest(experts=experts):
                Runtime(ranks=2, experts=experts).run(layers=1)

    def test_tp2_next_layer_drains_prior_readers(self):
        for experts in (1, 2, 5):
            with self.subTest(experts=experts):
                Runtime(ranks=2, experts=experts).run(layers=2)

    def test_tp1_remains_collective_free(self):
        Runtime(ranks=1, experts=5).run(layers=1)


if __name__ == "__main__":
    unittest.main()
