"""CPU-only contracts for exact GLM-5-Next block-FP8 layerwise prefill."""

import ast
import importlib.util
import inspect
import subprocess
import sys
import types
import unittest
from collections import namedtuple
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _package(name):
    module = _module(name)
    module.__path__ = []
    return module


def _identity_compile(fn=None, **_kwargs):
    if fn is None:
        return lambda wrapped: wrapped
    return fn


def _load_test_target():
    tp_group = SimpleNamespace(cpu_group=object(), device_group=object(), first_rank=0)
    stubs = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.moe": _package("sglang.srt.layers.moe"),
        "sglang.srt.layers.quantization": _package("sglang.srt.layers.quantization"),
        "sglang.srt.distributed": _module(
            "sglang.srt.distributed",
            get_tensor_model_parallel_rank=lambda: 0,
            get_tensor_model_parallel_world_size=lambda: 1,
            get_tp_group=lambda: tp_group,
        ),
        "sglang.srt.layers.quantization.base_config": _module(
            "sglang.srt.layers.quantization.base_config",
            FusedMoEMethodBase=object,
        ),
        "sglang.srt.layers.quantization.marlin_utils": _module(
            "sglang.srt.layers.quantization.marlin_utils",
            marlin_permute_scales=lambda value, *_args, **_kwargs: value,
        ),
        "sglang.srt.layers.moe.quant_method_registry": _module(
            "sglang.srt.layers.moe.quant_method_registry",
            register_moe_quant_wrapper=lambda *_args, **_kwargs: None,
        ),
        "sglang.srt.utils": _module(
            "sglang.srt.utils",
            get_compiler_backend=lambda: "eager",
            is_cuda=lambda: False,
        ),
        "kt_kernel": _module(
            "kt_kernel",
            KTMoEWrapper=object,
            generate_gpu_experts_masks=lambda *_args, **_kwargs: None,
        ),
    }
    target_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
    )
    module_name = "_kt_ep_wrapper_glm5_next_fp8_test_target"
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    target = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(sys.modules, stubs),
        mock.patch.object(torch, "compile", _identity_compile),
    ):
        sys.modules[module_name] = target
        try:
            spec.loader.exec_module(target)
        finally:
            sys.modules.pop(module_name, None)
    target._test_tp_group = tp_group
    return target


kt_ep_wrapper = _load_test_target()


class _RecordingEvent:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def record(self, stream=None):
        self.log.append(("record", self.name, getattr(stream, "name", stream)))


class _RecordingStream:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait_event(self, event):
        self.log.append(("wait_event", self.name, event.name))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class _StateSlot:
    def __init__(self, index, log):
        self.index = index
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1
        self.has_consumed_event = False
        self.reuse_guard = None
        self.ready_event = _RecordingEvent(f"ready{index}", log)
        self.consumed_event = _RecordingEvent(f"consumed{index}", log)

    def invalidate(self):
        self.state = "EMPTY"
        self.layer_idx = None
        self.epoch = -1


class _StandardCombineInput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


_TopkOutput = namedtuple(
    "_TopkOutput", ["topk_weights", "topk_ids", "token_expert_indices"]
)
_DispatchOutput = namedtuple("_DispatchOutput", ["hidden_states", "topk_output"])


def _runtime_stubs():
    return {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.layers.moe": _package("sglang.srt.layers.moe"),
        "sglang.srt.eplb": _package("sglang.srt.eplb"),
        "sglang.srt.eplb.expert_distribution": _module(
            "sglang.srt.eplb.expert_distribution",
            get_global_expert_distribution_recorder=lambda: mock.Mock(),
        ),
        "sglang.srt.layers.moe.token_dispatcher": _module(
            "sglang.srt.layers.moe.token_dispatcher",
            StandardCombineInput=_StandardCombineInput,
        ),
    }


class TestGlm5NextFp8Gates(unittest.TestCase):
    def setUp(self):
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS.clear()

    @staticmethod
    def _wrapper(*, exact=True, method="FP8", threshold=4096, mode="EXTEND"):
        wrapper = object.__new__(kt_ep_wrapper.KTEPWrapperMethod)
        wrapper.tp_rank = 1
        wrapper.kt_config = SimpleNamespace(
            layer_idx=3,
            method=method,
            is_glm5_next=exact,
            kt_enable_dynamic_expert_update=False,
        )
        wrapper.gpu_prefill_token_threshold = threshold
        wrapper._glm5_next_forward_mode = SimpleNamespace(name=mode)
        wrapper._glm5_next_fp8_pipeline_signature = ("glm", "cuda:0")
        wrapper.gpu_experts_mask = torch.tensor([True])
        wrapper.gpu_experts_mask_cuda = torch.tensor([True])
        wrapper.logical_to_gpu_index_cuda = torch.tensor([0], dtype=torch.int32)
        wrapper.num_gpu_experts = 1
        wrapper._cpu_stream = None
        wrapper.gpu_method = SimpleNamespace(
            apply=lambda _layer, dispatch: _StandardCombineInput(
                torch.full_like(dispatch.hidden_states, 7)
            )
        )
        return wrapper

    @staticmethod
    def _dispatch(num_tokens):
        hidden = torch.zeros((num_tokens, 2))
        topk = _TopkOutput(
            topk_weights=torch.ones((num_tokens, 1)),
            topk_ids=torch.zeros((num_tokens, 1), dtype=torch.long),
            token_expert_indices=None,
        )
        return _DispatchOutput(hidden_states=hidden, topk_output=topk)

    def _apply(self, wrapper, num_tokens):
        with (
            mock.patch.dict(sys.modules, _runtime_stubs()),
            mock.patch.object(
                kt_ep_wrapper,
                "mask_and_remap_expert_ids",
                side_effect=lambda ids, *_args: ids,
            ),
        ):
            return wrapper.apply(SimpleNamespace(), self._dispatch(num_tokens))

    def test_gate_is_exact_glm_block_fp8_only(self):
        self.assertTrue(
            kt_ep_wrapper._glm5_next_fp8_pipeline_requested(self._wrapper())
        )
        self.assertFalse(
            kt_ep_wrapper._glm5_next_fp8_pipeline_requested(self._wrapper(exact=False))
        )
        self.assertFalse(
            kt_ep_wrapper._glm5_next_fp8_pipeline_requested(
                self._wrapper(method="MXFP4")
            )
        )
        self.assertFalse(
            kt_ep_wrapper._glm5_next_fp8_pipeline_requested(self._wrapper(threshold=0))
        )

    def test_every_extend_chunk_uses_required_manager_below_threshold(self):
        wrapper = self._wrapper(threshold=4096, mode="EXTEND")
        manager = mock.Mock()
        manager.apply.return_value = "required-layerwise"
        kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS[
            wrapper._glm5_next_fp8_pipeline_signature
        ] = manager

        result = self._apply(wrapper, num_tokens=17)

        self.assertEqual(result, "required-layerwise")
        manager.apply.assert_called_once()

    def test_decode_and_idle_never_touch_manager(self):
        for mode in ("DECODE", "IDLE"):
            with self.subTest(mode=mode):
                wrapper = self._wrapper(mode=mode)
                manager = mock.Mock()
                kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS[
                    wrapper._glm5_next_fp8_pipeline_signature
                ] = manager

                result = self._apply(wrapper, num_tokens=1)

                manager.apply.assert_not_called()
                manager.abort_round.assert_not_called()
                torch.testing.assert_close(
                    result.hidden_states, torch.full((1, 2), 7.0)
                )

    def test_unsupported_modes_and_missing_manager_fail_closed(self):
        for mode in ("MIXED", "TARGET_VERIFY", "SPLIT_PREFILL", "DLLM_EXTEND"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "supports only plain EXTEND"):
                    self._apply(self._wrapper(mode=mode), num_tokens=8)

        with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
            self._apply(self._wrapper(mode="EXTEND"), num_tokens=8)

    def test_generic_fp8_does_not_enter_glm_manager(self):
        wrapper = self._wrapper(exact=False, mode="EXTEND")
        manager = mock.Mock()
        kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS[
            wrapper._glm5_next_fp8_pipeline_signature
        ] = manager

        result = self._apply(wrapper, num_tokens=1)

        manager.apply.assert_not_called()
        torch.testing.assert_close(result.hidden_states, torch.full((1, 2), 7.0))


class TestGlm5NextFp8Manager(unittest.TestCase):
    def setUp(self):
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS.clear()

    def _manager(self):
        signature = ("glm", "state")
        log = []
        methods = {
            layer_idx: SimpleNamespace(
                kt_config=SimpleNamespace(layer_idx=layer_idx), tp_rank=1
            )
            for layer_idx in range(3, 45)
        }
        layers = {layer_idx: object() for layer_idx in methods}
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY[signature] = {
            layer_idx: (methods[layer_idx], layers[layer_idx]) for layer_idx in methods
        }
        manager = object.__new__(kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager)
        manager.signature = signature
        manager.context = SimpleNamespace(
            gpu_layer=object(),
            gpu_method=SimpleNamespace(
                apply=lambda _layer, dispatch: (
                    log.append(("compute", dispatch)) or dispatch
                )
            ),
        )
        manager.slots = (_StateSlot(0, log), _StateSlot(1, log))
        manager.device = torch.device("cpu")
        manager.epoch = -1
        manager.last_layer_position = None
        manager.current_slot_index = None
        manager.round_active = False
        manager.failure_reason = None
        manager.visited_layer_indices = []
        manager.apply_count = 0
        manager.prime_count = 0
        manager.prefetch_hit_count = 0
        manager.completed_round_count = 0
        manager.fallback_count = 0

        def load_slot(slot, layer_idx, _method, _layer):
            log.append(("load", layer_idx, slot.index, manager.epoch))
            slot.state = "READY"
            slot.layer_idx = layer_idx
            slot.epoch = manager.epoch
            slot.reuse_guard = "ready"

        return manager, methods, layers, log, load_slot

    def test_two_slots_cover_every_moe_layer_and_complete_epoch(self):
        manager, methods, layers, log, load_slot = self._manager()
        stream = _RecordingStream("main", log)
        with (
            mock.patch.object(manager, "_load_slot", side_effect=load_slot),
            mock.patch.object(
                manager,
                "_bind_slot",
                side_effect=lambda slot: log.append(("bind", slot.index)),
            ),
            mock.patch.object(manager, "_record_raw_backing_on_stream"),
            mock.patch.object(manager, "_commit_tp_runtime_phase"),
            mock.patch.object(torch.cuda, "current_stream", return_value=stream),
        ):
            for layer_idx in range(3, 45):
                self.assertEqual(
                    manager.apply(
                        methods[layer_idx], layers[layer_idx], f"layer-{layer_idx}"
                    ),
                    f"layer-{layer_idx}",
                )

        self.assertEqual(manager.apply_count, 42)
        self.assertEqual(manager.prime_count, 1)
        self.assertEqual(manager.prefetch_hit_count, 41)
        self.assertEqual(manager.completed_round_count, 1)
        self.assertEqual(manager.fallback_count, 0)
        self.assertFalse(manager.round_active)
        self.assertEqual(
            [entry[2] for entry in log if entry[0] == "load"],
            [index % 2 for index in range(42)],
        )

    def test_two_consecutive_extend_epochs_keep_monotonic_state(self):
        manager, methods, layers, _log, load_slot = self._manager()
        with (
            mock.patch.object(manager, "_load_slot", side_effect=load_slot),
            mock.patch.object(manager, "_bind_slot"),
            mock.patch.object(manager, "_record_raw_backing_on_stream"),
            mock.patch.object(manager, "_commit_tp_runtime_phase"),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=_RecordingStream("main", []),
            ),
        ):
            for _epoch in range(2):
                for layer_idx in range(3, 45):
                    manager.apply(
                        methods[layer_idx], layers[layer_idx], layer_idx
                    )

        self.assertEqual(manager.epoch, 1)
        self.assertEqual(manager.apply_count, 84)
        self.assertEqual(manager.prime_count, 2)
        self.assertEqual(manager.prefetch_hit_count, 82)
        self.assertEqual(manager.completed_round_count, 2)
        self.assertEqual(manager.fallback_count, 0)
        self.assertFalse(manager.round_active)

    def test_skip_poison_is_sticky_and_never_falls_back(self):
        manager, methods, layers, _log, load_slot = self._manager()
        with (
            mock.patch.object(manager, "_load_slot", side_effect=load_slot),
            mock.patch.object(manager, "_bind_slot"),
            mock.patch.object(manager, "_record_raw_backing_on_stream"),
            mock.patch.object(manager, "_commit_tp_runtime_phase"),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=_RecordingStream("main", []),
            ),
        ):
            manager.apply(methods[3], layers[3], object())
            with self.assertRaisesRegex(RuntimeError, "layer order mismatch"):
                manager.apply(methods[5], layers[5], object())
            with self.assertRaisesRegex(RuntimeError, "manager is poisoned"):
                manager.apply(methods[4], layers[4], object())

        self.assertEqual(manager.fallback_count, 0)

    def test_manager_uses_events_without_global_cuda_synchronize(self):
        target_path = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
        )
        module_source = target_path.read_text(encoding="utf-8")
        tree = ast.parse(module_source)
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_Glm5NextFp8LayerwisePrefillManager"
        )
        source = ast.get_source_segment(module_source, node)
        self.assertNotIn("torch.cuda.synchronize", source)
        self.assertIn("self.transfer_stream", source)
        self.assertIn("__atomic_load_8", source)
        self.assertIn("__atomic_store_8", source)
        self.assertIn("_CONTROL_CACHELINE_BYTES = 64", source)
        self.assertIn("slot.ready_event", source)
        self.assertIn("slot.ready_event.synchronize()", source)
        self.assertIn("slot.consumed_event", source)
        self.assertNotIn("get_tp_group().device_group", source)
        load_source = inspect.getsource(
            kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager._load_slot
        )
        self.assertNotIn("dist.all_reduce", load_source)

    def test_atomic_control_is_cacheline_isolated_and_process_shared(self):
        manager_type = kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        offsets, control_nbytes = manager_type._build_control_offsets(2)
        flat_offsets = sorted(
            offset for group in offsets.values() for offset in group
        )
        self.assertTrue(
            all(
                right - left == manager_type._CONTROL_CACHELINE_BYTES
                for left, right in zip(flat_offsets, flat_offsets[1:])
            )
        )

        payload_offset = control_nbytes
        handle = shared_memory.SharedMemory(
            create=True, size=control_nbytes + 64
        )
        manager = object.__new__(manager_type)
        manager.transport_abandoned = False
        manager._CONTROL_TIMEOUT_SECONDS = 5.0
        try:
            handle.buf[:] = bytes(control_nbytes + 64)
            manager._configure_atomic_access(handle, tp_size=2)
            ready_offset = offsets["ready_generation"][0]
            child_code = r"""
import ctypes
import sys
from multiprocessing import shared_memory

try:
    handle = shared_memory.SharedMemory(
        name=sys.argv[1], create=False, track=False
    )
except TypeError:
    from multiprocessing import resource_tracker

    handle = shared_memory.SharedMemory(name=sys.argv[1], create=False)
    resource_tracker.unregister(handle._name, "shared_memory")
anchor = ctypes.c_char.from_buffer(handle.buf)
address = ctypes.addressof(anchor)
store = getattr(ctypes.CDLL("libatomic.so.1"), "__atomic_store_8")
store.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
store.restype = None
payload_offset = int(sys.argv[3])
handle.buf[payload_offset : payload_offset + 9] = b"published"
store(ctypes.c_void_p(address + int(sys.argv[2])), ctypes.c_uint64(7), 3)
anchor = None
handle.close()
"""
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    handle.name,
                    str(ready_offset),
                    str(payload_offset),
                ]
            )
            manager._wait_for_exact_control_value(
                ready_offset, 7, "child publication"
            )
            self.assertEqual(
                bytes(handle.buf[payload_offset : payload_offset + 9]),
                b"published",
            )
            self.assertEqual(process.wait(timeout=10), 0)
        finally:
            manager._transport_control_anchor = None
            handle.close()
            handle.unlink()

    def test_generation_drift_is_fail_closed(self):
        manager_type = kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        offsets, control_nbytes = manager_type._build_control_offsets(1)
        handle = shared_memory.SharedMemory(create=True, size=control_nbytes)
        manager = object.__new__(manager_type)
        manager.transport_abandoned = False
        try:
            handle.buf[:] = bytes(control_nbytes)
            manager._configure_atomic_access(handle, tp_size=1)
            ready_offset = offsets["ready_generation"][0]
            manager._atomic_store(ready_offset, 4)
            with self.assertRaisesRegex(RuntimeError, "generation drift"):
                manager._wait_for_exact_control_value(
                    ready_offset, 3, "stale consumer"
                )
            self.assertTrue(manager.transport_abandoned)
            self.assertEqual(
                manager._atomic_load(offsets["global_error"][0]), 1
            )
        finally:
            manager._transport_control_anchor = None
            handle.close()
            handle.unlink()

    def test_sticky_global_error_immediately_aborts_generation_wait(self):
        manager_type = kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        offsets, control_nbytes = manager_type._build_control_offsets(1)
        handle = shared_memory.SharedMemory(create=True, size=control_nbytes)
        manager = object.__new__(manager_type)
        manager.transport_abandoned = False
        manager._CONTROL_TIMEOUT_SECONDS = 60.0
        try:
            handle.buf[:] = bytes(control_nbytes)
            manager._configure_atomic_access(handle, tp_size=1)
            manager._atomic_store(offsets["global_error"][0], 1)
            with self.assertRaisesRegex(RuntimeError, "aborted by another"):
                manager._wait_for_exact_control_value(
                    offsets["ready_generation"][0],
                    1,
                    "failed writer publication",
                )
            self.assertTrue(manager.transport_abandoned)
        finally:
            manager._transport_control_anchor = None
            handle.close()
            handle.unlink()

    def test_writer_descriptor_mismatch_poison_is_sticky(self):
        manager_type = kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        _offsets, control_nbytes = manager_type._build_control_offsets(1)
        handle = shared_memory.SharedMemory(create=True, size=control_nbytes)
        manager = object.__new__(manager_type)
        manager.transport_abandoned = False
        try:
            handle.buf[:] = bytes(control_nbytes)
            manager._configure_atomic_access(handle, tp_size=1)
            manager._publish_writer_generation(
                host_slot=0,
                generation=1,
                layer_idx=3,
                expert_id=17,
                writer_ok=True,
            )
            with self.assertRaisesRegex(RuntimeError, "descriptor mismatch"):
                manager._consume_writer_generation(
                    host_slot=0,
                    generation=1,
                    layer_idx=3,
                    expert_id=18,
                )
            self.assertTrue(manager.transport_abandoned)
            self.assertTrue(manager._shared_transport_failed())
        finally:
            manager._transport_control_anchor = None
            handle.close()
            handle.unlink()

    def test_tp_contract_rejects_peer_mask_or_mapping_digest_drift(self):
        signature = ("glm", "contract")
        original_layer = SimpleNamespace(
            **{
                name: torch.empty(1)
                for name in kt_ep_wrapper._Glm5NextFp8PrefillSlot.RAW_NAMES
            }
        )
        method = SimpleNamespace(
            gpu_experts_mask=torch.tensor([True, False]),
            logical_to_gpu_index=torch.tensor([0, -1], dtype=torch.int32),
            num_gpu_experts=1,
            global_num_experts=2,
            _full_init_args=(4, 2, torch.bfloat16),
            kt_config=SimpleNamespace(weight_path="weights"),
        )
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY[signature] = {
            3: (method, original_layer)
        }
        manager = object.__new__(
            kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        )
        manager.signature = signature
        manager.device = torch.device("cpu")
        manager.slots = (
            SimpleNamespace(
                **{
                    name: torch.empty(1)
                    for name in kt_ep_wrapper._Glm5NextFp8PrefillSlot.RAW_NAMES
                }
            ),
        )

        def inject_peer_drift(gathered, local, **_kwargs):
            gathered[:] = [
                local,
                (local[0], "different-digest", local[2], None),
            ]

        with (
            mock.patch.object(
                kt_ep_wrapper.dist, "is_initialized", return_value=True
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            mock.patch.object(
                kt_ep_wrapper.dist,
                "all_gather_object",
                side_effect=inject_peer_drift,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                manager.validate_tp_contract()

    def test_tp_contract_rejects_processed_or_repacked_original_gpu_tensor(self):
        signature = ("glm", "raw-original")
        names = kt_ep_wrapper._Glm5NextFp8PrefillSlot.RAW_NAMES
        original_layer = SimpleNamespace(
            **{name: torch.empty(1) for name in names}
        )
        original_layer.w13_weight = torch.empty(1, dtype=torch.int32)
        method = SimpleNamespace(
            gpu_experts_mask=torch.tensor([True, False]),
            logical_to_gpu_index=torch.tensor([0, -1], dtype=torch.int32),
            num_gpu_experts=1,
            global_num_experts=2,
            _full_init_args=(4, 2, torch.bfloat16),
            kt_config=SimpleNamespace(weight_path="weights"),
        )
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY[signature] = {
            3: (method, original_layer)
        }
        manager = object.__new__(
            kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
        )
        manager.signature = signature
        manager.device = torch.device("cpu")
        manager.slots = (
            SimpleNamespace(**{name: torch.empty(1) for name in names}),
        )

        with self.assertRaisesRegex(RuntimeError, "failed to build") as raised:
            manager.validate_tp_contract()
        self.assertIn("dtype mismatch", str(raised.exception.__cause__))

    def test_transport_control_disables_resource_tracking_when_supported(self):
        opener = mock.Mock(return_value="untracked")
        with mock.patch.object(kt_ep_wrapper.shared_memory, "SharedMemory", opener):
            result = (
                kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
                ._open_transport_shared_memory(name="control", create=False)
            )

        self.assertEqual(result, "untracked")
        opener.assert_called_once_with(
            name="control", create=False, track=False
        )

    def test_transport_control_tracking_fallback_is_for_old_python_only(self):
        opener = mock.Mock(side_effect=[TypeError("no track"), "tracked"])
        with mock.patch.object(kt_ep_wrapper.shared_memory, "SharedMemory", opener):
            result = (
                kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager
                ._open_transport_shared_memory(name="control", create=False)
            )

        self.assertEqual(result, "tracked")
        self.assertEqual(
            opener.call_args_list,
            [
                mock.call(name="control", create=False, track=False),
                mock.call(name="control", create=False),
            ],
        )

    def test_bind_selects_only_the_four_stable_raw_tensors(self):
        names = kt_ep_wrapper._Glm5NextFp8PrefillSlot.RAW_NAMES
        old = {name: torch.nn.Parameter(torch.zeros(1)) for name in names}
        new = {name: torch.nn.Parameter(torch.ones(1)) for name in names}
        layer = SimpleNamespace(**old)
        manager = object.__new__(kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager)
        manager.context = SimpleNamespace(gpu_layer=layer)

        manager._bind_slot(SimpleNamespace(**new))

        for name in names:
            self.assertIs(getattr(layer, name), new[name])
        self.assertFalse(hasattr(layer, "_v4_marlin_weights"))

    def test_writer_abi_uses_fp8_and_fp32_host_slot_offsets(self):
        class HostBuffer:
            def __init__(self, numel, element_size):
                self._numel = numel
                self._element_size = element_size

            def numel(self):
                return self._numel

            def element_size(self):
                return self._element_size

        names = kt_ep_wrapper._Glm5NextFp8PrefillSlot.RAW_NAMES
        buffers = {
            names[0]: HostBuffer(200, 1),
            names[1]: HostBuffer(80, 4),
            names[2]: HostBuffer(120, 1),
            names[3]: HostBuffer(40, 4),
        }
        pointers = {
            name: [1000 + index * 100, 2000 + index * 100]
            for index, name in enumerate(names)
        }
        manager = object.__new__(kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager)
        manager.context = SimpleNamespace(
            cpu_buffers=buffers, all_rank_buffer_ptrs=pointers
        )
        wrapper = mock.Mock()
        method = SimpleNamespace(wrapper=wrapper)

        with mock.patch.object(
            kt_ep_wrapper, "get_tensor_model_parallel_world_size", return_value=2
        ):
            manager._submit_host_write(method, expert_id=17, host_slot=1)

        args = wrapper.submit_write_weight_scale_to_buffer.call_args.args
        self.assertEqual(args[:2], (2, 17))
        for position, name in enumerate(names, 2):
            offset = buffers[name].numel() // 2 * buffers[name].element_size()
            self.assertEqual(
                args[position], [pointer + offset for pointer in pointers[name]]
            )
        wrapper.sync_write_weight_scale_to_buffer.assert_called_once_with()


class TestGlm5NextFp8AllocationContracts(unittest.TestCase):
    def setUp(self):
        kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS.clear()

    def test_exact_two_slot_capacity_is_reserved_before_kv_profile(self):
        one_slot = kt_ep_wrapper._glm5_next_fp8_raw_slot_storage_nbytes(
            num_experts=288,
            hidden_size=4096,
            intermediate_size=256,
        )
        self.assertEqual(one_slot, 906_190_848)
        self.assertEqual(2 * one_slot, 1_812_381_696)

        model_runner_source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/model_executor/model_runner.py"
        ).read_text(encoding="utf-8")
        allocation_pos = model_runner_source.index(
            "initialize_glm5_next_fp8_layerwise_prefill()"
        )
        pool_pos = model_runner_source.index("self.init_memory_pool()", allocation_pos)
        self.assertLess(allocation_pos, pool_pos)

    def test_registry_requires_all_42_exact_checkpoint_moe_layers(self):
        mask = torch.zeros(288, dtype=torch.bool)
        mask[:4] = True
        mapping = torch.full((288,), -1, dtype=torch.int32)
        mapping[:4] = torch.arange(4, dtype=torch.int32)
        method = SimpleNamespace(
            kt_config=SimpleNamespace(
                is_glm5_next=True,
                method="FP8",
                num_layers=45,
                kt_enable_dynamic_expert_update=False,
            ),
            gpu_prefill_token_threshold=1,
            tp_rank=0,
            global_num_experts=288,
            _full_init_args=(4096, 256, torch.bfloat16),
            gpu_experts_mask=mask,
            logical_to_gpu_index=mapping,
            num_gpu_experts=4,
        )
        complete = {idx: (method, object()) for idx in range(3, 45)}
        kt_ep_wrapper._validate_glm5_next_fp8_registry(("glm",), complete)

        incomplete = dict(complete)
        incomplete.pop(44)
        with self.assertRaisesRegex(RuntimeError, "every MoE layer 3..44"):
            kt_ep_wrapper._validate_glm5_next_fp8_registry(("glm",), incomplete)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_registry_validates_cuda_resident_mask_and_mapping(self):
        device = torch.device("cuda", torch.cuda.current_device())
        mask = torch.zeros(288, dtype=torch.bool, device=device)
        mask[:4] = True
        mapping = torch.full((288,), -1, dtype=torch.int32, device=device)
        mapping[:4] = torch.arange(4, dtype=torch.int32, device=device)
        method = SimpleNamespace(
            kt_config=SimpleNamespace(
                is_glm5_next=True,
                method="FP8",
                num_layers=45,
                kt_enable_dynamic_expert_update=False,
            ),
            gpu_prefill_token_threshold=1,
            tp_rank=0,
            global_num_experts=288,
            _full_init_args=(4096, 256, torch.bfloat16),
            gpu_experts_mask=mask,
            logical_to_gpu_index=mapping,
            num_gpu_experts=4,
        )
        complete = {idx: (method, object()) for idx in range(3, 45)}

        kt_ep_wrapper._validate_glm5_next_fp8_registry(("glm",), complete)

    def test_context_accepts_only_raw_e4m3_and_fp32_128_scales(self):
        class TensorMetadata:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.dtype = dtype

        layer = SimpleNamespace(
            w13_weight=TensorMetadata((288, 512, 4096), torch.float8_e4m3fn),
            w13_weight_scale_inv=TensorMetadata((288, 4, 32), torch.float32),
            w2_weight=TensorMetadata((288, 4096, 256), torch.float8_e4m3fn),
            w2_weight_scale_inv=TensorMetadata((288, 32, 2), torch.float32),
            moe_runner_config=SimpleNamespace(
                glm5_next_hf_two_round_swiglu=True
            ),
        )
        context = SimpleNamespace(
            _is_fp8_quant=True,
            _get_base_quant_method=lambda: SimpleNamespace(
                block_quant=True,
                use_mxfp8=False,
                weight_block_size=[128, 128],
                runner=SimpleNamespace(
                    runner_backend=SimpleNamespace(is_triton=lambda: True)
                ),
            ),
            gpu_layer=layer,
        )
        kt_ep_wrapper._validate_glm5_next_fp8_context(context)

        layer.moe_runner_config.glm5_next_hf_two_round_swiglu = False
        with self.assertRaisesRegex(RuntimeError, "private HF two-round"):
            kt_ep_wrapper._validate_glm5_next_fp8_context(context)
        layer.moe_runner_config.glm5_next_hf_two_round_swiglu = True

        layer.w2_weight_scale_inv.dtype = torch.bfloat16
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            kt_ep_wrapper._validate_glm5_next_fp8_context(context)

        layer.w2_weight_scale_inv.dtype = torch.float32
        context._get_base_quant_method = lambda: SimpleNamespace(
            block_quant=True,
            use_mxfp8=False,
            weight_block_size=[128, 128],
            runner=SimpleNamespace(
                runner_backend=SimpleNamespace(is_triton=lambda: False)
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "Triton MoE runner"):
            kt_ep_wrapper._validate_glm5_next_fp8_context(context)

    def test_tp0_writer_abi_is_probed_before_serving(self):
        complete_wrapper = SimpleNamespace(
            submit_write_weight_scale_to_buffer=lambda *_args: None,
            sync_write_weight_scale_to_buffer=lambda: None,
            moe=SimpleNamespace(write_weight_scale_to_buffer_task=object()),
        )
        registry = {
            3: (SimpleNamespace(wrapper=complete_wrapper), object())
        }
        with mock.patch.object(
            kt_ep_wrapper, "get_tensor_model_parallel_rank", return_value=0
        ):
            kt_ep_wrapper._validate_glm5_next_fp8_writer_abi(registry)

            incomplete = {
                3: (
                    SimpleNamespace(
                        wrapper=SimpleNamespace(
                            submit_write_weight_scale_to_buffer=lambda *_args: None,
                            sync_write_weight_scale_to_buffer=lambda: None,
                            moe=object(),
                        )
                    ),
                    object(),
                )
            }
            with self.assertRaisesRegex(
                RuntimeError, "write_weight_scale_to_buffer_task"
            ):
                kt_ep_wrapper._validate_glm5_next_fp8_writer_abi(incomplete)

        with mock.patch.object(
            kt_ep_wrapper, "get_tensor_model_parallel_rank", return_value=1
        ):
            kt_ep_wrapper._validate_glm5_next_fp8_writer_abi(incomplete)

    def test_startup_oom_is_fatal_and_does_not_register_manager(self):
        signature = ("glm", "oom")
        mask = torch.zeros(288, dtype=torch.bool)
        mapping = torch.full((288,), -1, dtype=torch.int32)
        method = SimpleNamespace(
            kt_config=SimpleNamespace(
                is_glm5_next=True,
                method="FP8",
                num_layers=45,
                kt_enable_dynamic_expert_update=False,
            ),
            gpu_prefill_token_threshold=1,
            _full_init_args=(4096, 256, torch.bfloat16),
            global_num_experts=288,
            moe_runner_config=object(),
            tp_rank=0,
            gpu_experts_mask=mask,
            logical_to_gpu_index=mapping,
            num_gpu_experts=0,
        )
        registry = {idx: (method, object()) for idx in range(3, 45)}
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=8,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "SharedFullContext",
                side_effect=torch.cuda.OutOfMemoryError("slot 0"),
            ) as context_ctor,
            mock.patch.object(torch.cuda, "empty_cache"),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing.*fallback"):
                kt_ep_wrapper._initialize_glm5_next_fp8_layerwise_pipeline(
                    signature, registry
                )

        context_ctor.assert_called_once()
        self.assertNotIn(signature, kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS)

    def test_glm_manager_and_registry_are_not_mxfp4_state(self):
        self.assertIsNot(
            kt_ep_wrapper._GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY,
            kt_ep_wrapper._MXFP4_PREFILL_LAYER_REGISTRY,
        )
        self.assertIsNot(
            kt_ep_wrapper._GLM5_NEXT_FP8_LAYERWISE_MANAGERS,
            kt_ep_wrapper._MXFP4_LAYERWISE_MANAGERS,
        )
        self.assertFalse(
            issubclass(
                kt_ep_wrapper._Glm5NextFp8LayerwisePrefillManager,
                kt_ep_wrapper._Mxfp4LayerwisePrefillManager,
            )
        )

    def test_initializer_has_no_lazy_or_disabled_fallback_branch(self):
        source = inspect.getsource(
            kt_ep_wrapper._initialize_glm5_next_fp8_layerwise_pipeline
        )
        self.assertNotIn("_MXFP4", source)
        self.assertNotIn("DISABLED_REASONS", source)
        self.assertIn("refusing hybrid/serialized fallback", source)


if __name__ == "__main__":
    unittest.main()
