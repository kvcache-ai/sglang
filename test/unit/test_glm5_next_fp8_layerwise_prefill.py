"""CPU-only contracts for GLM-5-Next's generic FP8 prefill path."""

import contextlib
import importlib.util
import inspect
import math
import sys
import threading
import types
import unittest
from collections import namedtuple
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
        "sglang.srt.layers.quantization": _package(
            "sglang.srt.layers.quantization"
        ),
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


class _GpuMethod:
    def __init__(self, fill_value):
        self.fill_value = fill_value
        self.calls = []

    def apply(self, layer, dispatch_output):
        self.calls.append((layer, dispatch_output))
        return _StandardCombineInput(
            torch.full_like(dispatch_output.hidden_states, self.fill_value)
        )


class TestGlm5NextFp8GenericRouting(unittest.TestCase):
    def setUp(self):
        kt_ep_wrapper._SHARED_FULL_CONTEXT = None
        kt_ep_wrapper._MXFP4_LAYERWISE_MANAGERS.clear()
        kt_ep_wrapper._MXFP4_LAYERWISE_DISABLED_REASONS.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_NATIVE_PREFETCH_LAYER_REGISTRY.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_NATIVE_PREFETCH_MANAGERS.clear()

    @staticmethod
    def _wrapper(*, exact=True, threshold=1024, mode="EXTEND"):
        wrapper = object.__new__(kt_ep_wrapper.KTEPWrapperMethod)
        wrapper.tp_rank = 1
        wrapper.kt_config = SimpleNamespace(
            layer_idx=3,
            method="FP8",
            is_glm5_next=exact,
            kt_enable_dynamic_expert_update=False,
        )
        wrapper.gpu_prefill_token_threshold = threshold
        if mode is not None:
            wrapper._glm5_next_forward_mode = SimpleNamespace(name=mode)
        wrapper._glm5_next_has_image_inputs = False
        wrapper.gpu_experts_mask = torch.tensor([True])
        wrapper.gpu_experts_mask_cuda = torch.tensor([True])
        wrapper.logical_to_gpu_index = torch.tensor([0], dtype=torch.int32)
        wrapper.logical_to_gpu_index_cuda = torch.tensor([0], dtype=torch.int32)
        wrapper.num_gpu_experts = 1
        wrapper._cpu_stream = None
        wrapper.gpu_method = _GpuMethod(fill_value=7)
        wrapper._full_init_args = (4096, 512, torch.bfloat16)
        wrapper.global_num_experts = 288
        wrapper.moe_runner_config = object()
        wrapper.wrapper = object()
        return wrapper

    @staticmethod
    def _layer():
        return SimpleNamespace(
            w13_weight=object(),
            w13_weight_scale_inv=object(),
            w2_weight=object(),
            w2_weight_scale_inv=object(),
        )

    @staticmethod
    def _dispatch(num_tokens):
        hidden = torch.zeros((num_tokens, 2))
        topk = _TopkOutput(
            topk_weights=torch.ones((num_tokens, 1)),
            topk_ids=torch.zeros((num_tokens, 1), dtype=torch.long),
            token_expert_indices=None,
        )
        return _DispatchOutput(hidden_states=hidden, topk_output=topk)

    def _apply(self, wrapper, layer, num_tokens):
        with (
            mock.patch.dict(sys.modules, _runtime_stubs()),
            mock.patch.object(
                kt_ep_wrapper,
                "mask_and_remap_expert_ids",
                side_effect=lambda ids, *_args: ids,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            return wrapper.apply(layer, self._dispatch(num_tokens))

    @staticmethod
    def _generic_context(fill_value=11):
        return SimpleNamespace(
            gpu_layer=object(),
            gpu_method=_GpuMethod(fill_value=fill_value),
            load=mock.Mock(),
            initialize_cpu_buffers=mock.Mock(),
            _initialize_glm5_next_fp8_transport=mock.Mock(),
            _cleanup_cpu_buffers_after_failure=mock.Mock(),
            _is_mxfp4_quant=False,
        )

    def test_1023_is_hybrid_and_1024_lazily_builds_one_shared_context(self):
        wrapper = self._wrapper()
        layer = self._layer()
        context = self._generic_context()

        with (
            mock.patch.object(
                kt_ep_wrapper, "SharedFullContext", return_value=context
            ) as context_ctor,
            mock.patch.object(
                kt_ep_wrapper,
                "_validate_glm5_next_fp8_shared_full_context",
                return_value=(1_812_381_696, 12_585_984),
            ) as validate_context,
        ):
            below = self._apply(wrapper, layer, 1023)
            context_ctor.assert_not_called()
            context.load.assert_not_called()
            validate_context.assert_not_called()

            at_threshold = self._apply(wrapper, layer, 1024)
            second_layer = self._layer()
            wrapper.kt_config.layer_idx = 4
            second = self._apply(wrapper, second_layer, 1024)

        torch.testing.assert_close(
            below.hidden_states, torch.full((1023, 2), 7.0)
        )
        torch.testing.assert_close(
            at_threshold.hidden_states, torch.full((1024, 2), 11.0)
        )
        torch.testing.assert_close(
            second.hidden_states, torch.full((1024, 2), 11.0)
        )
        context_ctor.assert_called_once_with(
            layer=layer,
            init_args=(4096, 512, torch.bfloat16),
            global_num_experts=288,
            moe_runner_config=wrapper.moe_runner_config,
            defer_cpu_buffers=True,
        )
        context.initialize_cpu_buffers.assert_called_once_with()
        context._initialize_glm5_next_fp8_transport.assert_called_once_with(
            wrapper=wrapper.wrapper,
            num_gpu_experts=wrapper.num_gpu_experts,
        )
        self.assertIs(kt_ep_wrapper._SHARED_FULL_CONTEXT, context)
        validate_context.assert_called_once_with(context)
        self.assertEqual(context.load.call_count, 2)
        self.assertEqual(
            context.load.call_args_list,
            [
                mock.call(
                    layer_idx=3,
                    wrapper=wrapper.wrapper,
                    original_layer=layer,
                    gpu_experts_mask=wrapper.gpu_experts_mask,
                    logical_to_gpu_index=wrapper.logical_to_gpu_index,
                ),
                mock.call(
                    layer_idx=4,
                    wrapper=wrapper.wrapper,
                    original_layer=second_layer,
                    gpu_experts_mask=wrapper.gpu_experts_mask,
                    logical_to_gpu_index=wrapper.logical_to_gpu_index,
                ),
            ],
        )

    def test_image_extend_and_decode_idle_bypass_generic_context(self):
        cases = (("EXTEND", True), ("DECODE", False), ("IDLE", False))
        for mode, has_image in cases:
            with self.subTest(mode=mode, has_image=has_image):
                kt_ep_wrapper._SHARED_FULL_CONTEXT = None
                wrapper = self._wrapper(mode=mode)
                wrapper._glm5_next_has_image_inputs = has_image
                context = self._generic_context()
                with mock.patch.object(
                    kt_ep_wrapper, "SharedFullContext", return_value=context
                ) as context_ctor:
                    result = self._apply(wrapper, self._layer(), 1024)

                context_ctor.assert_not_called()
                context.load.assert_not_called()
                torch.testing.assert_close(
                    result.hidden_states, torch.full((1024, 2), 7.0)
                )
                if has_image:
                    self.assertEqual(wrapper._glm5_next_mm_hybrid_extend_count, 1)

    def test_single_rank_slot_oom_fails_collectively_before_host_buffers(self):
        wrapper = self._wrapper()
        layer = self._layer()

        with (
            mock.patch.object(
                kt_ep_wrapper,
                "SharedFullContext",
                side_effect=torch.cuda.OutOfMemoryError("test slot OOM"),
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                wrapper._build_full_context(layer)

        self.assertIsNone(kt_ep_wrapper._SHARED_FULL_CONTEXT)

    def test_cross_rank_validation_failure_cleans_host_buffers(self):
        wrapper = self._wrapper()
        context = self._generic_context()

        with (
            mock.patch.object(
                kt_ep_wrapper, "SharedFullContext", return_value=context
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_validate_glm5_next_fp8_shared_full_context",
                return_value=(1_812_381_696, 12_585_984),
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_all_tp_ranks_succeeded",
                side_effect=(True, False),
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                wrapper._build_full_context(self._layer())

        context.initialize_cpu_buffers.assert_called_once_with()
        context._cleanup_cpu_buffers_after_failure.assert_called_once_with()
        self.assertIsNone(kt_ep_wrapper._SHARED_FULL_CONTEXT)

    def test_missing_and_unsupported_forward_modes_fail_before_allocation(self):
        for mode, pattern in (
            (None, "ForwardMode metadata"),
            ("MIXED", "supports only plain EXTEND"),
        ):
            with self.subTest(mode=mode):
                kt_ep_wrapper._SHARED_FULL_CONTEXT = None
                wrapper = self._wrapper(mode=mode)
                context = self._generic_context()
                with mock.patch.object(
                    kt_ep_wrapper, "SharedFullContext", return_value=context
                ) as context_ctor:
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        self._apply(wrapper, self._layer(), 1024)

                context_ctor.assert_not_called()
                context.load.assert_not_called()

    def test_non_glm_keeps_the_existing_generic_threshold_behavior(self):
        wrapper = self._wrapper(exact=False, mode=None)
        layer = self._layer()
        context = self._generic_context(fill_value=13)

        with mock.patch.object(
            kt_ep_wrapper, "SharedFullContext", return_value=context
        ) as context_ctor:
            below = self._apply(wrapper, layer, 1023)
            at_threshold = self._apply(wrapper, layer, 1024)

        torch.testing.assert_close(
            below.hidden_states, torch.full((1023, 2), 7.0)
        )
        torch.testing.assert_close(
            at_threshold.hidden_states, torch.full((1024, 2), 13.0)
        )
        context_ctor.assert_called_once()
        context.load.assert_called_once()


class _TensorMetadata:
    def __init__(self, shape, dtype):
        self.shape = torch.Size(shape)
        self.dtype = dtype

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()


class _FakeSharedMemory:
    instances = []

    def __init__(self, *, name, create, size):
        self.name = name
        self.create = create
        self.size = size
        self.buf = bytearray(size)
        self.unlinked = False
        self.closed = False
        self.instances.append(self)

    def unlink(self):
        self.unlinked = True

    def close(self):
        self.closed = True


class _FakeCudaEvent:
    def __init__(self):
        self.record_count = 0
        self.synchronize_count = 0

    def record(self, _stream):
        self.record_count += 1

    def synchronize(self):
        self.synchronize_count += 1


class _FakeCudaStream:
    def __init__(self):
        self.synchronize_count = 0
        self.waited_streams = []

    def synchronize(self):
        self.synchronize_count += 1

    def wait_stream(self, stream):
        self.waited_streams.append(stream)


class _FakeFp8Writer:
    def __init__(self, context):
        self.context = context
        self.pending = None
        self.submitted = []
        self.sync_count = 0

    def submit_write_weight_scale_to_buffer(
        self,
        tp_size,
        expert_id,
        w13_weight_ptrs,
        w13_scale_ptrs,
        w2_weight_ptrs,
        w2_scale_ptrs,
    ):
        self.submitted.append((tp_size, expert_id))
        base = self.context.cpu_buffers["w13_weight"].data_ptr()
        per_slot_bytes = (
            self.context.cpu_buffers["w13_weight"].numel()
            // 2
            * self.context.cpu_buffers["w13_weight"].element_size()
        )
        slot = (w13_weight_ptrs[0] - base) // per_slot_bytes
        self.pending = (expert_id, slot)

    def sync_write_weight_scale_to_buffer(self):
        expert_id, slot = self.pending
        for offset, name in enumerate(
            kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8
        ):
            self.context.cpu_buffers[name][slot].fill_(expert_id + offset * 10)
        self.sync_count += 1
        self.pending = None


class _FakeNativeTransport:
    def __init__(self, args, calls):
        self.args = args
        self.calls = calls
        self.last_join = None
        self.close_count = 0

    def join(self, epoch, layer_id, expert_count):
        self.last_join = (epoch, layer_id, expert_count)
        self.calls.append(("join", *self.last_join))

    def wait(self, epoch):
        self.calls.append(("wait", epoch))
        joined_epoch, layer_id, expert_count = self.last_join
        return {
            "epoch": joined_epoch,
            "layer_id": layer_id,
            "expert_count": expert_count,
            "rank": 0,
            "writer_ms": 3.0,
            "slot_wait_ms": 1.0,
            "h2d_ms": 2.0,
            "total_ms": 4.0,
            "bytes": 64,
            "poisoned": False,
            "error_code": 0,
            "error_rank": -1,
        }

    def close(self):
        self.close_count += 1
        self.calls.append(("close",))


class _FakeNativeExtension:
    def __init__(self, calls):
        self.calls = calls
        self.initialize_args = None
        self.transport = None

    def initialize_fp8_layerwise_control(self, control_ptr, control_size, tp_size):
        self.initialize_args = (control_ptr, control_size, tp_size)
        self.calls.append(("initialize", control_size, tp_size))

    def FP8LayerwiseTransport(self, *args):
        self.transport = _FakeNativeTransport(args, self.calls)
        self.calls.append(("construct",))
        return self.transport


class _FakeNativeWriter:
    def __init__(self, calls, fail=False):
        self.calls = calls
        self.fail = fail
        self.moe = self
        self.cpu_infer = SimpleNamespace(sync=mock.Mock())
        self.public_run_count = 0

    def run_layerwise_fp8_batch(
        self, transport, epoch, layer_id, expert_count
    ):
        self.calls.append(("run", epoch, layer_id, expert_count))
        if self.fail:
            raise RuntimeError("injected producer failure")


class _FakePrefetchEvent:
    def __init__(self, calls):
        self.calls = calls
        self.recorded = threading.Event()

    def record(self, stream):
        self.calls.append(("consumed-record", stream))
        self.recorded.set()

    def synchronize(self):
        self.calls.append(("consumed-wait",))
        if not self.recorded.wait(timeout=1):
            raise RuntimeError("consumed event was never recorded")


class _FakePrefetchGpuMethod:
    def __init__(self, calls):
        self.calls = calls

    def process_weights_after_loading(self, _layer):
        self.calls.append(("process",))

    def apply(self, _layer, dispatch_output):
        layer_idx = dispatch_output.layer_idx
        self.calls.append(("apply", layer_idx))
        return f"result-{layer_idx}"


class _FakePrefetchContext:
    def __init__(self, calls, fail_layer=None, successor_release=None):
        self.calls = calls
        self.fail_layer = fail_layer
        self.successor_release = successor_release
        self.successor_entered = threading.Event()
        self._glm5_next_fp8_transport_poisoned = False
        self.gpu_layer = SimpleNamespace(
            w13_weight=torch.empty((1,), dtype=torch.float32)
        )
        self.gpu_method = _FakePrefetchGpuMethod(calls)

    def load_glm5_next_fp8_native(
        self, *, layer_idx, wrapper, consumed_event=None
    ):
        self.calls.append(("load-enter", layer_idx, wrapper.name))
        if consumed_event is not None:
            consumed_event.synchronize()
        if layer_idx == 4 and self.successor_release is not None:
            self.successor_entered.set()
            if not self.successor_release.wait(timeout=1):
                raise RuntimeError("test successor release timed out")
        if layer_idx == self.fail_layer:
            raise RuntimeError(f"injected layer {layer_idx} load failure")
        self.calls.append(("load-done", layer_idx))

    def _validate_glm5_next_fp8_native_gpu_slot(self):
        self.calls.append(("validate-slot",))


def _fake_prefetch_method(layer_idx, calls):
    cpu_infer = SimpleNamespace(
        sync=mock.Mock(side_effect=lambda: calls.append(("cpu-infer-sync",)))
    )
    writer = SimpleNamespace(
        name=f"writer-{layer_idx}", cpu_infer=cpu_infer, moe=object()
    )
    method = SimpleNamespace(
        kt_config=SimpleNamespace(
            layer_idx=layer_idx,
            num_layers=45,
            is_glm5_next=True,
            method="FP8",
        ),
        tp_rank=0,
        wrapper=writer,
    )
    return method


class TestGlm5NextFp8SingleSlotLayout(unittest.TestCase):
    def setUp(self):
        kt_ep_wrapper._SHARED_FULL_CONTEXT = None
        _FakeSharedMemory.instances.clear()
        for manager in list(
            kt_ep_wrapper._GLM5_NEXT_FP8_NATIVE_PREFETCH_MANAGERS.values()
        ):
            manager.close()
        kt_ep_wrapper._GLM5_NEXT_FP8_NATIVE_PREFETCH_LAYER_REGISTRY.clear()
        kt_ep_wrapper._GLM5_NEXT_FP8_NATIVE_PREFETCH_MANAGERS.clear()

    def test_generic_host_transport_allocates_exactly_two_expert_slots(self):
        context = object.__new__(kt_ep_wrapper.SharedFullContext)
        context.gpu_layer = SimpleNamespace(
            num_experts=3,
            w13_weight=_TensorMetadata((3, 4, 5), torch.float8_e4m3fn),
            w13_weight_scale_inv=_TensorMetadata((3, 1, 1), torch.float32),
            w2_weight=_TensorMetadata((3, 5, 2), torch.float8_e4m3fn),
            w2_weight_scale_inv=_TensorMetadata((3, 1, 1), torch.float32),
        )
        context._is_mxfp4_quant = False
        context._is_mxfp8_quant = False
        context._is_fp8_quant = True
        context._is_fp8_channel_quant = False
        context._is_bf16_quant = False
        context._commit_cpu_buffer_phase = mock.Mock()
        context._collect_all_rank_buffer_pointers = mock.Mock(
            return_value={name: [index + 1] for index, name in enumerate(
                kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8
            )}
        )
        fake_numa = SimpleNamespace(
            numa_available=lambda: 0,
            numa_set_localalloc=lambda: None,
        )

        with (
            mock.patch.object(kt_ep_wrapper.ctypes, "CDLL", return_value=fake_numa),
            mock.patch.object(
                kt_ep_wrapper.shared_memory,
                "SharedMemory",
                side_effect=_FakeSharedMemory,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            mock.patch.object(kt_ep_wrapper.dist, "is_initialized", return_value=False),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            context._create_cpu_buffers()

        expected_shapes = {
            "w13_weight": (2, 4, 5),
            "w13_weight_scale_inv": (2, 1, 1),
            "w2_weight": (2, 5, 2),
            "w2_weight_scale_inv": (2, 1, 1),
        }
        self.assertEqual(set(context.cpu_buffers), set(expected_shapes))
        for name, expected_shape in expected_shapes.items():
            self.assertEqual(tuple(context.cpu_buffers[name].shape), expected_shape)
        self.assertEqual(len(_FakeSharedMemory.instances), 4)
        self.assertTrue(all(item.unlinked for item in _FakeSharedMemory.instances))

    def test_exact_glm_transport_reuses_two_persistent_host_slot_events(self):
        context = object.__new__(kt_ep_wrapper.SharedFullContext)
        context._glm5_next_fp8_transport_initialized = True
        context._glm5_next_fp8_transport_backend = "legacy"
        context._glm5_next_fp8_copy_stream = _FakeCudaStream()
        context._glm5_next_fp8_host_slot_events = (
            _FakeCudaEvent(),
            _FakeCudaEvent(),
        )
        context._glm5_next_fp8_host_slot_was_used = [False, False]
        context._glm5_next_fp8_transport_status = None
        context.gpu_layer = SimpleNamespace(num_experts=4)
        context.cpu_buffers = {}
        context.all_rank_buffer_ptrs = {}
        for name in kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8:
            cpu_buffer = torch.empty((2, 1), dtype=torch.float32)
            gpu_tensor = torch.empty((4, 1), dtype=torch.float32)
            context.cpu_buffers[name] = cpu_buffer
            context.all_rank_buffer_ptrs[name] = [cpu_buffer.data_ptr()]
            setattr(context.gpu_layer, name, gpu_tensor)

        writer = _FakeFp8Writer(context)
        current_stream = _FakeCudaStream()
        with (
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            mock.patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda _stream: contextlib.nullcontext(),
            ),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                return_value=current_stream,
            ),
        ):
            context._prepare_weight_fp8_glm5_next(writer, layer_idx=3)

        self.assertEqual(writer.submitted, [(1, 0), (1, 1), (1, 2), (1, 3)])
        self.assertEqual(writer.sync_count, 4)
        for offset, name in enumerate(
            kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8
        ):
            expected = torch.arange(4, dtype=torch.float32) + offset * 10
            torch.testing.assert_close(
                getattr(context.gpu_layer, name).flatten(), expected
            )
        event0, event1 = context._glm5_next_fp8_host_slot_events
        self.assertEqual(event0.record_count, 2)
        self.assertEqual(event1.record_count, 2)
        self.assertEqual(event0.synchronize_count, 1)
        self.assertEqual(event1.synchronize_count, 1)
        self.assertEqual(context._glm5_next_fp8_copy_stream.synchronize_count, 1)

    @staticmethod
    def _native_context():
        context = object.__new__(kt_ep_wrapper.SharedFullContext)
        context.shm_unique_id = "native_test"
        context._commit_cpu_buffer_phase = mock.Mock()
        context.cpu_buffers = {}
        context.all_rank_buffer_ptrs = {}
        gpu_tensors = {}
        for index, name in enumerate(
            kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8
        ):
            cpu_buffer = torch.empty((2, index + 1), dtype=torch.float32)
            gpu_tensor = torch.empty((3, index + 1), dtype=torch.float32)
            context.cpu_buffers[name] = cpu_buffer
            context.all_rank_buffer_ptrs[name] = [
                cpu_buffer.data_ptr() + rank * 1_000_000
                for rank in range(4)
            ]
            gpu_tensors[name] = gpu_tensor
        context.gpu_layer = SimpleNamespace(num_experts=3, **gpu_tensors)
        return context

    def test_native_transport_uses_fixed_flattened_pointer_abi(self):
        context = self._native_context()
        calls = []
        extension = _FakeNativeExtension(calls)
        writer = _FakeNativeWriter(calls)

        with (
            mock.patch.dict(
                "os.environ",
                {"SGLANG_KT_GLM5_NEXT_FP8_TRANSPORT": "native"},
                clear=False,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_load_glm5_next_fp8_native_transport_api",
                return_value=(extension, None),
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            mock.patch.object(kt_ep_wrapper.dist, "is_initialized", return_value=False),
            mock.patch.object(
                kt_ep_wrapper.shared_memory,
                "SharedMemory",
                side_effect=_FakeSharedMemory,
            ),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            context._initialize_glm5_next_fp8_transport(
                wrapper=writer, num_gpu_experts=0
            )

        self.assertEqual(
            extension.initialize_args[1:],
            (kt_ep_wrapper._GLM5_NEXT_FP8_CONTROL_BYTES, 4),
        )
        args = extension.transport.args
        self.assertEqual(args[1:5], (4096, 0, 4, 0))
        local_host_ptrs = args[5]
        local_gpu_ptrs = args[6]
        all_rank_host_ptrs = args[7]
        expert_nbytes = args[8]
        self.assertEqual(len(local_host_ptrs), 8)
        self.assertEqual(len(local_gpu_ptrs), 4)
        self.assertEqual(len(all_rank_host_ptrs), 32)
        self.assertEqual(expert_nbytes, [4, 8, 12, 16])

        expected_local = []
        expected_all_rank = []
        names = kt_ep_wrapper.SharedFullContext.WEIGHT_NAMES_FP8
        for slot in range(2):
            for index, name in enumerate(names):
                expected_local.append(
                    context.cpu_buffers[name].data_ptr()
                    + slot * expert_nbytes[index]
                )
            for rank in range(4):
                for index, name in enumerate(names):
                    expected_all_rank.append(
                        context.all_rank_buffer_ptrs[name][rank]
                        + slot * expert_nbytes[index]
                    )
        self.assertEqual(local_host_ptrs, expected_local)
        self.assertEqual(all_rank_host_ptrs, expected_all_rank)
        self.assertEqual(
            local_gpu_ptrs,
            [getattr(context.gpu_layer, name).data_ptr() for name in names],
        )
        self.assertEqual(args[9:], (3, 60_000))
        self.assertEqual(context._glm5_next_fp8_transport_backend, "native")
        self.assertTrue(context._glm5_next_fp8_control_shm.unlinked)

    def test_native_prepare_is_one_join_run_wait_per_layer_and_epochs_advance(self):
        context = self._native_context()
        calls = []
        extension = _FakeNativeExtension(calls)
        transport = extension.FP8LayerwiseTransport()
        context._glm5_next_fp8_native_transport = transport
        context._glm5_next_fp8_transport_backend = "native"
        context._glm5_next_fp8_transport_initialized = True
        context._glm5_next_fp8_transport_poisoned = False
        context._glm5_next_fp8_transport_epoch = 0
        writer = _FakeNativeWriter(calls)

        with mock.patch.object(
            kt_ep_wrapper,
            "get_tensor_model_parallel_rank",
            return_value=0,
        ):
            context._prepare_weight_fp8_glm5_next(writer, layer_idx=3)
            context._prepare_weight_fp8_glm5_next(writer, layer_idx=4)

        self.assertEqual(
            [call for call in calls if call[0] in ("join", "run", "wait")],
            [
                ("join", 1, 3, 3),
                ("run", 1, 3, 3),
                ("wait", 1),
                ("join", 2, 4, 3),
                ("run", 2, 4, 3),
                ("wait", 2),
            ],
        )
        writer.cpu_infer.sync.assert_not_called()
        source = inspect.getsource(
            kt_ep_wrapper.SharedFullContext._prepare_weight_fp8_glm5_next_native
        )
        self.assertNotIn("dist.", source)
        self.assertNotIn(".item()", source)
        self.assertNotIn("for expert", source)

    def test_native_runtime_failure_poisoned_without_legacy_fallback(self):
        context = self._native_context()
        calls = []
        extension = _FakeNativeExtension(calls)
        transport = extension.FP8LayerwiseTransport()
        context._glm5_next_fp8_native_transport = transport
        context._glm5_next_fp8_transport_backend = "native"
        context._glm5_next_fp8_transport_initialized = True
        context._glm5_next_fp8_transport_poisoned = False
        context._glm5_next_fp8_transport_epoch = 0
        writer = _FakeNativeWriter(calls, fail=True)
        context._prepare_weight_fp8_glm5_next_legacy = mock.Mock()

        with mock.patch.object(
            kt_ep_wrapper,
            "get_tensor_model_parallel_rank",
            return_value=0,
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime fallback is disabled"):
                context._prepare_weight_fp8_glm5_next(writer, layer_idx=3)
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                context._prepare_weight_fp8_glm5_next(writer, layer_idx=4)

        self.assertTrue(context._glm5_next_fp8_transport_poisoned)
        self.assertEqual(transport.close_count, 1)
        context._prepare_weight_fp8_glm5_next_legacy.assert_not_called()

    def test_transport_mode_fallback_and_hard_gates(self):
        context = self._native_context()
        context._initialize_glm5_next_fp8_legacy_transport = mock.Mock()
        writer = _FakeNativeWriter([])
        with (
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            mock.patch.object(
                kt_ep_wrapper.dist, "is_initialized", return_value=False
            ),
            mock.patch.dict(
                "os.environ",
                {"SGLANG_KT_GLM5_NEXT_FP8_TRANSPORT": "auto"},
                clear=False,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_load_glm5_next_fp8_native_transport_api",
                return_value=(None, "missing test API"),
            ),
        ):
            context._initialize_glm5_next_fp8_transport(
                wrapper=writer, num_gpu_experts=0
            )
        context._initialize_glm5_next_fp8_legacy_transport.assert_called_once_with()

        context._initialize_glm5_next_fp8_legacy_transport.reset_mock()
        with (
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            mock.patch.object(kt_ep_wrapper.dist, "is_initialized", return_value=False),
            mock.patch.dict(
                "os.environ",
                {"SGLANG_KT_GLM5_NEXT_FP8_TRANSPORT": "native"},
                clear=False,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_load_glm5_next_fp8_native_transport_api",
                return_value=(None, "missing test API"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "was forced"):
                context._initialize_glm5_next_fp8_transport(
                    wrapper=writer, num_gpu_experts=0
                )
            with self.assertRaisesRegex(RuntimeError, "gpu_experts=0"):
                context._initialize_glm5_next_fp8_transport(
                    wrapper=writer, num_gpu_experts=1
                )
        context._initialize_glm5_next_fp8_legacy_transport.assert_not_called()

    def test_native_single_slot_state_machine_prefetches_and_wraps(self):
        calls = []
        successor_release = threading.Event()
        context = _FakePrefetchContext(
            calls, successor_release=successor_release
        )
        method3 = _fake_prefetch_method(3, calls)
        method4 = _fake_prefetch_method(4, calls)
        layer3 = object()
        layer4 = object()
        registry = {3: (method3, layer3), 4: (method4, layer4)}
        event = _FakePrefetchEvent(calls)
        main_stream = object()

        with (
            mock.patch.object(torch.cuda, "Event", return_value=event),
            mock.patch.object(
                kt_ep_wrapper,
                "_GLM5_NEXT_FP8_MOE_LAYER_INDICES",
                (3, 4),
            ),
            mock.patch.object(
                torch.cuda, "current_stream", return_value=main_stream
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_all_tp_ranks_succeeded",
                side_effect=lambda success: success,
            ),
        ):
            manager = (
                kt_ep_wrapper._Glm5NextFp8NativeSingleSlotPrefetchManager(
                    context, ("test",), registry
                )
            )
            try:
                first = manager.apply(
                    method3, layer3, SimpleNamespace(layer_idx=3)
                )
                self.assertEqual(first, "result-3")
                self.assertTrue(context.successor_entered.wait(timeout=1))
                self.assertIsNotNone(manager.future)
                self.assertFalse(manager.future.done())

                # The main thread has already returned from layer 3 while its
                # sole worker remains in successor transport.
                successor_release.set()
                second = manager.apply(
                    method4, layer4, SimpleNamespace(layer_idx=4)
                )
                self.assertEqual(second, "result-4")
                self.assertIsNone(manager.future)  # last layer never prefetches

                # A new chunk/request wraps to the first MoE.  It synchronously
                # fences layer 4 and re-primes the same GPU slot.
                wrapped = manager.apply(
                    method3, layer3, SimpleNamespace(layer_idx=3)
                )
                self.assertEqual(wrapped, "result-3")
                successor = manager.apply(
                    method4, layer4, SimpleNamespace(layer_idx=4)
                )
                self.assertEqual(successor, "result-4")
            finally:
                manager.close()

        self.assertEqual(
            [call[:2] for call in calls if call[0] == "load-enter"],
            [
                ("load-enter", 3),
                ("load-enter", 4),
                ("load-enter", 3),
                ("load-enter", 4),
            ],
        )
        self.assertEqual(calls.count(("cpu-infer-sync",)), 1)
        method3.wrapper.cpu_infer.sync.assert_called_once_with()
        method4.wrapper.cpu_infer.sync.assert_not_called()
        self.assertEqual(manager.round_id, 2)
        self.assertEqual(manager.last_acquired_layer_idx, 4)
        self.assertFalse(manager.context._glm5_next_fp8_transport_poisoned)

    def test_native_prefetch_rejects_partial_glm_registry_before_prime(self):
        calls = []
        context = _FakePrefetchContext(calls)
        method3 = _fake_prefetch_method(3, calls)
        with self.assertRaisesRegex(RuntimeError, "complete MoE registry"):
            kt_ep_wrapper._Glm5NextFp8NativeSingleSlotPrefetchManager(
                context, ("partial",), {3: (method3, object())}
            )
        self.assertNotIn(("cpu-infer-sync",), calls)
        self.assertFalse(
            any(call[0] == "load-enter" for call in calls if call)
        )

    def test_native_successor_failure_poisoned_without_fallback(self):
        calls = []
        context = _FakePrefetchContext(calls, fail_layer=4)
        method3 = _fake_prefetch_method(3, calls)
        method4 = _fake_prefetch_method(4, calls)
        layer3 = object()
        layer4 = object()
        registry = {3: (method3, layer3), 4: (method4, layer4)}
        event = _FakePrefetchEvent(calls)

        with (
            mock.patch.object(torch.cuda, "Event", return_value=event),
            mock.patch.object(
                kt_ep_wrapper,
                "_GLM5_NEXT_FP8_MOE_LAYER_INDICES",
                (3, 4),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=object()),
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            mock.patch.object(
                kt_ep_wrapper,
                "_all_tp_ranks_succeeded",
                side_effect=lambda success: success,
            ),
        ):
            manager = (
                kt_ep_wrapper._Glm5NextFp8NativeSingleSlotPrefetchManager(
                    context, ("failure-test",), registry
                )
            )
            try:
                manager.apply(method3, layer3, SimpleNamespace(layer_idx=3))
                with self.assertRaisesRegex(
                    RuntimeError, "runtime fallback is disabled"
                ):
                    manager.apply(
                        method4, layer4, SimpleNamespace(layer_idx=4)
                    )
                with self.assertRaisesRegex(RuntimeError, "poisoned"):
                    manager.apply(
                        method3, layer3, SimpleNamespace(layer_idx=3)
                    )
            finally:
                manager.close()

        self.assertTrue(context._glm5_next_fp8_transport_poisoned)
        self.assertNotIn("legacy", " ".join(map(str, calls)).lower())

    def test_native_prefetch_source_contracts(self):
        exact_load_source = inspect.getsource(
            kt_ep_wrapper.SharedFullContext.load_glm5_next_fp8_native
        )
        self.assertIn("consumed_event.synchronize()", exact_load_source)
        self.assertNotIn("torch.cuda.synchronize", exact_load_source)
        self.assertNotIn("_all_tp_ranks_succeeded", exact_load_source)

        native_prepare_source = inspect.getsource(
            kt_ep_wrapper.SharedFullContext._prepare_weight_fp8_glm5_next_native
        )
        self.assertIn('getattr(wrapper, "moe", None)', native_prepare_source)
        self.assertNotIn("cpu_infer.sync", native_prepare_source)

        worker_source = inspect.getsource(
            kt_ep_wrapper._Glm5NextFp8NativeSingleSlotPrefetchManager._load_layer
        )
        self.assertNotIn("dist.", worker_source)
        self.assertNotIn("_all_tp_ranks_succeeded", worker_source)

        create_source = inspect.getsource(
            kt_ep_wrapper.KTEPWrapperMethod.create_weights
        )
        self.assertIn(
            "_register_glm5_next_fp8_native_prefetch_layer", create_source
        )

        close_source = inspect.getsource(
            kt_ep_wrapper.SharedFullContext._close_glm5_next_fp8_transport
        )
        self.assertLess(
            close_source.index("prefetch_manager.close()"),
            close_source.index("transport.close()"),
        )

    def test_tp4_fp8_geometry_and_single_full_layer_slot_bytes(self):
        num_experts = 288
        hidden_size = 4096
        global_intermediate_size = 2048
        tp_size = 4
        block_size = 128
        intermediate_size = global_intermediate_size // tp_size

        self.assertEqual(intermediate_size, 512)
        gpu_tensors = {
            "w13_weight": _TensorMetadata(
                (num_experts, 2 * intermediate_size, hidden_size),
                torch.float8_e4m3fn,
            ),
            "w13_weight_scale_inv": _TensorMetadata(
                (
                    num_experts,
                    math.ceil(2 * intermediate_size / block_size),
                    math.ceil(hidden_size / block_size),
                ),
                torch.float32,
            ),
            "w2_weight": _TensorMetadata(
                (num_experts, hidden_size, intermediate_size),
                torch.float8_e4m3fn,
            ),
            "w2_weight_scale_inv": _TensorMetadata(
                (
                    num_experts,
                    math.ceil(hidden_size / block_size),
                    math.ceil(intermediate_size / block_size),
                ),
                torch.float32,
            ),
        }
        host_buffers = {
            name: _TensorMetadata((2, *tensor.shape[1:]), tensor.dtype)
            for name, tensor in gpu_tensors.items()
        }
        context = SimpleNamespace(
            _is_fp8_quant=True,
            _glm5_next_fp8_transport_initialized=True,
            _glm5_next_fp8_copy_stream=object(),
            _glm5_next_fp8_host_slot_events=(object(), object()),
            _get_base_quant_method=lambda: SimpleNamespace(
                block_quant=True,
                use_mxfp8=False,
                weight_block_size=[128, 128],
                runner=SimpleNamespace(
                    runner_backend=SimpleNamespace(is_triton=lambda: True)
                ),
            ),
            gpu_layer=SimpleNamespace(
                num_experts=num_experts,
                moe_runner_config=SimpleNamespace(
                    glm5_next_hf_two_round_swiglu=True
                ),
                **gpu_tensors,
            ),
            cpu_buffers=host_buffers,
        )

        with mock.patch.object(
            kt_ep_wrapper,
            "get_tensor_model_parallel_world_size",
            return_value=4,
        ):
            gpu_slot_bytes, host_buffer_bytes = (
                kt_ep_wrapper._validate_glm5_next_fp8_shared_full_context(
                    context
                )
            )
        self.assertEqual(gpu_slot_bytes, 1_812_381_696)
        self.assertEqual(host_buffer_bytes, 12_585_984)

        context.cpu_buffers["w13_weight"] = _TensorMetadata(
            (1, *gpu_tensors["w13_weight"].shape[1:]), torch.float8_e4m3fn
        )
        with (
            mock.patch.object(
                kt_ep_wrapper,
                "get_tensor_model_parallel_world_size",
                return_value=4,
            ),
            self.assertRaisesRegex(
                RuntimeError, "exactly two expert Host slots"
            ),
        ):
            kt_ep_wrapper._validate_glm5_next_fp8_shared_full_context(context)

    def test_private_glm_two_slot_pipeline_symbols_are_gone(self):
        removed_symbols = (
            "_GLM5_NEXT_FP8_PREFILL_LAYER_REGISTRY",
            "_GLM5_NEXT_FP8_LAYERWISE_MANAGERS",
            "_Glm5NextFp8PrefillSlot",
            "_Glm5NextFp8LayerwisePrefillManager",
            "_register_glm5_next_fp8_prefill_layer",
            "_initialize_glm5_next_fp8_layerwise_pipeline",
            "initialize_glm5_next_fp8_layerwise_prefill",
            "_get_glm5_next_fp8_layerwise_manager",
        )
        for symbol in removed_symbols:
            with self.subTest(symbol=symbol):
                self.assertFalse(hasattr(kt_ep_wrapper, symbol))

        model_runner_source = (
            Path(__file__).resolve().parents[2]
            / "python/sglang/srt/model_executor/model_runner.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "initialize_glm5_next_fp8_layerwise_prefill", model_runner_source
        )


if __name__ == "__main__":
    unittest.main()
