"""CPU guards for GLM-5-Next's opt-in FP32 KDA output partial."""

from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next.py"


def _compile_functions(*function_names, globals_):
    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    functions = {
        node.name: copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    }
    assert functions.keys() == set(function_names)
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                *(functions[name] for name in function_names),
            ],
            type_ignores=[],
        )
    )
    namespace = {"__builtins__": __builtins__, **globals_}
    exec(compile(module, str(MODEL_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in function_names)


class _Unquantized:
    pass


class _FakeTensor:
    def __init__(self, *, is_cuda=True, dtype=torch.bfloat16, device="cuda:0"):
        self.is_cuda = is_cuda
        self.dtype = dtype
        self.device = torch.device(device)


class _RowParallel:
    def __init__(self):
        self.quant_method = _Unquantized()
        self.weight = _FakeTensor()
        self.bias = None
        self.reduce_results = False
        self.input_is_parallel = True


def _predicate():
    (predicate,) = _compile_functions(
        "_glm5_next_kda_can_use_fp32_o_proj",
        globals_={
            "nn": object(),
            "torch": torch,
            "RowParallelLinear": _RowParallel,
            "UnquantizedLinearMethod": _Unquantized,
        },
    )
    return predicate


def test_fp32_partial_requires_every_glm_kda_contract_guard():
    predicate = _predicate()
    hidden_states = _FakeTensor()

    assert predicate(_RowParallel(), hidden_states)

    cases = (
        lambda layer, hidden: setattr(layer, "quant_method", object()),
        lambda layer, hidden: setattr(hidden, "is_cuda", False),
        lambda layer, hidden: setattr(hidden, "dtype", torch.float32),
        lambda layer, hidden: setattr(layer.weight, "is_cuda", False),
        lambda layer, hidden: setattr(layer.weight, "device", torch.device("cuda:1")),
        lambda layer, hidden: setattr(layer.weight, "dtype", torch.float32),
        lambda layer, hidden: setattr(layer, "bias", object()),
        lambda layer, hidden: setattr(layer, "reduce_results", True),
        lambda layer, hidden: setattr(layer, "input_is_parallel", False),
    )
    for invalidate in cases:
        layer = _RowParallel()
        hidden = _FakeTensor()
        invalidate(layer, hidden)
        assert not predicate(layer, hidden)

    assert not predicate(object(), hidden_states)


def test_cpu_uses_original_row_parallel_forward():
    calls = []

    class CPUFallback(_RowParallel):
        def __init__(self):
            super().__init__()
            self.weight = torch.eye(2, dtype=torch.bfloat16)

        def __call__(self, hidden_states):
            calls.append(hidden_states)
            return hidden_states + 3, None

    @contextmanager
    def forbidden_symmetric_memory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CPU fallback must not enter symmetric memory")
        yield

    predicate, project = _compile_functions(
        "_glm5_next_kda_can_use_fp32_o_proj",
        "_glm5_next_kda_o_proj",
        globals_={
            "nn": object(),
            "torch": torch,
            "RowParallelLinear": _RowParallel,
            "UnquantizedLinearMethod": _Unquantized,
            "use_symmetric_memory": forbidden_symmetric_memory,
            "get_tp_group": lambda: object(),
            "is_allocation_symmetric": lambda: True,
        },
    )
    del predicate
    hidden_states = torch.ones(1, 2, dtype=torch.bfloat16)
    output = project(CPUFallback(), hidden_states)

    assert calls == [hidden_states]
    torch.testing.assert_close(output, hidden_states + 3)


def test_cuda_contract_mismatches_fail_closed_without_calling_fallback():
    fallback_calls = []

    class RecordingRow(_RowParallel):
        def __call__(self, hidden_states):
            fallback_calls.append(hidden_states)
            return hidden_states, None

    @contextmanager
    def forbidden_symmetric_memory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid contract must fail before symmetric memory")
        yield

    predicate, project = _compile_functions(
        "_glm5_next_kda_can_use_fp32_o_proj",
        "_glm5_next_kda_o_proj",
        globals_={
            "nn": object(),
            "torch": torch,
            "RowParallelLinear": _RowParallel,
            "UnquantizedLinearMethod": _Unquantized,
            "use_symmetric_memory": forbidden_symmetric_memory,
            "get_tp_group": lambda: object(),
            "is_allocation_symmetric": lambda: True,
        },
    )
    del predicate

    cases = (
        lambda layer, hidden: setattr(layer, "quant_method", object()),
        lambda layer, hidden: setattr(hidden, "dtype", torch.float32),
        lambda layer, hidden: setattr(layer.weight, "is_cuda", False),
        lambda layer, hidden: setattr(layer.weight, "device", torch.device("cuda:1")),
        lambda layer, hidden: setattr(layer.weight, "dtype", torch.float32),
        lambda layer, hidden: setattr(layer, "bias", object()),
        lambda layer, hidden: setattr(layer, "reduce_results", True),
        lambda layer, hidden: setattr(layer, "input_is_parallel", False),
    )
    for invalidate in cases:
        layer = RecordingRow()
        hidden_states = _FakeTensor()
        invalidate(layer, hidden_states)
        with pytest.raises(RuntimeError, match="CUDA KDA o_proj requires"):
            project(layer, hidden_states)

    with pytest.raises(RuntimeError, match="CUDA KDA o_proj requires"):
        project(object(), _FakeTensor())
    assert fallback_calls == []


def test_production_path_preserves_allocator_and_dtype_contracts():
    source = MODEL_PATH.read_text(encoding="utf-8")
    function_source = ast.get_source_segment(
        source,
        next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_glm5_next_kda_o_proj"
        ),
    )
    assert function_source is not None
    assert "use_symmetric_memory(" in function_source
    assert "get_tp_group(), disabled=not is_allocation_symmetric()" in function_source
    assert "torch.mm(" in function_source
    assert "out_dtype=torch.float32" in function_source

    # Only KDA opts into the post-all-reduce BF16 cast; DSA explicitly keeps
    # the communicator's default no-cast policy.
    assert "torch.bfloat16 if config.is_kda_layer(layer_idx) else None" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fp32_partial_executes_without_bf16_output_round():
    entries = []

    class CUDARow(_RowParallel):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(3, 4, device="cuda", dtype=torch.bfloat16)

        def __call__(self, hidden_states):
            raise AssertionError("eligible CUDA path must bypass ordinary forward")

    @contextmanager
    def symmetric_memory(group, *, disabled):
        entries.append((group, disabled))
        yield

    group = object()
    predicate, project = _compile_functions(
        "_glm5_next_kda_can_use_fp32_o_proj",
        "_glm5_next_kda_o_proj",
        globals_={
            "nn": object(),
            "torch": torch,
            "RowParallelLinear": _RowParallel,
            "UnquantizedLinearMethod": _Unquantized,
            "use_symmetric_memory": symmetric_memory,
            "get_tp_group": lambda: group,
            "is_allocation_symmetric": lambda: False,
        },
    )
    del predicate
    layer = CUDARow()
    hidden_states = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
    output = project(layer, hidden_states)

    assert entries == [(group, True)]
    assert output.dtype is torch.float32
    torch.testing.assert_close(
        output,
        torch.mm(hidden_states, layer.weight.t(), out_dtype=torch.float32),
    )
