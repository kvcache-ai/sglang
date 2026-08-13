"""CPU-only regression guards for the GLM-5-Next integration boundary.

These tests intentionally load or compile only the small production-code surface
under test.  That keeps the guards runnable without importing the full SGLang
package (and therefore without CUDA, Transformers, or optional serving deps).
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse(relative_path: str) -> ast.Module:
    path = REPO_ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_function(
    tree: ast.Module, function_name: str, class_name: str | None = None
) -> ast.FunctionDef:
    container = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        container = class_node.body
    return next(
        node
        for node in container
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _compile_function(
    relative_path: str,
    function_name: str,
    *,
    class_name: str | None = None,
    globals_: dict | None = None,
):
    """Compile one function/method from production source with deferred annotations."""
    function = copy.deepcopy(
        _find_function(_parse(relative_path), function_name, class_name)
    )
    function.decorator_list = []
    # Model-specific adjustment imports helpers at method entry.  The harness
    # supplies those helpers directly and avoids importing the full package.
    function.body = [
        node
        for node in function.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    body: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    symbol_name = function_name
    if class_name is None:
        body.append(function)
    else:
        symbol_name = "_SourceHarness"
        body.append(
            ast.ClassDef(
                name=symbol_name,
                bases=[],
                keywords=[],
                body=[function],
                decorator_list=[],
            )
        )

    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"__builtins__": __builtins__}
    if globals_:
        namespace.update(globals_)
    exec(compile(module, relative_path, "exec"), namespace)
    compiled = namespace[symbol_name]
    return compiled if class_name is None else getattr(compiled, function_name)


def _load_attention_registry():
    path = REPO_ROOT / "python/sglang/srt/layers/attention/attention_registry.py"
    spec = importlib.util.spec_from_file_location("_glm_guard_attention_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pool_registry():
    """Load the real registry with only its two eager imports stubbed."""

    class MiniMaxPool:
        pass

    packages = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.configs", "sglang.srt.mem_cache"):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    model_config = types.ModuleType("sglang.srt.configs.model_config")
    model_config.is_minimax_sparse = lambda config: (
        (getattr(config, "architectures", None) or [None])[0]
        in {
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3SparseForConditionalGeneration",
        }
    )
    memory_pool = types.ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.MiniMaxSparseKVPool = MiniMaxPool
    packages[model_config.__name__] = model_config
    packages[memory_pool.__name__] = memory_pool

    path = REPO_ROOT / "python/sglang/srt/mem_cache/pool_registry.py"
    spec = importlib.util.spec_from_file_location("_glm_guard_pool_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module, MiniMaxPool


def _config(architecture: str, **kwargs):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture], **kwargs)
    )


class _LogStub:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _EnvFieldStub:
    def __init__(self, name: str):
        self.name = name

    def get(self):
        return os.environ.get(self.name)

    def is_set(self):
        return self.name in os.environ

    def set(self, value):
        os.environ[self.name] = str(value)


class _EnvsStub:
    def __getattr__(self, name: str):
        return _EnvFieldStub(name)


def _server_adjustment_method():
    model_config_path = "python/sglang/srt/configs/model_config.py"
    model_config_tree = _parse(model_config_path)
    predicate_name = (
        "is_deepseek_nsa"
        if any(
            isinstance(node, ast.FunctionDef) and node.name == "is_deepseek_nsa"
            for node in model_config_tree.body
        )
        else "is_deepseek_dsa"
    )
    dsa_predicate = _compile_function(model_config_path, predicate_name)
    return _compile_function(
        "python/sglang/srt/server_args.py",
        "_handle_model_specific_adjustments",
        class_name="ServerArgs",
        globals_={
            "ConnectorType": SimpleNamespace(INSTANCE="instance"),
            "envs": _EnvsStub(),
            "get_device_name": lambda: None,
            "is_blackwell_supported": lambda: False,
            "is_deepseek_dsa": dsa_predicate,
            "is_deepseek_nsa": dsa_predicate,
            "is_hip": lambda: False,
            "is_npu": lambda: True,
            "is_sm100_supported": lambda: False,
            "logger": _LogStub(),
            "parse_connector_type": lambda model_path: "local",
        },
    )


def _server_stub(architecture: str, **config_kwargs):
    server = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        enable_flashinfer_allreduce_fusion=False,
        enable_nsa_prefill_context_parallel=False,
        enable_two_batch_overlap=False,
        kv_cache_dtype="auto",
        model_path="stub/model",
        page_size=None,
        prefill_attention_backend=None,
    )
    hf_config = SimpleNamespace(architectures=[architecture], **config_kwargs)
    server.get_model_config = types.MethodType(
        lambda self: SimpleNamespace(hf_config=hf_config), server
    )
    server.is_attention_backend_not_set = types.MethodType(
        lambda self: (
            self.attention_backend is None
            and self.prefill_attention_backend is None
            and self.decode_attention_backend is None
        ),
        server,
    )
    return server


def _without_sglang_env():
    clean_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("SGLANG_") and not name.startswith("SGL_")
    }
    return patch.dict(os.environ, clean_env, clear=True)


class TestNonGlmRegressionBoundary(unittest.TestCase):
    def test_scheduler_glm_lifecycle_dispatches_are_exactly_gated(self):
        release_req = _find_function(
            _parse("python/sglang/srt/managers/schedule_batch.py"),
            "release_req",
            class_name="ScheduleBatch",
        )
        release_source = ast.unparse(release_req)
        self.assertIn("_glm5_next_session_ab_active", release_source)
        self.assertIn("dispatch_named('glm5_next_kpool'", release_source)

        output_source = (
            REPO_ROOT / "python/sglang/srt/managers/scheduler_output_processor_mixin.py"
        ).read_text(encoding="utf-8")
        self.assertEqual(output_source.count('"glm5_next_kpool"'), 2)
        self.assertEqual(output_source.count('"is_glm5_next"'), 2)

        flush_cache = _find_function(
            _parse("python/sglang/srt/managers/scheduler.py"),
            "flush_cache",
            class_name="Scheduler",
        )
        flush_source = ast.unparse(flush_cache)
        self.assertIn("getattr(self.model_config, 'is_glm5_next', False)", flush_source)
        self.assertIn(
            "dispatch_named('glm5_next_kpool', 'on_cache_flush')", flush_source
        )

    def test_attention_tp_context_cleans_state_when_forward_raises(self):
        maybe_input_scattered = _compile_function(
            "python/sglang/srt/layers/communicator.py",
            "maybe_input_scattered",
            class_name="AttnTpContext",
        )

        class _Context:
            input_scattered_ = False
            attn_inputs_ = object()

            @property
            def input_scattered(self):
                return self.input_scattered_

            def use_input_scattered(self, forward_batch):
                del forward_batch
                return True

        context = _Context()
        guarded = contextmanager(maybe_input_scattered)
        with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
            with guarded(context, object()):
                self.assertTrue(context.input_scattered_)
                raise RuntimeError("injected forward failure")

        self.assertFalse(context.input_scattered_)
        self.assertIsNone(context.attn_inputs_)

    def test_named_lifecycle_dispatch_does_not_call_legacy_hooks(self):
        path = REPO_ROOT / "python/sglang/srt/managers/forward_hooks_registry.py"
        spec = importlib.util.spec_from_file_location(
            "_glm_guard_forward_hooks_registry", path
        )
        registry = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(registry)

        calls = []

        class _Hook:
            def __init__(self, name):
                self.name = name

            def on_request_finished(self, req):
                calls.append((self.name, req))

        req = object()
        registry.register_forward_hook("legacy", _Hook("legacy"))
        registry.register_forward_hook("glm5_next_kpool", _Hook("glm"))

        registry.dispatch_named("glm5_next_kpool", "on_request_finished", req)
        self.assertEqual(calls, [("glm", req)])

        calls.clear()
        registry.dispatch("on_request_finished", req)
        self.assertEqual(calls, [("legacy", req), ("glm", req)])

    def test_existing_attention_backend_registrations_are_not_replaced(self):
        registry = _load_attention_registry()
        expected = {
            "aiter": "create_aiter_backend",
            "ascend": "create_ascend_backend",
            "compressed": "create_compressed_backend",
            "cutlass_mla": "create_cutlass_mla_backend",
            "dual_chunk_flash_attn": "create_dual_chunk_flash_attn_backend",
            "fa3": "create_flashattention_v3_backend",
            "fa4": "create_flashattention_v4_backend",
            "flashinfer": "create_flashinfer_backend",
            "flashmla": "create_flashmla_backend",
            "flex_attention": "create_flex_attention_backend",
            "intel_amx": "create_intel_amx_backend",
            "intel_xpu": "create_intel_xpu_backend",
            "nsa": "create_nsa_backend",
            "torch_native": "create_torch_native_backend",
            "triton": "create_triton_backend",
            "trtllm_mha": "create_trtllm_mha_backend",
            "trtllm_mla": "create_trtllm_mla_backend",
            "wave": "create_wave_backend",
        }
        actual = {
            name: factory.__name__
            for name, factory in registry.ATTENTION_BACKENDS.items()
        }
        # GLM may add a backend, but it must not replace an existing mapping.
        self.assertEqual({name: actual.get(name) for name in expected}, expected)

    def test_server_args_defaults_and_non_glm_environment_stay_unchanged(self):
        tree = _parse("python/sglang/srt/server_args.py")
        server_args = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
        )
        defaults = {
            node.target.id: ast.literal_eval(node.value)
            for node in server_args.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and node.target.id
            in {
                "attention_backend",
                "decode_attention_backend",
                "kv_cache_dtype",
                "page_size",
                "prefill_attention_backend",
            }
        }
        self.assertEqual(
            defaults,
            {
                "attention_backend": None,
                "decode_attention_backend": None,
                "kv_cache_dtype": "auto",
                "page_size": None,
                "prefill_attention_backend": None,
            },
        )

        server = _server_stub("RegressionSentinelForCausalLM")
        before_fields = {
            name: getattr(server, name)
            for name in (
                "attention_backend",
                "decode_attention_backend",
                "kv_cache_dtype",
                "page_size",
                "prefill_attention_backend",
            )
        }
        with _without_sglang_env():
            before_env = {
                name: value
                for name, value in os.environ.items()
                if name.startswith("SGLANG_") or name.startswith("SGL_")
            }

            _server_adjustment_method()(server)

            self.assertEqual(
                {name: getattr(server, name) for name in before_fields},
                before_fields,
            )
            self.assertEqual(
                {
                    name: value
                    for name, value in os.environ.items()
                    if name.startswith("SGLANG_") or name.startswith("SGL_")
                },
                before_env,
            )


class TestLegacyAttentionRoutes(unittest.TestCase):
    def test_index_kpool_one_keeps_legacy_nsa_backend_and_pool(self):
        server = _server_stub("GlmMoeDsaForCausalLM", index_topk=2048, index_kpool=1)
        with _without_sglang_env():
            _server_adjustment_method()(server)
        self.assertEqual(server.attention_backend, "nsa")

        model_config_tree = _parse("python/sglang/srt/configs/model_config.py")
        helper_names = {
            node.name
            for node in model_config_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        if "get_dsa_index_kpool" in helper_names:
            get_index_kpool = _compile_function(
                "python/sglang/srt/configs/model_config.py", "get_dsa_index_kpool"
            )
            self.assertEqual(get_index_kpool(server.get_model_config().hf_config), 1)

        # The legacy branch must remain a direct NSATokenToKVPool fallback.  A
        # future GLM kpool plugin may claim index_kpool > 1 before this branch.
        init_pool = _find_function(
            _parse("python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"),
            "init_memory_pool",
            "ModelRunnerKVCacheMixin",
        )
        legacy_branches = [
            node
            for node in ast.walk(init_pool)
            if isinstance(node, ast.If)
            and "is_nsa_model" in ast.unparse(node.test)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "NSATokenToKVPool"
                for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            )
        ]
        self.assertTrue(
            legacy_branches,
            "index_kpool=1 must retain the legacy NSATokenToKVPool fallback",
        )

    def test_lower_bound_none_uses_the_existing_kda_decode_kernel(self):
        decode = _compile_function(
            "python/sglang/srt/layers/attention/linear/kda_backend.py",
            "decode",
            class_name="KDAKernelDispatcher",
            globals_={"TritonKDAKernel": type("TritonKDAKernel", (), {})},
        )

        calls = []

        class ExistingKimiDecodeKernel:
            def decode(self, *args, **kwargs):
                calls.append((args, kwargs))
                return "legacy-kimi-kda"

        dispatcher = SimpleNamespace(decode_kernel=ExistingKimiDecodeKernel())
        marker = object()
        result = decode(
            dispatcher,
            marker,
            marker,
            marker,
            marker,
            marker,
            A_log=marker,
            dt_bias=marker,
            ssm_states=marker,
            cache_indices=marker,
            query_start_loc=marker,
            lower_bound=None,
            regression_marker="kimi",
        )

        self.assertEqual(result, "legacy-kimi-kda")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][1].get("lower_bound"))
        self.assertEqual(calls[0][1]["regression_marker"], "kimi")


class TestGlmKPoolMemoryAccounting(unittest.TestCase):
    @staticmethod
    def _cell_size_method():
        class FakeNSAPool:
            quant_block_size = 128
            index_k_with_scale_buffer_dtype = __import__("torch").uint8

        return _compile_function(
            "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
            "get_cell_size_per_token",
            class_name="ModelRunnerKVCacheMixin",
            globals_={
                "NSATokenToKVPool": FakeNSAPool,
                "get_nsa_index_head_dim": lambda config: 128,
                "get_attention_tp_size": lambda: 1,
                "is_deepseek_compressed": lambda config: False,
                "is_deepseek_nsa": lambda config: bool(
                    getattr(config, "legacy_nsa", False)
                ),
                "is_float4_e2m1fn_x2": lambda dtype: False,
                "torch": __import__("torch"),
            },
        )

    @staticmethod
    def _runner(*, exact_glm: bool, legacy_nsa: bool = False):
        torch = __import__("torch")
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(legacy_nsa=legacy_nsa),
            is_glm5_next=exact_glm,
            uses_kpool4_compress=exact_glm,
            index_head_dim=128,
            qk_nope_head_dim=256 if exact_glm else 128,
            qk_rope_head_dim=0 if exact_glm else 64,
            kv_lora_rank=512,
        )
        return SimpleNamespace(
            kv_cache_dtype=torch.float8_e4m3fn,
            model_config=model_config,
            use_mla_backend=True,
            calculate_mla_kv_cache_dim=lambda: 512,
        )

    def test_exact_glm_cell_counts_raw_latent_and_scale_sidecar(self):
        get_cell_size = self._cell_size_method()
        runner = self._runner(exact_glm=True)

        # latent: (512 FP8 + four FP32 scales) * 11;
        # index: (128 FP8 + one FP32 scale) * 11.
        self.assertEqual(get_cell_size(runner, 11), 7260)

    def test_non_glm_and_legacy_nsa_keep_historical_arithmetic(self):
        get_cell_size = self._cell_size_method()
        non_glm = self._runner(exact_glm=False)
        legacy_nsa = self._runner(exact_glm=False, legacy_nsa=True)

        self.assertEqual(get_cell_size(non_glm, 11), (128 + 64) * 11)
        self.assertEqual(
            get_cell_size(legacy_nsa, 11),
            (128 + 64) * 11 + (128 + 4) * 11,
        )

    def test_fixed_tail_is_reserved_once_and_only_for_exact_glm(self):
        torch = __import__("torch")
        source = "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
        tail_size = _compile_function(
            source,
            "_get_glm5_next_kpool_tail_size_bytes",
            class_name="ModelRunnerKVCacheMixin",
            globals_={"torch": torch},
        )
        reserve = _compile_function(
            source,
            "_reserve_glm5_next_kpool_tail",
            class_name="ModelRunnerKVCacheMixin",
            globals_={"logger": _LogStub()},
        )

        text_config = SimpleNamespace(
            index_kpool=4,
            index_head_dim=128,
            index_kpool_compress=True,
        )
        exact = SimpleNamespace(
            model_config=SimpleNamespace(
                is_glm5_next=True,
                uses_kpool4_compress=True,
                hf_text_config=text_config,
            )
        )
        exact._get_glm5_next_kpool_tail_size_bytes = types.MethodType(tail_size, exact)
        exact.get_cell_size_per_token = lambda num_layers: 7260

        expected_tail_bytes = 11 * 2048 * 4 * 128 * 2 * 2
        self.assertEqual(
            tail_size(exact, num_layers=11, max_num_reqs=2048),
            expected_tail_bytes,
        )
        expected_slots = (expected_tail_bytes + 7260 - 1) // 7260
        self.assertEqual(
            reserve(
                exact,
                max_total_num_tokens=100_000,
                num_layers=11,
                max_num_reqs=2048,
            ),
            100_000 - expected_slots,
        )

        other = SimpleNamespace(
            model_config=SimpleNamespace(
                is_glm5_next=False,
                uses_kpool4_compress=False,
            )
        )
        self.assertEqual(tail_size(other, num_layers=11, max_num_reqs=2048), 0)

        init_pool = _find_function(
            _parse(source), "init_memory_pool", "ModelRunnerKVCacheMixin"
        )
        reserve_calls = [
            node
            for node in ast.walk(init_pool)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_reserve_glm5_next_kpool_tail"
        ]
        self.assertEqual(len(reserve_calls), 1)


class TestPoolRegistryOwnership(unittest.TestCase):
    def test_glm_first_match_does_not_steal_minimax_or_dsv4(self):
        registry, minimax_factory = _load_pool_registry()

        class GlmKPool:
            pass

        class DeepSeekV4Pool:
            pass

        def glm_next_kpool_predicate(model_config, server_args):
            hf_config = model_config.hf_config
            return (
                getattr(hf_config, "model_type", None) == "glm5_next"
                and getattr(hf_config, "index_kpool", 1) > 1
            )

        def deepseek_v4_predicate(model_config, server_args):
            return model_config.hf_config.architectures[0] in {
                "DeepseekV4ForCausalLM",
                "DeepseekV4ForCausalLMNextN",
            }

        # Deliberately put GLM first: first-match is safe only if ownership
        # predicates are mutually exclusive and the GLM predicate is narrow.
        registry._KV_POOL_FACTORIES[:] = [
            ("glm5_next_kpool", glm_next_kpool_predicate, GlmKPool),
            (
                "minimax_m3_sparse",
                registry._minimax_sparse_pool_predicate,
                minimax_factory,
            ),
            ("deepseek_v4", deepseek_v4_predicate, DeepSeekV4Pool),
        ]

        minimax = _config("MiniMaxM3SparseForCausalLM", model_type="minimax_m3")
        dsv4 = _config("DeepseekV4ForCausalLM", model_type="deepseek_v4")
        legacy_glm = _config(
            "GlmMoeDsaForCausalLM",
            model_type="glm4_moe",
            index_topk=2048,
            index_kpool=1,
        )

        self.assertEqual(
            registry.resolve_kv_pool_factory(minimax, SimpleNamespace())[0],
            "minimax_m3_sparse",
        )
        self.assertEqual(
            registry.resolve_kv_pool_factory(dsv4, SimpleNamespace())[0],
            "deepseek_v4",
        )
        self.assertIsNone(
            registry.resolve_kv_pool_factory(legacy_glm, SimpleNamespace())
        )


class TestForwardContextPublication(unittest.TestCase):
    def test_linear_sidecar_owns_runner_token_pool_reference(self):
        init = _find_function(
            _parse("python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"),
            "__init__",
            "MambaAttnBackendBase",
        )
        assignments = {
            ast.unparse(node.targets[0]): ast.unparse(node.value)
            for node in ast.walk(init)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }
        annotated_assignments = {
            ast.unparse(node.target): ast.unparse(node.value)
            for node in ast.walk(init)
            if isinstance(node, ast.AnnAssign) and node.value is not None
        }

        self.assertEqual(
            assignments.get("self.token_to_kv_pool"),
            "model_runner.token_to_kv_pool",
        )
        self.assertEqual(
            annotated_assignments.get("self.req_to_token_pool"),
            "model_runner.req_to_token_pool",
        )

    def test_native_sparse_backend_owns_runner_pool_references(self):
        init = _find_function(
            _parse("python/sglang/srt/layers/attention/nsa_backend.py"),
            "__init__",
            "NativeSparseAttnBackend",
        )
        assignments = {
            ast.unparse(node.targets[0]): ast.unparse(node.value)
            for node in ast.walk(init)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
        }

        self.assertEqual(
            assignments.get("self.req_to_token_pool"),
            "model_runner.req_to_token_pool",
        )
        self.assertEqual(
            assignments.get("self.token_to_kv_pool"),
            "model_runner.token_to_kv_pool",
        )

    def test_hybrid_backend_exposes_sidecar_pools_with_legacy_full_backend(self):
        init = _compile_function(
            "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
            "__init__",
            class_name="HybridLinearAttnBackend",
        )
        token_pool = object()
        req_pool = object()
        linear_backend = SimpleNamespace(
            token_to_kv_pool=token_pool,
            req_to_token_pool=req_pool,
        )
        # Legacy full backends on this KT base do not all expose pool attrs.
        # Construction must therefore use the shared linear sidecar contract.
        full_backend = SimpleNamespace()
        wrapper = SimpleNamespace()

        init(wrapper, full_backend, linear_backend, [2, 6])

        self.assertIs(wrapper.token_to_kv_pool, token_pool)
        self.assertIs(wrapper.req_to_token_pool, req_pool)
        self.assertIs(wrapper.full_attn_backend, full_backend)
        self.assertIs(wrapper.linear_attn_backend, linear_backend)

    def test_model_runner_publishes_context_and_preserves_overrides(self):
        tree = _parse("python/sglang/srt/model_executor/model_runner.py")
        forward_raw = _find_function(tree, "_forward_raw", "ModelRunner")
        source = ast.unparse(forward_raw)

        self.assertIn("has_forward_context()", source)
        self.assertIn("self.model_config, 'is_glm5_next', False", source)
        self.assertIn("ForwardContext(attn_backend=self.attn_backend)", source)
        self.assertIn("with ctx_mgr", source)
        self.assertIn("self._forward_raw_with_context", source)

        delegated_calls = [
            node
            for node in ast.walk(forward_raw)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_forward_raw_with_context"
        ]
        self.assertEqual(len(delegated_calls), 1)

    def test_model_runner_forward_context_is_exact_glm_only(self):
        active_context = False
        calls: list[tuple[str, object | None]] = []

        @contextmanager
        def empty_context_stub():
            calls.append(("empty", None))
            yield

        @contextmanager
        def forward_context_stub(context):
            calls.append(("forward", context.attn_backend))
            yield

        method = _compile_function(
            "python/sglang/srt/model_executor/model_runner.py",
            "_forward_raw",
            class_name="ModelRunner",
            globals_={
                "empty_context": empty_context_stub,
                "ForwardContext": SimpleNamespace,
                "forward_context": forward_context_stub,
                "has_forward_context": lambda: active_context,
            },
        )
        backend = object()
        result = object()
        runner = SimpleNamespace(
            model_config=SimpleNamespace(is_glm5_next=False),
            attn_backend=backend,
            _forward_raw_with_context=lambda *args: result,
        )

        self.assertIs(method(runner, object(), False, None), result)
        self.assertEqual(calls, [("empty", None)])

        calls.clear()
        runner.model_config.is_glm5_next = True
        self.assertIs(method(runner, object(), False, None), result)
        self.assertEqual(calls, [("forward", backend)])

        calls.clear()
        active_context = True
        self.assertIs(method(runner, object(), False, None), result)
        self.assertEqual(calls, [("empty", None)])

    def test_forward_context_scope_covers_metadata_and_model_forward(self):
        tree = _parse("python/sglang/srt/model_executor/model_runner.py")
        implementation = _find_function(
            tree, "_forward_raw_with_context", "ModelRunner"
        )
        source = ast.unparse(implementation)
        self.assertIn("self.forward_decode", source)
        self.assertIn("self.forward_extend", source)
        self.assertIn("self.forward_split_prefill", source)
        self.assertIn("self.forward_idle", source)

    def test_decode_graph_capture_publishes_selected_backend(self):
        tree = _parse("python/sglang/srt/model_executor/cuda_graph_runner.py")
        capture = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_capture_one_stream"
        )
        source = ast.unparse(capture)

        self.assertIn("self.model_runner.model_config, 'is_glm5_next', False", source)
        self.assertIn("ForwardContext(attn_backend=attn_backend)", source)
        self.assertIn("ctx_mgr = empty_context()", source)
        self.assertIn("self.capture_one_batch_size", source)
        self.assertLess(
            source.index("ForwardContext(attn_backend=attn_backend)"),
            source.index("self.capture_one_batch_size"),
        )

    def test_hybrid_backend_delegates_indexer_metadata(self):
        method = _compile_function(
            "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
            "get_indexer_metadata",
            class_name="HybridLinearAttnBackend",
        )
        expected = object()
        calls = []
        backend = SimpleNamespace(
            get_indexer_metadata=lambda layer_id, batch: (
                calls.append((layer_id, batch)),
                expected,
            )[1]
        )
        wrapper = SimpleNamespace(full_attn_backend=backend)
        batch = object()

        self.assertIs(method(wrapper, 7, batch), expected)
        self.assertEqual(calls, [(7, batch)])


class TestKPoolBatchOwnedPool(unittest.TestCase):
    _SOURCE = "python/sglang/srt/layers/attention/nsa/nsa_indexer_kpool.py"

    @classmethod
    def setUpClass(cls):
        cls.pool_from_batch = staticmethod(
            _compile_function(cls._SOURCE, "_token_pool_from_batch")
        )

    def _compile_method(self, name):
        def reject_ambient_pool():
            raise AssertionError("ambient ForwardContext pool must not be read")

        return _compile_function(
            self._SOURCE,
            name,
            class_name="IndexerKPool",
            globals_={
                "_token_pool_from_batch": self.pool_from_batch,
                # This makes the test fail if the old ambient lookup returns.
                "get_token_to_kv_pool": reject_ambient_pool,
            },
        )

    def test_pool_helper_returns_batch_pool_and_fails_closed(self):
        pool = object()
        self.assertIs(
            self.pool_from_batch(SimpleNamespace(token_to_kv_pool=pool)), pool
        )
        for batch in (SimpleNamespace(), SimpleNamespace(token_to_kv_pool=None)):
            with self.assertRaisesRegex(RuntimeError, "ForwardBatch.token_to_kv_pool"):
                self.pool_from_batch(batch)

    def test_extend_write_uses_batch_pool_not_ambient_context(self):
        calls = []

        class BatchPool:
            def invalidate_index_buffer_for_layer(self, layer_id):
                calls.append(("invalidate", layer_id))

            def _is_layer_owned(self, layer_id):
                calls.append(("owned", layer_id))
                return True

        empty = SimpleNamespace(is_empty=True)
        batch_pool = BatchPool()
        batch = SimpleNamespace(
            seq_lens_cpu=[4],
            extend_seq_lens_cpu=[4],
            token_to_kv_pool=batch_pool,
        )
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=SimpleNamespace(writes=empty, tails=empty)
            )
        )

        method = self._compile_method("_compress_write_extend")
        result = method(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            None,
            batch,
            9,
            metadata,
        )

        self.assertIsNone(result)
        self.assertEqual(calls, [("invalidate", 9), ("owned", 9)])

    def test_decode_write_uses_batch_pool_during_graph_compatible_forward(self):
        calls = []

        class BatchPool:
            def invalidate_index_buffer_for_layer(self, layer_id):
                calls.append(("invalidate", layer_id))

            def _is_layer_owned(self, layer_id):
                calls.append(("owned", layer_id))
                return True

            def kpool_decode_update_index_cache(self, **kwargs):
                calls.append(("update", kwargs))

        batch_pool = BatchPool()
        batch = SimpleNamespace(
            token_to_kv_pool=batch_pool,
            req_pool_indices=[3, 4],
            out_cache_loc=[30, 40],
        )
        metadata = SimpleNamespace(
            get_page_table_64=lambda: "page-table",
            get_seqlens_int32=lambda: [7, 8],
        )
        self_obj = SimpleNamespace(
            index_kpool_compress_ape="ape",
            scale_fmt=None,
        )

        method = self._compile_method("_compress_write_decode")
        method(
            self_obj,
            SimpleNamespace(shape=(2, 128)),
            "scores",
            [6, 7],
            batch,
            11,
            metadata,
        )

        self.assertEqual(calls[:2], [("invalidate", 11), ("owned", 11)])
        self.assertEqual(calls[2][0], "update")
        update = calls[2][1]
        self.assertEqual(update["layer_id"], 11)
        self.assertEqual(update["block_tables"], "page-table")
        self.assertEqual(update["req_pool_indices"], [3, 4])
        self.assertEqual(update["out_cache_loc"], [30, 40])

    def test_indexer_kpool_has_no_runtime_ambient_pool_reads(self):
        tree = _parse(self._SOURCE)
        indexer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IndexerKPool"
        )
        ambient_calls = [
            node
            for node in ast.walk(indexer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_token_to_kv_pool"
        ]
        self.assertEqual(ambient_calls, [])

    def test_kpool_ragged_row_chunks_slice_all_metadata_and_keep_padding(self):
        iterator_calls = []

        def fake_logits_rows(q_fp8, kv_fp8, weights, ks, ke):
            iterator_calls.append((q_fp8, kv_fp8, weights, ks, ke))
            for start, end in ((0, 2), (2, 4), (4, 5)):
                yield start, end, torch.full((end - start, 7), float(start))

        selector_calls = []

        def select(logits, pool_lens, **kwargs):
            selector_calls.append((logits.clone(), pool_lens.clone(), kwargs))
            rows = logits.shape[0]
            marker = kwargs["row_starts"].to(torch.int32).unsqueeze(1)
            return marker.expand(rows, 11).clone()

        method = _compile_function(
            self._SOURCE,
            "_topk_from_glm5_next_eager_logits_rows",
            class_name="IndexerKPool",
            globals_={
                "torch": torch,
                "iter_glm5_next_eager_fp8_mqa_logits": fake_logits_rows,
            },
        )
        q = torch.zeros((5, 3, 128), dtype=torch.float32)
        k = torch.zeros((7, 128), dtype=torch.float32)
        scale = torch.ones(7)
        weights = torch.ones((5, 3))
        pool_lens = torch.tensor([1, 2, 3, 4, 5])
        seq_lens = torch.tensor([4, 8, 12, 16, 20])
        ks = torch.tensor([0, 1, 3, 6, 10])
        ke = torch.tensor([1, 3, 6, 10, 15])
        topk_offsets = torch.tensor([100, 101, 102, 103, 104])
        page_table = torch.arange(80).reshape(10, 8)
        page_rows = torch.tensor([9, 7, 5, 3, 1])
        runner = SimpleNamespace(
            index_topk=8,
            index_kpool=4,
            _topk_from_kpool_logits=select,
        )

        result = method(
            runner,
            q,
            (k, scale),
            weights,
            pool_lens,
            seq_lens,
            ks,
            ke,
            total_q=7,
            page_table=page_table,
            topk_offsets=topk_offsets,
            page_table_row_index=page_rows,
        )

        self.assertEqual(len(iterator_calls), 1)
        self.assertEqual(len(selector_calls), 3)
        for index, (start, end) in enumerate(((0, 2), (2, 4), (4, 5))):
            logits, lengths, kwargs = selector_calls[index]
            self.assertEqual(tuple(logits.shape), (end - start, 7))
            self.assertTrue(torch.equal(lengths, pool_lens[start:end]))
            self.assertTrue(torch.equal(kwargs["seq_lens"], seq_lens[start:end]))
            self.assertTrue(torch.equal(kwargs["row_starts"], ks[start:end]))
            self.assertTrue(
                torch.equal(kwargs["topk_offsets"], topk_offsets[start:end])
            )
            self.assertTrue(
                torch.equal(kwargs["page_table_row_index"], page_rows[start:end])
            )
            self.assertIs(kwargs["page_table"], page_table)
            self.assertNotIn("out_rows", kwargs)
        self.assertTrue(torch.equal(result[:5, 0], ks.to(torch.int32)))
        self.assertTrue(
            torch.equal(result[5:], torch.full((2, 11), -1, dtype=torch.int32))
        )


if __name__ == "__main__":
    unittest.main()
