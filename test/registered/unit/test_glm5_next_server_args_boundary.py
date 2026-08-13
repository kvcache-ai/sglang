"""CPU-only guards for the exact GLM-5-Next Session AB launch boundary."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_ARGS_PATH = REPO_ROOT / "python/sglang/srt/server_args.py"
NSA_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/nsa_backend.py"


def _server_args_tree() -> ast.Module:
    return ast.parse(
        SERVER_ARGS_PATH.read_text(encoding="utf-8"), filename=str(SERVER_ARGS_PATH)
    )


def _find_function(name: str, *, class_name: str | None = None) -> ast.FunctionDef:
    tree = _server_args_tree()
    body = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    return next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _compile_function(name: str, *, class_name: str | None = None, globals_=None):
    function = copy.deepcopy(_find_function(name, class_name=class_name))
    function.decorator_list = []
    module_body: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    symbol = name
    if class_name is None:
        module_body.append(function)
    else:
        symbol = "_ServerArgsHarness"
        module_body.append(
            ast.ClassDef(
                name=symbol,
                bases=[],
                keywords=[],
                body=[function],
                decorator_list=[],
            )
        )
    namespace = {"__builtins__": __builtins__}
    namespace.update(globals_ or {})
    module = ast.fix_missing_locations(ast.Module(body=module_body, type_ignores=[]))
    exec(compile(module, str(SERVER_ARGS_PATH), "exec"), namespace)
    compiled = namespace[symbol]
    return compiled if class_name is None else getattr(compiled, name)


class _LoggerStub:
    def warning(self, *args, **kwargs):
        pass


def _boundary_args(**overrides):
    values = dict(
        speculative_algorithm=None,
        enable_nsa_prefill_context_parallel=False,
        disaggregation_mode="null",
        enable_dp_attention=False,
        enable_two_batch_overlap=False,
        enable_piecewise_cuda_graph=False,
        enable_hierarchical_cache=False,
        enable_lmcache=False,
        enable_hisparse=False,
        enable_multimodal=None,
        is_embedding=False,
        dllm_algorithm=None,
        dllm_algorithm_config=None,
        quantization="fp8",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        moe_dp_size=1,
        attn_cp_size=1,
        ep_size=1,
        moe_a2a_backend="none",
        moe_dense_tp_size=None,
        moe_runner_backend="auto",
        kt_weight_path=None,
        kt_method=None,
        kt_gpu_prefill_token_threshold=None,
        kt_enable_dynamic_expert_update=False,
        disable_shared_experts_fusion=False,
        disable_cuda_graph=False,
        disable_radix_cache=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestGlm5NextSessionABOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validate = staticmethod(
            _compile_function(
                "_validate_glm5_next_session_ab_boundary", class_name="ServerArgs"
            )
        )

    def test_defaults_enable_only_small_decode_graph_batches(self):
        args = _boundary_args()

        self.validate(args)

        self.assertEqual(args.moe_runner_backend, "triton")
        self.assertTrue(args.disable_shared_experts_fusion)
        self.assertFalse(args.disable_cuda_graph)
        self.assertEqual(args.cuda_graph_bs, [1, 2, 4])
        self.assertEqual(args.cuda_graph_max_bs, 4)
        self.assertTrue(args.disable_radix_cache)
        self.assertTrue(args._glm5_next_session_ab_active)

    def test_explicit_cuda_graph_disable_is_preserved(self):
        args = _boundary_args(disable_cuda_graph=True)

        self.validate(args)

        self.assertTrue(args.disable_cuda_graph)
        self.assertFalse(hasattr(args, "cuda_graph_bs"))
        self.assertFalse(hasattr(args, "cuda_graph_max_bs"))

    def test_prefix_cache_is_always_disabled_and_external_caches_are_rejected(self):
        already_disabled = _boundary_args(disable_radix_cache=True)
        self.validate(already_disabled)
        self.assertTrue(already_disabled.disable_radix_cache)

        cases = (
            ({"enable_hierarchical_cache": True}, "HiCache"),
            ({"enable_lmcache": True}, "LMCache"),
            ({"enable_hisparse": True}, "HiSparse"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(_boundary_args(**overrides))

    def test_multimodal_and_non_fp8_weights_are_deferred(self):
        with self.assertRaisesRegex(ValueError, "Session D"):
            self.validate(_boundary_args(enable_multimodal=True))

        for quantization in (None, "mxfp8", "bf16", "compressed-tensors"):
            with self.subTest(quantization=quantization):
                with self.assertRaisesRegex(ValueError, "FP8 weight format"):
                    self.validate(_boundary_args(quantization=quantization))

    def test_non_generation_request_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "is-embedding"):
            self.validate(_boundary_args(is_embedding=True))

        for overrides in (
            {"dllm_algorithm": "LowConfidence"},
            {"dllm_algorithm_config": "stub.yaml"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "diffusion-LLM"):
                    self.validate(_boundary_args(**overrides))

    def test_safe_cache_gate_composes_with_later_generic_validation(self):
        args = _boundary_args()
        args.disaggregation_decode_enable_offload_kvcache = False
        args.swa_full_tokens_ratio = 0.9

        self.validate(args)
        generic_validate = _compile_function(
            "_handle_cache_compatibility", class_name="ServerArgs"
        )
        generic_validate(args)

        self.assertFalse(args.enable_hierarchical_cache)
        self.assertTrue(args.disable_radix_cache)

    def test_pd_dp_attention_tbo_spec_and_cp_are_rejected(self):
        cases = (
            ({"disaggregation_mode": "prefill"}, "PD/disaggregation"),
            ({"disaggregation_mode": "decode"}, "PD/disaggregation"),
            ({"enable_dp_attention": True}, "enable-dp-attention"),
            ({"enable_two_batch_overlap": True}, "two-batch-overlap"),
            (
                {"enable_piecewise_cuda_graph": True},
                "piecewise-cuda-graph",
            ),
            ({"speculative_algorithm": "EAGLE"}, "MTP/speculative"),
            (
                {"enable_nsa_prefill_context_parallel": True},
                "context-parallel",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(_boundary_args(**overrides))

    def test_only_tp1_structural_and_tp8_production_widths_are_accepted(self):
        for tp_size in (1, 8):
            with self.subTest(tp_size=tp_size):
                args = _boundary_args(tp_size=tp_size)
                self.validate(args)
                self.assertEqual(args.tp_size, tp_size)

        for tp_size in (2, 4, 16):
            with self.subTest(tp_size=tp_size):
                with self.assertRaisesRegex(ValueError, "TP=8 production"):
                    self.validate(_boundary_args(tp_size=tp_size))

    def test_only_tensor_parallel_topology_is_accepted(self):
        cases = (
            {"pp_size": 2},
            {"dp_size": 2},
            {"moe_dp_size": 2},
            {"attn_cp_size": 2},
            {"ep_size": 8},
            {"moe_a2a_backend": "deepep"},
            {"moe_dense_tp_size": 1},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "tensor parallelism only"):
                    self.validate(_boundary_args(**overrides))

    def test_only_triton_moe_runner_is_accepted(self):
        explicit_triton = _boundary_args(moe_runner_backend="triton")
        self.validate(explicit_triton)
        self.assertEqual(explicit_triton.moe_runner_backend, "triton")

        for backend in (
            "cutlass",
            "deep_gemm",
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
        ):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "swiglu_limit"):
                    self.validate(_boundary_args(moe_runner_backend=backend))

    def test_kt_offload_accepts_only_checkpoint_native_block_fp8(self):
        args = _boundary_args(
            moe_runner_backend="triton",
            kt_weight_path="stub/experts",
            kt_method="fp8",
        )
        self.validate(args)
        self.assertEqual(args.moe_runner_backend, "triton")

        for method in (
            None,
            "AMXINT4",
            "FP8_PERCHANNEL",
            "MXFP4",
            "MXFP8",
            "RAWINT4",
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "--kt-method FP8"):
                    self.validate(
                        _boundary_args(
                            moe_runner_backend="triton",
                            kt_weight_path="stub/experts",
                            kt_method=method,
                        )
                    )

    def test_layerwise_prefill_requires_tp8_and_static_experts(self):
        accepted = _boundary_args(
            tp_size=8,
            kt_weight_path="stub/experts",
            kt_method="fp8",
            kt_gpu_prefill_token_threshold=4096,
        )
        self.validate(accepted)

        with self.assertRaisesRegex(ValueError, "requires TP=8"):
            self.validate(
                _boundary_args(
                    tp_size=1,
                    kt_weight_path="stub/experts",
                    kt_method="fp8",
                    kt_gpu_prefill_token_threshold=4096,
                )
            )

        with self.assertRaisesRegex(ValueError, "dynamic-expert-update"):
            self.validate(
                _boundary_args(
                    tp_size=8,
                    kt_weight_path="stub/experts",
                    kt_method="fp8",
                    kt_gpu_prefill_token_threshold=4096,
                    kt_enable_dynamic_expert_update=True,
                )
            )

        structural = _boundary_args(
            tp_size=1,
            kt_weight_path="stub/experts",
            kt_method="fp8",
            kt_gpu_prefill_token_threshold=0,
        )
        self.validate(structural)


class TestGlm5NextSessionABNSA(unittest.TestCase):
    @staticmethod
    def _configure():
        return _compile_function(
            "_configure_glm5_next_session_ab_nsa",
            class_name="ServerArgs",
            globals_={"logger": _LoggerStub()},
        )

    @staticmethod
    def _args(**overrides):
        values = dict(
            kv_cache_dtype="fp8_e4m3",
            nsa_prefill_backend=None,
            nsa_decode_backend=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_fp8_blackwell_sets_both_trtllm_dispatcher_paths(self):
        configure = self._configure()
        args = self._args()

        configure(args, 12)

        self.assertEqual(args.nsa_prefill_backend, "trtllm")
        self.assertEqual(args.nsa_decode_backend, "trtllm")

    def test_bf16_and_pre_blackwell_are_rejected(self):
        for dtype in ("bf16", "bfloat16"):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
                    self._configure()(self._args(kv_cache_dtype=dtype), 12)

        with self.assertRaisesRegex(ValueError, "Blackwell"):
            self._configure()(self._args(), 9)

    def test_non_trtllm_sparse_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both NSA prefill and decode"):
            self._configure()(self._args(nsa_prefill_backend="flashmla_sparse"), 12)

    def test_startup_does_not_require_new_flashinfer_abi(self):
        configure = _find_function(
            "_configure_glm5_next_session_ab_nsa", class_name="ServerArgs"
        )
        source = ast.unparse(configure)
        self.assertNotIn("0.6.17", source)
        self.assertNotIn("sparse_mla_top_k_lens", source)

    def test_cuda_graph_kpool_metadata_is_initialized_and_updated(self):
        tree = ast.parse(
            NSA_BACKEND_PATH.read_text(encoding="utf-8"),
            filename=str(NSA_BACKEND_PATH),
        )
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeSparseAttnBackend"
        )
        methods = {
            node.name: ast.unparse(node)
            for node in backend.body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn(
            "self._init_glm5_next_kpool_graph_metadata(metadata, forward_mode)",
            methods["init_forward_metadata_capture_cuda_graph"],
        )
        replay = methods["init_forward_metadata_replay_cuda_graph"]
        self.assertIn(
            "self._update_glm5_next_kpool_graph_metadata(metadata, forward_mode)",
            replay,
        )
        self.assertLess(
            replay.index("metadata.real_page_table[:new_rows, :new_cols].copy_"),
            replay.index("self._update_glm5_next_kpool_graph_metadata"),
        )

        for method_name in (
            "_init_glm5_next_kpool_graph_metadata",
            "_update_glm5_next_kpool_graph_metadata",
        ):
            source = methods[method_name]
            self.assertIn("self.is_glm5_next", source)
            self.assertIn("self.nsa_index_kpool == 4", source)
            self.assertIn("forward_mode.is_decode_or_idle()", source)


class TestGlm5NextBoundaryIsolation(unittest.TestCase):
    def test_exact_gate_is_nested_under_exact_model_detection(self):
        method = _find_function(
            "_handle_model_specific_adjustments", class_name="ServerArgs"
        )
        exact_if_bodies = [
            node.body
            for node in ast.walk(method)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "is_glm5_next"
        ]
        self.assertTrue(exact_if_bodies)
        self.assertTrue(
            any(
                any(
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_validate_glm5_next_session_ab_boundary"
                    for statement in body
                    for call in ast.walk(statement)
                    if isinstance(call, ast.Call)
                )
                for body in exact_if_bodies
            )
        )

    def test_non_glm_cache_moe_and_shared_fusion_defaults_are_unchanged(self):
        tree = _server_args_tree()
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
                "moe_runner_backend",
                "disable_shared_experts_fusion",
                "disable_radix_cache",
                "enable_hierarchical_cache",
                "enable_lmcache",
                "enable_hisparse",
            }
        }
        self.assertEqual(
            defaults,
            {
                "moe_runner_backend": "auto",
                "enable_hierarchical_cache": False,
                "enable_hisparse": False,
                "enable_lmcache": False,
                "disable_radix_cache": False,
                "disable_shared_experts_fusion": False,
            },
        )

    def test_generic_moe_normalization_revalidates_only_exact_glm(self):
        method = _find_function("_handle_moe_kernel_config", class_name="ServerArgs")
        exact_guards = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If)
            and "_glm5_next_session_ab_active" in ast.unparse(node.test)
        ]
        self.assertEqual(len(exact_guards), 1)
        self.assertTrue(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "_validate_glm5_next_session_ab_boundary"
                for call in ast.walk(exact_guards[0])
                if isinstance(call, ast.Call)
            )
        )

        normalize = _compile_function(
            "_handle_moe_kernel_config",
            class_name="ServerArgs",
            globals_={
                "get_bool_env_var": lambda name: False,
                "is_hip": lambda: False,
                "logger": _LoggerStub(),
                "mxfp8_block_convert_required": lambda: False,
            },
        )
        validate = _compile_function(
            "_validate_glm5_next_session_ab_boundary", class_name="ServerArgs"
        )

        def make_args(*, exact_glm: bool, backend: str):
            args = _boundary_args(moe_runner_backend="triton" if exact_glm else backend)
            args.quantization = "fp8"
            args.ep_size = 1
            args.tp_size = 1
            args._validate_glm5_next_session_ab_boundary = lambda: validate(args)
            if exact_glm:
                validate(args)
                # Simulate a later generic rewrite after model-specific setup.
                args.moe_runner_backend = backend
            return args

        with self.assertRaisesRegex(ValueError, "swiglu_limit"):
            normalize(make_args(exact_glm=True, backend="deep_gemm"))

        non_glm = make_args(exact_glm=False, backend="deep_gemm")
        normalize(non_glm)
        self.assertEqual(non_glm.moe_runner_backend, "deep_gemm")

    def test_prefill_page_table_transform_is_glm_only(self):
        tree = ast.parse(
            NSA_BACKEND_PATH.read_text(encoding="utf-8"),
            filename=str(NSA_BACKEND_PATH),
        )
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NativeSparseAttnBackend"
        )
        forward_extend = next(
            node
            for node in backend.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_extend"
        )
        trtllm_calls = [
            node
            for node in ast.walk(forward_extend)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_forward_trtllm"
        ]
        self.assertEqual(len(trtllm_calls), 1)
        is_prefill = next(
            keyword.value
            for keyword in trtllm_calls[0].keywords
            if keyword.arg == "is_prefill"
        )
        self.assertEqual(ast.unparse(is_prefill), "self.is_glm5_next")


if __name__ == "__main__":
    unittest.main()
