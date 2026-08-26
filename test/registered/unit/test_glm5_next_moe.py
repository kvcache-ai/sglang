"""CPU-only contracts for the isolated GLM-5-Next MLP/MoE adapter."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next_moe.py"
DEEPSEEK_PATH = REPO_ROOT / "python/sglang/srt/models/deepseek_v2.py"
WEIGHT_LOADER_PATH = (
    REPO_ROOT / "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py"
)
FUSED_MOE_LAYER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
)
FUSED_MOE_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py"
)
TRITON_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/triton.py"
MOE_RUNNER_BASE_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/base.py"
)
GLM_SWIGLU_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/glm5_next_swiglu.py"
)
FP8_PATH = REPO_ROOT / "python/sglang/srt/layers/quantization/fp8.py"
KT_EP_PATH = REPO_ROOT / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
FLASHINFER_RUNNER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py"
)
TRITON_KERNELS_RUNNER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/triton_kernels.py"
)
MARLIN_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/marlin.py"
CPU_MOE_PATH = REPO_ROOT / "sgl-kernel/csrc/cpu/moe.cpp"
CPU_EXTENSION_PATH = REPO_ROOT / "sgl-kernel/csrc/cpu/torch_extension_cpu.cpp"


def _compile_top_level_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )
    )
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _Projection(nn.Module):
    def __init__(self, output_size: int, input_size: int, prefix: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight_scale_inv = nn.Parameter(
            torch.ones(max(1, output_size // 2), max(1, input_size // 2)),
            requires_grad=False,
        )
        self.prefix = prefix

    def forward(self, x, **kwargs):
        del kwargs
        return F.linear(x, self.weight), None


class _UnclampedSiluAndMul(nn.Module):
    def forward(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


class _DeepseekV2MLPStub(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        quant_config=None,
        reduce_results=True,
        prefix="",
        tp_rank=None,
        tp_size=None,
    ):
        super().__init__()
        del quant_config, tp_rank
        if hidden_act != "silu":
            raise ValueError(hidden_act)
        self.tp_size = tp_size
        self.reduce_results = reduce_results
        self.gate_up_proj = _Projection(
            2 * intermediate_size, hidden_size, f"{prefix}.gate_up_proj"
        )
        self.down_proj = _Projection(
            hidden_size, intermediate_size, f"{prefix}.down_proj"
        )
        self.act_fn = _UnclampedSiluAndMul()

    def forward(self, x, **kwargs):
        del kwargs
        gate_up, _ = self.gate_up_proj(x)
        activated = self.act_fn(gate_up)
        output, _ = self.down_proj(activated)
        return output


class _GateStub(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )


class _ExpertsStub(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        )
        self.w13_weight_scale_inv = nn.Parameter(
            torch.ones(config.n_routed_experts, 1, 1), requires_grad=False
        )
        self.w2_weight_scale_inv = nn.Parameter(
            torch.ones(config.n_routed_experts, 1, 1), requires_grad=False
        )
        self.moe_runner_config = SimpleNamespace(swiglu_limit=config.swiglu_limit)
        if getattr(config, "kt_method", None) is not None:
            self.quant_method = SimpleNamespace(
                kt_config=SimpleNamespace(method=config.kt_method),
                gpu_experts_mask=torch.tensor(
                    config.kt_gpu_experts_mask, dtype=torch.bool
                ),
            )


class _DeepseekV2MoEStub(nn.Module):
    def __init__(
        self,
        config,
        layer_id,
        quant_config=None,
        prefix="",
        alt_stream=None,
        is_nextn=False,
        glm5_next_hf_two_round_swiglu=False,
    ):
        super().__init__()
        del quant_config, alt_stream, is_nextn
        self.config = config
        self.layer_id = layer_id
        self.is_hash = False
        self.use_grouped_topk = config.n_group > config.topk_group
        self.gate = _GateStub(config)
        self.experts = _ExpertsStub(config)
        self.experts.moe_runner_config.glm5_next_hf_two_round_swiglu = (
            glm5_next_hf_two_round_swiglu
        )
        self.topk = SimpleNamespace(
            topk_config=SimpleNamespace(
                use_grouped_topk=self.use_grouped_topk,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
            )
        )
        self.shared_experts = _DeepseekV2MLPStub(
            hidden_size=config.hidden_size,
            intermediate_size=(config.moe_intermediate_size * config.n_shared_experts),
            hidden_act=config.hidden_act,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.base_forward_calls = []

    def forward(self, hidden_states, *args, **kwargs):
        self.base_forward_calls.append((args, kwargs))
        quant_method = getattr(self.experts, "quant_method", None)
        self.base_forward_mode_seen = getattr(
            quant_method, "_glm5_next_forward_mode", None
        )
        if getattr(self, "raise_in_base_forward", False):
            raise RuntimeError("base forward failed")
        return hidden_states + 1


def _load_module():
    packages = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.models",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    quantization = types.ModuleType("sglang.srt.layers.quantization.base_config")
    quantization.QuantizationConfig = type("QuantizationConfig", (), {})
    packages[quantization.__name__] = quantization

    deepseek = types.ModuleType("sglang.srt.models.deepseek_v2")
    deepseek.DeepseekV2MLP = _DeepseekV2MLPStub
    deepseek.DeepseekV2MoE = _DeepseekV2MoEStub
    packages[deepseek.__name__] = deepseek

    module_name = "_glm5_next_moe_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    packages[module_name] = module
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


def _config(**overrides):
    values = dict(
        hidden_size=4,
        intermediate_size=3,
        moe_intermediate_size=2,
        hidden_act="silu",
        n_routed_experts=3,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        swiglu_limit=10.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestGlm5NextSwiGLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_cpu_numerics_match_clamp_before_silu_contract(self):
        # gate = [12, -20, 2], up = [15, -15, .5].  The negative gate is
        # intentionally below -limit: GLM must not lower-clamp it.
        gate_up = torch.tensor(
            [[12.0, -20.0, 2.0, 15.0, -15.0, 0.5]], dtype=torch.float32
        )
        actual = self.module.glm5_next_swiglu(gate_up, 10.0)
        gate, up = gate_up.chunk(2, dim=-1)
        expected = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        symmetric_gate_result = F.silu(gate.clamp(-10.0, 10.0)) * up.clamp(-10.0, 10.0)
        self.assertNotEqual(actual[0, 1].item(), symmetric_gate_result[0, 1].item())

    def test_limit_is_semantically_different_from_kimi_swiglu(self):
        gate_up = torch.tensor([[100.0, 100.0]])
        glm = self.module.glm5_next_swiglu(gate_up, 10.0)
        kimi = F.silu(gate_up[:, :1]) * gate_up[:, 1:]

        self.assertLess(glm.item(), kimi.item() / 50)

    def test_none_retains_ordinary_swiglu_and_odd_width_is_rejected(self):
        torch.manual_seed(0)
        gate_up = torch.randn(2, 8, dtype=torch.float64)
        gate, up = gate_up.chunk(2, dim=-1)
        torch.testing.assert_close(
            self.module.glm5_next_swiglu(gate_up, None),
            F.silu(gate) * up,
        )
        with self.assertRaisesRegex(ValueError, "even gate/up width"):
            self.module.Glm5NextSiluAndMul(10.0)(torch.randn(2, 7))

    def test_bfloat16_materializes_silu_before_multiply(self):
        # This pair separates Transformers' two BF16 operations from the
        # former fused CUDA implementation, which rounded only once after an
        # FP32 SiLU-and-multiply expression.
        gate_up = torch.tensor(
            [[-11.25, -6.125]],
            dtype=torch.bfloat16,
        )
        actual = self.module.Glm5NextSiluAndMul(10.0)(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        expected = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        fused_one_round = (
            F.silu(gate.float().clamp(max=10.0))
            * up.float().clamp(-10.0, 10.0)
        ).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertNotEqual(actual.item(), fused_one_round.item())

    def test_private_primitive_cpu_reference_and_output_reuse(self):
        spec = importlib.util.spec_from_file_location(
            "_glm5_next_private_swiglu_test", GLM_SWIGLU_PATH
        )
        private_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(private_module)

        gate_up = torch.tensor([[-11.25, -6.125]], dtype=torch.bfloat16)
        output = torch.empty((1, 1), dtype=torch.bfloat16)
        actual = private_module.glm5_next_hf_two_round_swiglu(
            gate_up, 10.0, output=output
        )
        gate, up = gate_up.chunk(2, dim=-1)
        expected = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        fused_one_round = (
            F.silu(gate.float().clamp(max=10.0))
            * up.float().clamp(-10.0, 10.0)
        ).to(torch.bfloat16)

        self.assertIs(actual, output)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertNotEqual(actual.item(), fused_one_round.item())

        with self.assertRaisesRegex(ValueError, "output shape mismatch"):
            private_module.glm5_next_hf_two_round_swiglu(
                gate_up, 10.0, output=torch.empty((1, 2), dtype=torch.bfloat16)
            )


class TestGlm5NextMLPAndMoE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_dense_mlp_preserves_projection_layout_and_is_numerically_clamped(self):
        mlp = self.module.Glm5NextMLP(
            hidden_size=2,
            intermediate_size=2,
            hidden_act="silu",
            prefix="model.layers.0.mlp",
            swiglu_limit=1.0,
        )
        with torch.no_grad():
            mlp.gate_up_proj.weight.copy_(
                torch.tensor(
                    [
                        [2.0, 0.0],
                        [0.0, -3.0],
                        [4.0, 0.0],
                        [0.0, 5.0],
                    ]
                )
            )
            mlp.down_proj.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        x = torch.tensor([[2.0, 2.0]])
        gate_up = F.linear(x, mlp.gate_up_proj.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        expected = F.linear(
            F.silu(gate.clamp(max=1.0)) * up.clamp(-1.0, 1.0),
            mlp.down_proj.weight,
        )

        torch.testing.assert_close(mlp(x), expected, rtol=0, atol=0)
        self.assertEqual(mlp.gate_up_proj.prefix, "model.layers.0.mlp.gate_up_proj")
        self.assertEqual(mlp.down_proj.prefix, "model.layers.0.mlp.down_proj")
        self.assertFalse(mlp.down_proj is None)

    def test_moe_forces_official_grouped_noaux_and_patches_only_shared_activation(self):
        moe = self.module.Glm5NextMoE(
            config=_config(),
            layer_idx=3,
            prefix="model.layers.3.mlp",
        )

        self.assertEqual(moe.layer_id, 3)
        self.assertEqual(moe.layer_idx, 3)
        self.assertTrue(moe.use_grouped_topk)
        self.assertTrue(moe.topk.topk_config.use_grouped_topk)
        self.assertEqual(moe.topk.topk_config.num_expert_group, 1)
        self.assertEqual(moe.topk.topk_config.topk_group, 1)
        self.assertIs(
            moe.topk.topk_config.correction_bias,
            moe.gate.e_score_correction_bias,
        )
        self.assertEqual(moe.experts.moe_runner_config.swiglu_limit, 10.0)
        self.assertTrue(
            moe.experts.moe_runner_config.glm5_next_hf_two_round_swiglu
        )
        self.assertIsInstance(moe.shared_experts.act_fn, self.module.Glm5NextSiluAndMul)
        self.assertEqual(moe.shared_experts.swiglu_limit, 10.0)
        self.assertFalse(moe.shared_experts.reduce_results)

    def test_hf_weight_names_keep_deepseek_packed_and_fused_expert_contract(self):
        dense = self.module.Glm5NextMLP(
            hidden_size=4,
            intermediate_size=3,
            hidden_act="silu",
            swiglu_limit=10.0,
        )
        dense_names = set(dict(dense.named_parameters()))
        self.assertIn("gate_up_proj.weight", dense_names)
        self.assertIn("gate_up_proj.weight_scale_inv", dense_names)
        self.assertIn("down_proj.weight", dense_names)
        self.assertIn("down_proj.weight_scale_inv", dense_names)

        moe = self.module.Glm5NextMoE(config=_config(), layer_id=3)
        moe_names = set(dict(moe.named_parameters()))
        expected_names = {
            "gate.weight",
            "gate.e_score_correction_bias",
            "experts.w13_weight",
            "experts.w2_weight",
            "experts.w13_weight_scale_inv",
            "experts.w2_weight_scale_inv",
            "shared_experts.gate_up_proj.weight",
            "shared_experts.down_proj.weight",
        }
        self.assertTrue(expected_names.issubset(moe_names))

        loader_source = WEIGHT_LOADER_PATH.read_text(encoding="utf-8")
        self.assertIn('("gate_up_proj", "gate_proj", 0)', loader_source)
        self.assertIn('("gate_up_proj", "up_proj", 1)', loader_source)
        self.assertIn("if self.num_fused_shared_experts > 0", loader_source)
        self.assertIn('"mlp.shared_experts"', loader_source)
        self.assertIn('f"mlp.experts.{self.config.n_routed_experts}"', loader_source)

    def test_layer_id_alias_validation_and_base_forward_delegation(self):
        with self.assertRaisesRegex(ValueError, "Conflicting GLM MoE layer ids"):
            self.module.Glm5NextMoE(config=_config(), layer_id=3, layer_idx=4)
        with self.assertRaisesRegex(TypeError, "requires layer_id"):
            self.module.Glm5NextMoE(config=_config())

        # A no-limit compatibility config is safe on CPU and proves the GLM
        # wrapper delegates execution/reduction behavior to the base class.
        moe = self.module.Glm5NextMoE(config=_config(swiglu_limit=None), layer_id=3)
        hidden = torch.zeros(2, 4)
        actual = moe(hidden, "forward-batch", use_reduce_scatter=True)
        torch.testing.assert_close(actual, hidden + 1)
        self.assertEqual(
            moe.base_forward_calls,
            [(("forward-batch",), {"use_reduce_scatter": True})],
        )
        self.assertNotIn("forward_normal", self.module.Glm5NextMoE.__dict__)
        self.assertNotIn("forward_deepep", self.module.Glm5NextMoE.__dict__)

    def test_required_layerwise_mode_is_scoped_to_one_base_forward(self):
        moe = self.module.Glm5NextMoE(
            config=_config(swiglu_limit=None), layer_id=3
        )
        quant_method = SimpleNamespace(
            kt_config=SimpleNamespace(
                is_glm5_next=True,
                method="FP8",
                gpu_prefill_token_threshold=4096,
            )
        )
        moe.experts.quant_method = quant_method
        forward_mode = object()
        forward_batch = SimpleNamespace(forward_mode=forward_mode)

        moe(torch.zeros(1, 4), forward_batch=forward_batch)

        self.assertIs(moe.base_forward_mode_seen, forward_mode)
        self.assertFalse(hasattr(quant_method, "_glm5_next_forward_mode"))

        moe.raise_in_base_forward = True
        with self.assertRaisesRegex(RuntimeError, "base forward failed"):
            moe(torch.zeros(1, 4), forward_batch=forward_batch)
        self.assertFalse(hasattr(quant_method, "_glm5_next_forward_mode"))

    def test_cpu_amx_moe_fails_fast_until_both_expert_abis_gain_limit(self):
        moe = self.module.Glm5NextMoE(config=_config(), layer_id=3)
        with self.assertRaisesRegex(
            NotImplementedError, "current AMX ABI has no swiglu_limit"
        ):
            moe(torch.zeros(1, 4))

        extension_source = CPU_EXTENSION_PATH.read_text(encoding="utf-8")
        cpu_source = CPU_MOE_PATH.read_text(encoding="utf-8")
        fused_schema = extension_source.split('"fused_experts_cpu(Tensor', 1)[1].split(
            ");", 1
        )[0]
        shared_schema = extension_source.split('"shared_expert_cpu(Tensor', 1)[1].split(
            ");", 1
        )[0]
        self.assertNotIn("swiglu_limit", fused_schema)
        self.assertNotIn("swiglu_limit", shared_schema)

        # These are the two minimum C++ extension points: routed/shared paths
        # both eventually call the same unclamped vector activation helper.
        activation_body = cpu_source.split("inline void silu_and_mul(", 1)[1].split(
            "template <typename scalar_t, int BLOCK_M", 1
        )[0]
        self.assertIn("x0 = x0 / (one + x0.neg().exp_u20())", activation_body)
        self.assertNotIn("clamp", activation_body)

    def test_kt_cpu_experts_accept_only_checkpoint_native_block_fp8(self):
        for method in (
            "AMXINT4",
            "FP8_PERCHANNEL",
            "MXFP4",
            "MXFP8",
            "RAWINT4",
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                    NotImplementedError, "require --kt-method FP8"
                ):
                    self.module.Glm5NextMoE(
                        config=_config(
                            kt_method=method,
                            kt_gpu_experts_mask=[True, False, True],
                        ),
                        layer_id=3,
                    )

        # The same format is harmless when every routed expert remains on GPU.
        all_gpu = self.module.Glm5NextMoE(
            config=_config(
                kt_method="AMXINT4",
                kt_gpu_experts_mask=[True, True, True],
            ),
            layer_id=3,
        )
        self.assertEqual(all_gpu.layer_id, 3)

        supported = self.module.Glm5NextMoE(
            config=_config(
                kt_method="FP8",
                kt_gpu_experts_mask=[False, False, False],
            ),
            layer_id=3,
        )
        self.assertEqual(supported.layer_id, 3)


class TestExistingKernelReuseEvidence(unittest.TestCase):
    def test_glm_two_round_flag_is_private_default_off_and_reaches_both_runners(self):
        base_source = MOE_RUNNER_BASE_PATH.read_text(encoding="utf-8")
        deepseek_source = DEEPSEEK_PATH.read_text(encoding="utf-8")
        layer_source = FUSED_MOE_LAYER_PATH.read_text(encoding="utf-8")
        fused_source = FUSED_MOE_PATH.read_text(encoding="utf-8")
        triton_source = TRITON_RUNNER_PATH.read_text(encoding="utf-8")
        kt_source = KT_EP_PATH.read_text(encoding="utf-8")
        private_source = GLM_SWIGLU_PATH.read_text(encoding="utf-8")

        private_flag = "glm5_next_hf_two_round_swiglu"
        self.assertIn(f"{private_flag}: bool = False", base_source)
        self.assertIn(f"{private_flag}: bool = False", deepseek_source)
        self.assertIn(f"{private_flag}: bool = False", layer_source)
        self.assertIn(f"{private_flag}={private_flag}", deepseek_source)
        self.assertIn(f"{private_flag}={private_flag}", layer_source)
        self.assertIn(f"moe_runner_config.{private_flag}", fused_source)
        self.assertIn(f"self.config.{private_flag}", triton_source)
        self.assertIn(
            f'getattr(runner_config, "{private_flag}", False)', kt_source
        )

        # Two independent launches plus an in-place BF16 output reload are the
        # source-level guard against a compiler collapsing both rounds again.
        self.assertIn("_glm5_next_silu_to_bf16_kernel[grid]", private_source)
        self.assertIn("_glm5_next_bf16_mul_kernel[grid]", private_source)
        self.assertIn(
            "activated_gate = tl.load(output_ptr + offsets", private_source
        )

    def test_deepseek_base_supplies_noaux_shared_tp_and_ep_contracts(self):
        source = DEEPSEEK_PATH.read_text(encoding="utf-8")
        self.assertIn('if config.topk_method == "noaux_tc"', source)
        self.assertIn("self.e_score_correction_bias = nn.Parameter", source)
        self.assertIn("self.num_fused_shared_experts", source)
        self.assertIn("self.shared_experts_is_fp8", source)
        self.assertIn("self._enable_a2a_moe", source)
        self.assertIn("return self.forward_deepep(", source)
        self.assertIn("tensor_model_parallel_all_reduce(final_hidden_states)", source)

    def test_standard_gpu_fp8_chain_carries_limit_to_exact_clamp(self):
        deepseek_source = DEEPSEEK_PATH.read_text(encoding="utf-8")
        layer_source = FUSED_MOE_LAYER_PATH.read_text(encoding="utf-8")
        fp8_source = FP8_PATH.read_text(encoding="utf-8")
        triton_source = TRITON_RUNNER_PATH.read_text(encoding="utf-8")
        fused_source = FUSED_MOE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'swiglu_limit=getattr(config, "swiglu_limit", None)',
            deepseek_source,
        )
        self.assertIn("swiglu_limit=swiglu_limit", layer_source)
        self.assertIn("moe_runner_backend = MoeRunnerBackend.TRITON", fp8_source)
        self.assertIn("moe_runner_config=runner_config", triton_source)
        self.assertIn("swiglu_limit=moe_runner_config.swiglu_limit", fused_source)
        self.assertIn("gate = gate.clamp(min=None, max=swiglu_limit)", fused_source)
        self.assertIn(
            "up = up.clamp(min=-swiglu_limit, max=swiglu_limit)",
            fused_source,
        )
        self.assertIn("return F.silu(gate) * up", fused_source)

    def test_kt_offload_preserves_limit_for_block_fp8_without_widening_others(self):
        source = KT_EP_PATH.read_text(encoding="utf-8")
        supports = _compile_top_level_function(
            KT_EP_PATH, "_kt_supports_swiglu_parameters"
        )

        self.assertTrue(supports("FP8", True))
        self.assertFalse(supports("FP8", False))
        for method in ("MXFP4", "MXFP8"):
            with self.subTest(method=method):
                self.assertTrue(supports(method, False))
        for method in (None, "BF16", "RAWINT4", "FP8_PERCHANNEL"):
            with self.subTest(method=method):
                self.assertFalse(supports(method, True))

        self.assertIn('_cfg_swglim = getattr(_mrc, "swiglu_limit", None)', source)
        self.assertIn("is_glm5_next: bool = False", source)
        self.assertIn(
            'is_glm5_next=getattr(server_args, "_glm5_next_session_ab_active", False)',
            source,
        )
        self.assertIn("supports_swiglu_params = _kt_supports_swiglu_parameters", source)
        self.assertIn("swiglu_limit=_kt_swiglu_limit", source)
        self.assertIn("_kt_swiglu_limit = 0.0", source)

    def test_deepgemm_ep_path_is_still_dsv4_gated_and_not_claimed_reusable(self):
        path = REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn(
            'is_2604b = envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B"', source
        )
        self.assertIn("swiglu_limit must be non-None iff submode=2604B", source)

    def test_other_explicit_gpu_backends_do_not_yet_expose_glm_limit(self):
        fp8_source = FP8_PATH.read_text(encoding="utf-8")
        cutlass_call = fp8_source.split("output = cutlass_fused_experts_fp8(", 1)[
            1
        ].split("return StandardCombineInput(hidden_states=output)", 1)[0]
        self.assertNotIn("swiglu_limit", cutlass_call)

        flashinfer_source = FLASHINFER_RUNNER_PATH.read_text(encoding="utf-8")
        triton_kernels_source = TRITON_KERNELS_RUNNER_PATH.read_text(encoding="utf-8")
        marlin_source = MARLIN_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("swiglu_limit", flashinfer_source)
        self.assertNotIn("swiglu_limit", triton_kernels_source)
        self.assertNotIn("swiglu_limit", marlin_source)


if __name__ == "__main__":
    unittest.main()
