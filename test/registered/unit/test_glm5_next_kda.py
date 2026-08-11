"""CPU-only contracts for the isolated GLM-5-Next KDA implementation."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OPS_PATH = REPO_ROOT / (
    "python/sglang/srt/layers/attention/linear/kernels/glm5_next_kda_ops.py"
)
KERNEL_PATH = REPO_ROOT / (
    "python/sglang/srt/layers/attention/linear/kernels/glm5_next_kda.py"
)
BACKEND_PATH = REPO_ROOT / (
    "python/sglang/srt/layers/attention/linear/glm5_next_kda_backend.py"
)
KIMI_BACKEND_PATH = REPO_ROOT / (
    "python/sglang/srt/layers/attention/linear/kda_backend.py"
)
KIMI_KERNEL_PATH = REPO_ROOT / (
    "python/sglang/srt/layers/attention/linear/kernels/kda_triton.py"
)
KIMI_FLA_PATH = REPO_ROOT / "python/sglang/srt/layers/attention/fla/kda.py"


def _load_ops():
    spec = importlib.util.spec_from_file_location("_glm5_next_kda_ops_test", OPS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_kernel_adapter(ops):
    class RecordingTritonKDAKernel:
        def extend(self, *args, **kwargs):
            self.extend_call = (args, kwargs)
            return "chunk-kda-output"

    packages = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.linear",
        "sglang.srt.layers.attention.linear.kernels",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    ops_name = "sglang.srt.layers.attention.linear.kernels.glm5_next_kda_ops"
    base_name = "sglang.srt.layers.attention.linear.kernels.kda_triton"
    base_module = types.ModuleType(base_name)
    base_module.TritonKDAKernel = RecordingTritonKDAKernel
    packages[ops_name] = ops
    packages[base_name] = base_module

    spec = importlib.util.spec_from_file_location(
        "_glm5_next_kda_adapter_test", KERNEL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module.Glm5NextTritonKDAKernel


def _small_decode_reference(
    *,
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    lower_bound,
    states,
    state_indices,
    query_start_loc,
):
    """Independent matrix-form reference for the bounded delta recurrence."""

    output = torch.empty_like(v)
    head_dim = q.shape[-1]
    scale = head_dim**-0.5
    num_heads = q.shape[2]
    bias = dt_bias.float().reshape(num_heads, head_dim)
    gate_scale = A_log.float().reshape(num_heads).exp()

    for sequence_id in range(query_start_loc.numel() - 1):
        bos = int(query_start_loc[sequence_id])
        eos = int(query_start_loc[sequence_id + 1])
        state_id = int(state_indices[sequence_id])
        state = states[state_id].float().clone()
        for token_id in range(bos, eos):
            for head_id in range(num_heads):
                q_row = q[0, token_id, head_id].float()
                k_row = k[0, token_id, head_id].float()
                q_row /= torch.sqrt(q_row @ q_row + 1e-6)
                k_row /= torch.sqrt(k_row @ k_row + 1e-6)
                q_row *= scale

                gate = lower_bound * torch.sigmoid(
                    gate_scale[head_id]
                    * (raw_gate[0, token_id, head_id] + bias[head_id])
                )
                beta = torch.sigmoid(raw_beta[0, token_id, head_id].float())
                state[head_id] *= gate.exp().unsqueeze(-1)
                delta = v[0, token_id, head_id].float() - (k_row @ state[head_id])
                delta *= beta
                state[head_id] += torch.outer(k_row, delta)
                output[0, token_id, head_id] = q_row @ state[head_id]
        states[state_id].copy_(state.to(states.dtype))
    return output


class TestGlm5NextKDAReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = _load_ops()

    def test_raw_gate_and_raw_beta_match_scalar_formula(self):
        raw_gate = torch.tensor([[[0.0, 1.0, -1.0, 2.0], [0.5, -0.5, 1.5, -1.5]]])
        raw_beta = torch.tensor([[[0.0, 2.0], [-2.0, 1.0]]])
        A_log = torch.tensor([0.0, math.log(2.0)]).view(1, 1, 2, 1)
        dt_bias = torch.tensor([0.25, -0.25, 0.5, -0.5])

        gate = self.ops.glm5_next_safe_gate(
            raw_gate,
            A_log,
            2,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )
        expected_gate = torch.empty(1, 2, 2, 2)
        for token_id in range(2):
            for head_id in range(2):
                for dim_id in range(2):
                    raw = float(raw_gate[0, token_id, head_id * 2 + dim_id])
                    biased = raw + float(dt_bias[head_id * 2 + dim_id])
                    scale = math.exp(float(A_log.reshape(-1)[head_id]))
                    expected_gate[0, token_id, head_id, dim_id] = -5.0 / (
                        1.0 + math.exp(-(scale * biased))
                    )

        self.assertTrue(torch.allclose(gate, expected_gate, atol=1e-6, rtol=0))
        expected_beta = 1.0 / (1.0 + torch.exp(-raw_beta.float()))
        self.assertTrue(torch.equal(raw_beta.float().sigmoid(), expected_beta))

    def test_prefill_and_decode_small_reference(self):
        torch.manual_seed(7)
        tokens, heads, head_dim, value_dim = 3, 2, 2, 3
        q = torch.randn(1, tokens, heads, head_dim)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, heads, value_dim)
        raw_gate = torch.randn(1, tokens, heads, head_dim)
        raw_beta = torch.randn(1, tokens, heads)
        A_log = torch.tensor([0.0, math.log(1.5)]).view(1, 1, heads, 1)
        dt_bias = torch.linspace(-0.2, 0.3, heads * head_dim)
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
        state_indices = torch.tensor([1, 3], dtype=torch.int32)
        initial_states = torch.randn(5, heads, head_dim, value_dim) * 0.1
        actual_states = initial_states.clone()
        expected_states = initial_states.clone()

        actual = self.ops.glm5_next_safe_decode(
            A_log=A_log,
            raw_gate=raw_gate,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            q=q,
            k=k,
            v=v,
            raw_beta=raw_beta,
            state_source=actual_states,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
        )
        expected = _small_decode_reference(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            states=expected_states,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(actual_states, expected_states, atol=1e-6, rtol=1e-6)
        )

    def test_prefill_adapter_activates_raw_inputs_and_maps_padding_to_slot_zero(self):
        kernel_cls = _load_kernel_adapter(self.ops)
        kernel = kernel_cls()
        q = torch.ones(1, 2, 2, 2)
        raw_gate = torch.tensor([[[0.0, 1.0, -1.0, 2.0], [0.5, 0.0, 1.0, -0.5]]])
        raw_beta = torch.tensor([[[0.0, 2.0], [-2.0, 1.0]]])
        A_log = torch.zeros(1, 1, 2, 1)
        dt_bias = torch.zeros(4)
        cache_indices = torch.tensor([-1, 3], dtype=torch.int64)

        result = kernel.extend(
            q,
            q,
            q,
            raw_gate,
            raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            ssm_states=torch.zeros(4, 2, 2, 2),
            cache_indices=cache_indices,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        self.assertEqual(result, "chunk-kda-output")
        args, kwargs = kernel.extend_call
        expected_gate = self.ops.glm5_next_safe_gate(
            raw_gate,
            A_log,
            2,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )
        self.assertTrue(torch.equal(args[3], expected_gate))
        self.assertTrue(torch.equal(args[4], raw_beta.float().sigmoid()))
        self.assertTrue(
            torch.equal(
                kwargs["cache_indices"],
                torch.tensor([0, 3], dtype=torch.int32),
            )
        )

    def test_padding_trim_and_restore(self):
        mixed_qkv = torch.arange(5 * 12, dtype=torch.float32).view(5, 12)
        raw_gate = torch.arange(1 * 5 * 4, dtype=torch.float32).view(1, 5, 4)
        raw_beta = torch.arange(1 * 5 * 2, dtype=torch.float32).view(1, 5, 2)
        query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)

        qkv, gate, beta, physical = self.ops.trim_glm5_next_kda_padding(
            mixed_qkv, raw_gate, raw_beta, query_start_loc
        )
        self.assertEqual(qkv.shape, (3, 12))
        self.assertEqual(gate.shape, (1, 3, 4))
        self.assertEqual(beta.shape, (1, 3, 2))
        self.assertEqual(physical, 5)

        logical_output = torch.ones(1, 3, 2, 4)
        restored = self.ops.restore_glm5_next_kda_padding(logical_output, physical)
        self.assertEqual(restored.shape, (1, 5, 2, 4))
        self.assertTrue(torch.equal(restored[:, :3], logical_output))
        self.assertEqual(torch.count_nonzero(restored[:, 3:]).item(), 0)

    def test_large_decode_grid_splits_only_past_cuda_z_limit(self):
        normal_grid, normal_split = self.ops.glm5_next_decode_grid(1, 4, 511, 128)
        split_grid, split = self.ops.glm5_next_decode_grid(1, 4, 512, 128)
        self.assertEqual(normal_grid, (1, 4, 65408))
        self.assertFalse(normal_split)
        self.assertEqual(split_grid, (4, 512, 128))
        self.assertTrue(split)


class TestGlm5NextKDAIsolation(unittest.TestCase):
    def test_backend_contract_requires_raw_inputs_and_has_no_rope(self):
        source = BACKEND_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        backend = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Glm5NextKDAAttnBackend"
        )
        module_doc = ast.get_docstring(tree)
        self.assertIn("raw gate and raw beta logits", module_doc)
        identifiers = {
            node.id for node in ast.walk(backend) if isinstance(node, ast.Name)
        } | {node.attr for node in ast.walk(backend) if isinstance(node, ast.Attribute)}
        self.assertFalse(any("rope" in name.lower() for name in identifiers))
        self.assertFalse(any("position" in name.lower() for name in identifiers))

    def test_glm_kernel_requires_explicit_lower_bound(self):
        tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Glm5NextTritonKDAKernel"
        )
        methods = {
            node.name: node for node in kernel.body if isinstance(node, ast.FunctionDef)
        }
        for method_name in ("decode", "extend"):
            method = methods[method_name]
            keyword_defaults = dict(
                zip(method.args.kwonlyargs, method.args.kw_defaults)
            )
            lower_bound_arg = next(
                arg for arg in method.args.kwonlyargs if arg.arg == "lower_bound"
            )
            self.assertIsNone(keyword_defaults[lower_bound_arg])

        source = KERNEL_PATH.read_text(encoding="utf-8")
        self.assertIn("beta.float().sigmoid()", source)
        self.assertIn("glm5_next_safe_gate", source)

    def test_kimi_kernel_launch_autotune_tf32_and_padding_are_untouched(self):
        backend_source = KIMI_BACKEND_PATH.read_text(encoding="utf-8")
        kernel_source = KIMI_KERNEL_PATH.read_text(encoding="utf-8")
        fla_source = KIMI_FLA_PATH.read_text(encoding="utf-8")

        self.assertNotIn("Glm5Next", backend_source)
        self.assertNotIn("lower_bound", backend_source)
        self.assertNotIn("trim_glm5_next_kda_padding", backend_source)
        self.assertNotIn("Glm5Next", kernel_source)
        self.assertNotIn("lower_bound", kernel_source)
        self.assertIn("BT_LIST_AUTOTUNE = [32, 64, 128]", fla_source)
        self.assertIn('key=["H", "D"]', fla_source)
        self.assertIn("allow_tf32=False", fla_source)


if __name__ == "__main__":
    unittest.main()
