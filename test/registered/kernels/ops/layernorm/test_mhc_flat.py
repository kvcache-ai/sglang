"""Micro tests for the flat GLM-5-Next mHC functional contract."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[5]
MHC_PATH = REPO_ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"


def _load_mhc_module():
    spec = importlib.util.spec_from_file_location("_glm5_next_mhc_flat", MHC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MHC = _load_mhc_module()


class TestMHCFlatReference(unittest.TestCase):
    TOKENS = 3
    HC_MULT = 4
    HIDDEN = 5

    def _inputs(self, device="cpu"):
        generator = torch.Generator(device=device).manual_seed(20260808)
        total = self.HC_MULT * self.HIDDEN
        mix_width = self.HC_MULT * (2 + self.HC_MULT)
        x = torch.randn(self.TOKENS, total, generator=generator, device=device)
        fn = torch.randn(mix_width, total, generator=generator, device=device)
        scale = torch.randn(3, generator=generator, device=device)
        base = torch.randn(mix_width, generator=generator, device=device)
        return x, fn, scale, base

    def _hc_pre(self, x, fn, scale, base):
        with patch.dict(
            os.environ,
            {"SGLANG_OPT_USE_TILELANG_MHC_PRE": "0"},
        ):
            return MHC.hc_pre(
                x=x,
                hc_fn=fn,
                hc_scale=scale,
                hc_base=base,
                hc_mult=self.HC_MULT,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=8,
            )

    def test_expand_contract_round_trip(self):
        x = torch.randn(self.TOKENS, self.HIDDEN)
        expanded = MHC.hc_expand(x, self.HC_MULT)
        self.assertEqual(expanded.shape, (self.TOKENS, 20))
        torch.testing.assert_close(
            MHC.hc_contract(expanded, self.HC_MULT), x, rtol=0, atol=0
        )

    def test_zero_projection_has_exact_pre_and_post_gate(self):
        x, fn, scale, base = self._inputs()
        fn.zero_()
        scale.fill_(1)
        base.zero_()
        layer_input, h_res, h_post, norm_fused = self._hc_pre(x, fn, scale, base)

        expected_pre = 0.5 + 1e-6
        expected_input = (
            x.reshape(self.TOKENS, self.HC_MULT, self.HIDDEN).sum(dim=1) * expected_pre
        )
        torch.testing.assert_close(layer_input, expected_input, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(h_post, torch.ones_like(h_post), rtol=0, atol=0)
        self.assertFalse(norm_fused)

        comb = h_res.reshape(self.TOKENS, self.HC_MULT, self.HC_MULT)
        torch.testing.assert_close(
            comb.sum(dim=1),
            torch.ones((self.TOKENS, self.HC_MULT)),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_flat_frontend_matches_three_dimensional_reference(self):
        x, fn, scale, base = self._inputs()
        layer_input, h_res, h_post, norm_fused = self._hc_pre(x, fn, scale, base)
        post, comb, expected_input = MHC._mhc_pre_torch(
            residual=x.reshape(self.TOKENS, self.HC_MULT, self.HIDDEN),
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=8,
        )
        torch.testing.assert_close(layer_input, expected_input)
        torch.testing.assert_close(h_res, comb.flatten(1))
        torch.testing.assert_close(h_post, post.flatten(1))
        self.assertFalse(norm_fused)

    def test_bfloat16_state_keeps_fp32_mix_metadata(self):
        x, fn, scale, base = self._inputs()
        x = x.to(torch.bfloat16)
        layer_input, h_res, h_post, _ = self._hc_pre(x, fn, scale, base)
        output = MHC.hc_post(
            x=layer_input, residual=x, h_post=h_post, h_res=h_res, hc_mult=self.HC_MULT
        )
        self.assertEqual(layer_input.dtype, torch.bfloat16)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(h_res.dtype, torch.float32)
        self.assertEqual(h_post.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())

    def test_post_uses_input_to_output_combination_orientation(self):
        x = torch.tensor([[10.0, 20.0]])
        residual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        h_post = torch.tensor([[2.0, 3.0]])
        # Output branch 0 takes residual branch 0; output branch 1 takes branch 1.
        h_res = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        with patch.dict(
            os.environ,
            {"SGLANG_OPT_USE_TILELANG_MHC_POST": "0"},
        ):
            actual = MHC.hc_post(x, residual, h_post, h_res, hc_mult=2)
        expected = torch.tensor([[21.0, 42.0, 33.0, 64.0]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_shapes_preserve_flat_contract(self):
        total = self.HC_MULT * self.HIDDEN
        mix_width = self.HC_MULT * (2 + self.HC_MULT)
        x = torch.empty((0, total))
        fn = torch.empty((mix_width, total), dtype=torch.float32)
        scale = torch.empty((3,), dtype=torch.float32)
        base = torch.empty((mix_width,), dtype=torch.float32)
        layer_input, h_res, h_post, norm_fused = self._hc_pre(x, fn, scale, base)
        self.assertEqual(layer_input.shape, (0, self.HIDDEN))
        self.assertEqual(h_res.shape, (0, self.HC_MULT * self.HC_MULT))
        self.assertEqual(h_post.shape, (0, self.HC_MULT))
        self.assertFalse(norm_fused)
        self.assertEqual(
            MHC.hc_post(layer_input, x, h_post, h_res, self.HC_MULT).shape,
            (0, total),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_reference_matches_cpu(self):
        cpu_inputs = self._inputs("cpu")
        cpu_outputs = self._hc_pre(*cpu_inputs)[:3]
        cuda_inputs = tuple(t.cuda() for t in cpu_inputs)
        cuda_outputs = self._hc_pre(*cuda_inputs)[:3]
        for cpu, cuda in zip(cpu_outputs, cuda_outputs):
            torch.testing.assert_close(cuda.cpu(), cpu, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
