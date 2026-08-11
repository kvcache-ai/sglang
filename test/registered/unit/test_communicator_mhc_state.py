"""CPU micro tests for the non-CP mHC communicator state machine."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from enum import Enum, auto
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_communicator_module():
    mhc = _load_source(
        "_glm5_next_mhc_for_communicator_test",
        REPO_ROOT / "python/sglang/kernels/ops/layernorm/mhc.py",
    )

    stubs = {}
    for name in (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.layernorm",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.model_executor",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        stubs[name] = package
    stubs["sglang.kernels.ops.layernorm.mhc"] = mhc

    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.tensor_model_parallel_all_reduce = lambda x: x
    stubs[distributed.__name__] = distributed

    class ScatterMode(Enum):
        SCATTERED = auto()
        TP_ATTN_FULL = auto()
        FULL = auto()

    def _unused_simple(*args, **kwargs):
        del args, kwargs
        raise AssertionError("stub communication path must not run")

    def _unused_gather(*args, **kwargs):
        del args, kwargs
        raise AssertionError("stub gather path must not run")

    def _unused_trivial(*args, **kwargs):
        del args, kwargs
        raise AssertionError("stub layer-output path must not run")

    class CommunicateContext:
        attn_dp_size = 1
        attn_cp_size = 1

        @classmethod
        def init_new(cls):
            return types.SimpleNamespace(
                attn_dp_size=cls.attn_dp_size,
                attn_cp_size=cls.attn_cp_size,
                attn_tp_size=1,
            )

    class CommunicateSimpleFn:
        @staticmethod
        def get_fn(*args, **kwargs):
            del args, kwargs
            return CommunicateSimpleFn._trivial

        @staticmethod
        def _trivial(hidden_states, **kwargs):
            del kwargs
            return hidden_states

    class CommunicateWithAllReduceAndLayerNormFn:
        _simple = staticmethod(_unused_simple)
        _gather_hidden_states_and_residual = staticmethod(_unused_gather)

        @staticmethod
        def get_fn(*args, **kwargs):
            del args, kwargs
            return CommunicateWithAllReduceAndLayerNormFn._simple

    class CommunicateSummableTensorPairFn:
        _trivial = staticmethod(_unused_trivial)

        @staticmethod
        def get_fn(*args, **kwargs):
            del args, kwargs
            return CommunicateSummableTensorPairFn._trivial

    class LayerCommunicator:
        def __init__(
            self,
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter=False,
            is_last_layer=False,
            qkv_latent_func=None,
        ):
            self.layer_scatter_modes = layer_scatter_modes
            self.input_layernorm = input_layernorm
            self.post_attention_layernorm = post_attention_layernorm
            self.allow_reduce_scatter = allow_reduce_scatter
            self.is_last_layer = is_last_layer
            self.qkv_latent_func = qkv_latent_func
            self._context = CommunicateContext.init_new()
            self._post_init_communicate()

    class AttentionInputs:
        def __init__(self, hidden_states, forward_batch, qkv_latent_func):
            self.hidden_states = hidden_states
            self.forward_batch = forward_batch
            self.qkv_latent_func = qkv_latent_func

        def fetch_qkv_latent(self):
            return self.qkv_latent_func(self.hidden_states, self.forward_batch)

    class _AttnTpContext:
        input_scattered = False
        attn_inputs = None

        def set_attn_inputs(self, attn_inputs):
            self.attn_inputs = attn_inputs

        def fetch_qkv_latent(self):
            assert self.attn_inputs is not None
            return self.attn_inputs.fetch_qkv_latent()

    attn_tp_context = _AttnTpContext()

    communicator = types.ModuleType("sglang.srt.layers.communicator")
    communicator.AttentionInputs = AttentionInputs
    communicator.CommunicateContext = CommunicateContext
    communicator.CommunicateSimpleFn = CommunicateSimpleFn
    communicator.CommunicateSummableTensorPairFn = CommunicateSummableTensorPairFn
    communicator.CommunicateWithAllReduceAndLayerNormFn = (
        CommunicateWithAllReduceAndLayerNormFn
    )
    communicator.LayerCommunicator = LayerCommunicator
    communicator.LayerScatterModes = object
    communicator.ScatterMode = ScatterMode
    communicator.get_attn_tp_context = lambda: attn_tp_context
    stubs[communicator.__name__] = communicator

    forward_batch_info = types.ModuleType(
        "sglang.srt.model_executor.forward_batch_info"
    )
    forward_batch_info.ForwardBatch = object
    stubs[forward_batch_info.__name__] = forward_batch_info

    with patch.dict(sys.modules, stubs):
        return _load_source(
            "_glm5_next_communicator_mhc_under_test",
            REPO_ROOT / "python/sglang/srt/layers/communicator_mhc.py",
        )


COMMUNICATOR_MHC = _load_communicator_module()


class _ScaleNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, scale: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-5
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class TestMHCState(unittest.TestCase):
    def test_attention_ffn_and_reset_lifecycle(self):
        calls = []

        def attn_pre(x, norm_weight, norm_eps):
            calls.append(("attn_pre", norm_weight, norm_eps))
            tokens = x.shape[0]
            return (
                x[:, :2],
                torch.full((tokens, 4), 11.0),
                torch.full((tokens, 2), 13.0),
                False,
            )

        def ffn_pre(x, norm_weight, norm_eps):
            calls.append(("ffn_pre", norm_weight, norm_eps))
            tokens = x.shape[0]
            return (
                x[:, :2],
                torch.full((tokens, 4), 17.0),
                torch.full((tokens, 2), 19.0),
                False,
            )

        def hc_post(x, residual, h_res, h_post):
            calls.append(("post", h_res.clone(), h_post.clone()))
            self.assertEqual(h_res.shape[-1], 4)
            self.assertEqual(h_post.shape[-1], 2)
            return residual + x.repeat(1, 2)

        state = COMMUNICATOR_MHC.MHCState(
            hc_mult=2,
            hc_attn_pre=attn_pre,
            hc_ffn_pre=ffn_pre,
            hc_post=hc_post,
        )
        norm = _ScaleNorm(hidden_size=2, scale=3.0)
        flat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        attn_input, attn_residual = state.attn_split(flat, out_norm=norm)
        torch.testing.assert_close(attn_input, torch.tensor([[3.0, 6.0]]))
        torch.testing.assert_close(attn_residual, flat)
        self.assertEqual(calls[0][1].data_ptr(), norm.weight.data_ptr())
        self.assertEqual(calls[0][2], norm.variance_epsilon)

        ffn_input, ffn_residual = state.attn_to_mlp(
            torch.tensor([[5.0, 7.0]]), attn_residual, out_norm=norm
        )
        torch.testing.assert_close(ffn_residual, torch.tensor([[6.0, 9.0, 8.0, 11.0]]))
        torch.testing.assert_close(ffn_input, torch.tensor([[18.0, 27.0]]))

        combined = state.mlp_combine(torch.tensor([[1.0, 1.0]]), ffn_residual)
        torch.testing.assert_close(combined, torch.tensor([[7.0, 10.0, 9.0, 12.0]]))
        self.assertEqual(calls[-1][1][0, 0].item(), 17.0)
        self.assertEqual(calls[-1][2][0, 0].item(), 19.0)

        state.reset_aux()
        self.assertIsNone(state.h_res)
        self.assertIsNone(state.h_post)

    def test_fused_norm_flag_avoids_second_normalization(self):
        def pre(x, norm_weight, norm_eps):
            del norm_weight, norm_eps
            tokens = x.shape[0]
            return x[:, :2], torch.zeros(tokens, 4), torch.zeros(tokens, 2), True

        state = COMMUNICATOR_MHC.MHCState(2, pre, pre, lambda *args: args[1])
        flat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output, _ = state.attn_split(flat, out_norm=_ScaleNorm(2, 10.0))
        torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))


class TestMHCLayerCommunicatorTopology(unittest.TestCase):
    @staticmethod
    def _modes():
        mode = COMMUNICATOR_MHC.ScatterMode.TP_ATTN_FULL
        return types.SimpleNamespace(
            layer_input_mode=mode,
            attn_mode=mode,
            mlp_mode=mode,
            middle_residual_mode=mode,
            layer_output_mode=mode,
        )

    def _construct(
        self,
        *,
        is_first_layer=False,
        is_last_layer=False,
        pre=None,
        post=None,
        qkv_latent_func=None,
    ):
        norm = _ScaleNorm(hidden_size=2, scale=1.0)

        if pre is None:

            def pre(*args):
                return args[0][:, :2], None, None, False

        if post is None:

            def post(*args):
                return args[1]

        return COMMUNICATOR_MHC.MHCLayerCommunicator(
            layer_scatter_modes=self._modes(),
            input_layernorm=norm,
            post_attention_layernorm=norm,
            is_first_layer=is_first_layer,
            is_last_layer=is_last_layer,
            hc_mult=2,
            hc_attn_pre=pre,
            hc_ffn_pre=pre,
            hc_post=post,
            qkv_latent_func=qkv_latent_func,
        )

    def test_dsa_latent_is_installed_after_mhc_attention_preprocess(self):
        batch = object()
        calls = []

        def prepare_qkv_latent(hidden_states, forward_batch):
            calls.append((hidden_states.clone(), forward_batch))
            return hidden_states + 7

        communicator = self._construct(qkv_latent_func=prepare_qkv_latent)
        hidden_states, _ = communicator.prepare_attn(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            None,
            batch,
        )
        latent = COMMUNICATOR_MHC.get_attn_tp_context().fetch_qkv_latent()

        torch.testing.assert_close(calls[0][0], hidden_states)
        self.assertIs(calls[0][1], batch)
        torch.testing.assert_close(latent, hidden_states + 7)

    def test_attention_dp_and_cp_fail_during_construction(self):
        context = COMMUNICATOR_MHC.CommunicateContext
        for field in ("attn_dp_size", "attn_cp_size"):
            setattr(context, field, 2)
            try:
                with self.assertRaisesRegex(NotImplementedError, "DP=1 and CP=1"):
                    self._construct()
            finally:
                setattr(context, field, 1)

    def test_runtime_scattered_input_fails_before_mhc_math(self):
        communicator = self._construct()
        attn_context = COMMUNICATOR_MHC.get_attn_tp_context()
        attn_context.input_scattered = True
        try:
            with self.assertRaisesRegex(NotImplementedError, "scattered TP input"):
                communicator.prepare_attn(
                    torch.zeros(1, 4),
                    None,
                    object(),
                )
        finally:
            attn_context.input_scattered = False

    def test_first_and_last_layer_expand_then_contract(self):
        pre_widths = []

        def pre(x, norm_weight, norm_eps):
            del norm_weight, norm_eps
            pre_widths.append(x.shape[-1])
            tokens = x.shape[0]
            return (
                x[:, :2],
                torch.zeros(tokens, 4),
                torch.zeros(tokens, 2),
                False,
            )

        def post(x, residual, h_res, h_post):
            del h_res, h_post
            return residual + x.repeat(1, 2)

        communicator = self._construct(
            is_first_layer=True,
            is_last_layer=True,
            pre=pre,
            post=post,
        )
        hidden_states, residual = communicator.prepare_attn(
            torch.tensor([[1.0, 2.0]]),
            None,
            object(),
        )
        hidden_states, residual = communicator.prepare_mlp(
            torch.zeros_like(hidden_states),
            residual,
            object(),
        )
        hidden_states, residual = communicator.postprocess_layer(
            torch.zeros_like(hidden_states),
            residual,
            object(),
        )

        self.assertEqual(pre_widths, [4, 4])
        torch.testing.assert_close(hidden_states, torch.tensor([[1.0, 2.0]]))
        self.assertIsNone(residual)


if __name__ == "__main__":
    unittest.main()
