"""CPU coverage for CUDA-IPC model-loader lifecycle fixups."""

import torch
from torch import nn

from sglang.srt.layers.moe.topk import TopK, TopKConfig
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.test.test_utils import CustomTestCase


def _make_topk(correction_bias: torch.Tensor) -> TopK:
    topk = TopK.__new__(TopK)
    nn.Module.__init__(topk)
    topk.topk_config = TopKConfig(top_k=1, correction_bias=correction_bias)
    return topk


class TestIpcModelLoaderLifecycle(CustomTestCase):
    def test_restores_fp8_scale_format_metadata(self):
        tensor = nn.Parameter(torch.ones(1))

        IpcModelLoader._restore_tensor_metadata(tensor, {"format_ue8m0": True})

        self.assertTrue(tensor.format_ue8m0)

    def test_rebinds_cached_topk_tensor_by_identity(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Module()
                self.gate.correction_bias = nn.Parameter(torch.empty(4))
                self.cached_topk = _make_topk(self.gate.correction_bias)
                self.unrelated_topk = _make_topk(nn.Parameter(torch.zeros(4)))

        model = Model()
        stale_bias, replacement = IpcModelLoader._set_module_tensor(
            model, "gate.correction_bias", torch.ones(4)
        )
        IpcModelLoader._finalize_model_after_ipc_mapping(
            model, {id(stale_bias): replacement}
        )

        self.assertIs(model.cached_topk.topk_config.correction_bias, replacement)
        self.assertIsNot(model.unrelated_topk.topk_config.correction_bias, replacement)

    def test_runs_post_load_hook_after_ipc_mapping(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.derived_weight = None

            def post_load_weights(self):
                self.derived_weight = self.weight + 1

        model = Model()
        IpcModelLoader._finalize_model_after_ipc_mapping(model, {})
        self.assertTrue(torch.equal(model.derived_weight, torch.tensor([2.0])))
