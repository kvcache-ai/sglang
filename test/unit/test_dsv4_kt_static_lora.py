"""Static BF16 non-experts plus native CPU MXFP4 expert loading contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    _configure_kt_sft_wrapper_for_serving,
    _map_kt_method_to_sft_method,
    _validate_kt_sft_runtime,
)
from sglang.srt.models.deepseek_v4 import _dequant_fp8_wo_a


def test_bf16_config_does_not_invent_a_quantizer():
    config = DeepSeekV4Config()
    assert "quantization_config" not in config.to_dict()
    assert "quantization_config" not in config.to_diff_dict()
    quantization = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert (
        DeepSeekV4Config(quantization_config=quantization).to_dict()[
            "quantization_config"
        ]
        == quantization
    )


def test_mxfp4_serving_dispatch_uses_transpose_free_native_lora(monkeypatch):
    import kt_kernel.sft

    called = []
    monkeypatch.setattr(kt_kernel.sft, "get_mxfp4_runtime", lambda: called.append(True))
    method = _map_kt_method_to_sft_method("MXFP4")
    assert method == "MXFP4_SFT"
    wrapper = SimpleNamespace(share_backward_bb=True)
    _configure_kt_sft_wrapper_for_serving(wrapper, method)
    assert not wrapper.share_backward_bb
    _validate_kt_sft_runtime(method)
    assert called == [True]


@pytest.mark.parametrize(
    "weight_path,gpu_experts,size,expected",
    [
        ("/native", 0, 0, True),
        (None, 0, 0, False),
        ("/native", 1, 0, False),
        ("/native", 0, 4, False),
    ],
)
def test_missing_weight_exemption_requires_empty_cpu_owned_slot(
    weight_path, gpu_experts, size, expected
):
    wrapper = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    wrapper.kt_config = SimpleNamespace(weight_path=weight_path)
    wrapper.num_gpu_experts = gpu_experts
    assert wrapper.is_cpu_owned_checkpoint_parameter(torch.empty(size)) is expected


def test_bf16_grouped_projection_requires_no_fp8_scale():
    weights = {
        "model.layers.0.attn.wo_a.weight": torch.ones(8, 4, 4, dtype=torch.bfloat16),
        "model.layers.0.attn.wq_a.weight": torch.ones(4, 4, dtype=torch.bfloat16),
    }
    result = dict(_dequant_fp8_wo_a(weights.items()))
    assert result.keys() == weights.keys()
    assert all(result[key] is value for key, value in weights.items())


def test_fp8_grouped_projection_still_requires_its_scale():
    weights = [
        ("model.layers.0.attn.wo_a.weight", torch.ones(4, 4).to(torch.float8_e4m3fn))
    ]
    with pytest.raises(AssertionError):
        list(_dequant_fp8_wo_a(weights))
