from types import SimpleNamespace

import torch
from torch import nn

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.lora.layers import TensorOutputLinearWithLoRA, get_lora_layer
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _dispatch_mla_subtype,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
    MoEGate,
)


class _TorchLoRABackend:
    def run_lora_a_sgemm(self, x, weights):
        return x @ weights.T

    def run_lora_b_sgemm(self, x, weights, output_offset, base_output):
        assert output_offset.tolist() == [0, weights.shape[0]]
        return base_output + x @ weights.T


class _TupleProjection(nn.Module):
    def __init__(self, weight, lora_a, lora_b):
        super().__init__()
        self.weight = weight
        self.lora_a = lora_a
        self.lora_b = lora_b

    def forward(self, x):
        return x @ self.weight.T + (x @ self.lora_a.T) @ self.lora_b.T, None


def _deepseek_config():
    return SimpleNamespace(
        hidden_size=32,
        q_lora_rank=8,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        num_attention_heads=4,
        n_routed_experts=6,
        intermediate_size=24,
        moe_intermediate_size=12,
        n_shared_experts=1,
    )


def test_deepseek_router_uses_tensor_output_replicated_lora(monkeypatch):
    gate = MoEGate.__new__(MoEGate)
    nn.Module.__init__(gate)
    gate.in_features = 5
    gate.out_features = 3
    gate.weight = nn.Parameter(torch.randn(3, 5))
    gate.register_parameter("bias", None)
    gate.e_score_correction_bias = None
    monkeypatch.setattr(
        deepseek_v2,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_deterministic_inference=True),
    )

    x = torch.randn(4, 5)
    base = torch.nn.functional.linear(x, gate.weight)
    wrapped = get_lora_layer(gate, _TorchLoRABackend())
    assert isinstance(wrapped, TensorOutputLinearWithLoRA)
    assert "in_features=5" in repr(gate)
    assert torch.equal(wrapped(x, None, forward_batch=None), base)
    assert wrapped(x).dtype == torch.float32

    lora_a = torch.randn(2, 5)
    lora_b = torch.randn(3, 2)
    wrapped.set_lora_info(lora_a, lora_b)
    torch.testing.assert_close(
        wrapped(x, None, forward_batch=None),
        base + (x @ lora_a.T) @ lora_b.T,
    )
    assert wrapped.slice_lora_a_weights(lora_a, tp_rank=1) is lora_a
    assert wrapped.slice_lora_b_weights(lora_b, tp_rank=1) is lora_b


def test_deepseek_q_a_and_kv_a_keep_independent_lora_pairs():
    torch.manual_seed(0)
    hidden_size, q_rank, kv_out, lora_rank = 7, 3, 5, 2
    x = torch.randn(4, hidden_size)
    q_weight = torch.randn(q_rank, hidden_size)
    kv_weight = torch.randn(kv_out, hidden_size)
    q_a, q_b = torch.randn(lora_rank, hidden_size), torch.randn(q_rank, lora_rank)
    kv_a = torch.randn(lora_rank, hidden_size)
    kv_b = torch.randn(kv_out, lora_rank)

    attn = DeepseekV2AttentionMLA.__new__(DeepseekV2AttentionMLA)
    nn.Module.__init__(attn)
    attn.q_lora_rank = q_rank
    attn.fuse_qkv_a_proj = False
    attn.q_a_proj = _TupleProjection(q_weight, q_a, q_b)
    attn.kv_a_proj_with_mqa = _TupleProjection(kv_weight, kv_a, kv_b)

    actual = attn.prepare_qkv_latent(x, forward_batch=None)
    expected_q = x @ q_weight.T + (x @ q_a.T) @ q_b.T
    expected_kv = x @ kv_weight.T + (x @ kv_a.T) @ kv_b.T
    torch.testing.assert_close(actual, torch.cat((expected_q, expected_kv), dim=-1))

    wrong_shared_a = torch.cat(
        (
            x @ q_weight.T + (x @ q_a.T) @ q_b.T,
            x @ kv_weight.T + (x @ q_a.T) @ kv_b.T,
        ),
        dim=-1,
    )
    assert not torch.allclose(actual, wrong_shared_a)


def test_deepseek_all_target_dimensions_cover_mla_router_and_shared_mlp():
    model = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
    nn.Module.__init__(model)
    model.config = _deepseek_config()

    assert model.get_hidden_dim("q_a_proj", 0) == (32, 8)
    assert model.get_hidden_dim("q_b_proj", 0) == (8, 32)
    assert model.get_hidden_dim("kv_a_proj_with_mqa", 0) == (32, 20)
    assert model.get_hidden_dim("kv_b_proj", 0) == (16, 32)
    assert model.get_hidden_dim("o_proj", 0) == (16, 32)
    assert model.get_hidden_dim("gate", 0) == (32, 6)
    assert model.get_hidden_dim("gate_up_proj", 0) == (32, 48)
    assert model.get_hidden_dim("down_proj", 0) == (24, 32)


def test_qkv_a_base_fusion_is_disabled_only_for_lora(monkeypatch):
    config = _deepseek_config()
    monkeypatch.setattr(
        deepseek_v2,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_lora=False),
    )
    assert deepseek_v2._should_fuse_qkv_a_proj(config)

    monkeypatch.setattr(
        deepseek_v2,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_lora=True),
    )
    assert not deepseek_v2._should_fuse_qkv_a_proj(config)


def test_no_fuse_attention_cannot_select_cpu_fused_mla_path():
    attn = SimpleNamespace(rocm_fused_decode_mla=False)
    assert _dispatch_mla_subtype(attn, forward_batch=None) == AttnForwardMethod.MLA


def test_weight_loader_uses_model_fusion_policy():
    class _FakeLoader(nn.Module, DeepseekV2WeightLoaderMixin):
        def __init__(self, fuse):
            super().__init__()
            self.fuse_qkv_a_proj = fuse
            self.config = SimpleNamespace(q_lora_rank=2, n_routed_experts=0)
            self.quant_config = None
            self.num_fused_shared_experts = 0
            self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].self_attn = nn.Module()
            if fuse:
                projection = nn.Module()
                projection.weight = nn.Parameter(torch.zeros(5, 4))
                self.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa = projection
            else:
                q_projection = nn.Module()
                q_projection.weight = nn.Parameter(torch.zeros(2, 4))
                kv_projection = nn.Module()
                kv_projection.weight = nn.Parameter(torch.zeros(3, 4))
                self.model.layers[0].self_attn.q_a_proj = q_projection
                self.model.layers[0].self_attn.kv_a_proj_with_mqa = kv_projection

        def post_load_weights(self, **kwargs):
            pass

    q_weight = torch.randn(2, 4)
    kv_weight = torch.randn(3, 4)
    weights = [
        ("model.layers.0.self_attn.q_a_proj.weight", q_weight),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_weight),
    ]

    separate = _FakeLoader(fuse=False)
    separate.do_load_weights(iter(weights))
    torch.testing.assert_close(
        separate.model.layers[0].self_attn.q_a_proj.weight, q_weight
    )
    torch.testing.assert_close(
        separate.model.layers[0].self_attn.kv_a_proj_with_mqa.weight, kv_weight
    )

    fused = _FakeLoader(fuse=True)
    fused.do_load_weights(iter(weights))
    torch.testing.assert_close(
        fused.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa.weight,
        torch.cat((q_weight, kv_weight), dim=0),
    )
