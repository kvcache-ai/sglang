from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.lora.layers import (
    TensorOutputLinearWithLoRA,
    get_lora_layer,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock


class _TorchLoRABackend:
    scaling = 0.25

    def run_lora_a_sgemm(self, x, weights):
        # Production backends apply alpha/rank scaling in the A projection.
        return (x @ weights.T) * self.scaling

    def run_lora_b_sgemm(self, x, weights, output_offset, base_output):
        assert output_offset.tolist() == [0, weights.shape[0]]
        return base_output + x @ weights.T


class _IdentitySharedExpert(nn.Module):
    def forward(self, x):
        return x


class _TupleSharedGate(nn.Module):
    def forward(self, x):
        return torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device), None


def _uninitialized_model(model_cls, config):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = config
    return model


def test_tensor_output_linear_preserves_base_contract_and_applies_lora():
    torch.manual_seed(0)
    base = nn.Linear(5, 3, bias=True)
    x = torch.randn(4, 5)
    expected_base = base(x)

    wrapped = get_lora_layer(base, _TorchLoRABackend())
    assert isinstance(wrapped, TensorOutputLinearWithLoRA)
    assert torch.equal(wrapped(x), expected_base)

    lora_a = torch.randn(2, 5)
    lora_b = torch.randn(3, 2)
    wrapped.set_lora_info(lora_a, lora_b)
    expected = (
        expected_base + ((x @ lora_a.T) * wrapped.lora_backend.scaling) @ lora_b.T
    )
    torch.testing.assert_close(wrapped(x), expected)

    assert wrapped.slice_lora_a_weights(lora_a, tp_rank=1) is lora_a
    assert wrapped.slice_lora_b_weights(lora_b, tp_rank=1) is lora_b


def test_shared_expert_accepts_tuple_returning_replicated_lora_gate():
    block = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.shared_expert = _IdentitySharedExpert()
    block.shared_expert_gate = _TupleSharedGate()
    hidden_states = torch.randn(3, 4)

    torch.testing.assert_close(
        block._forward_shared_experts(hidden_states), hidden_states * 0.5
    )


def test_qwen3_moe_all_target_dimensions_include_replicated_router():
    config = SimpleNamespace(
        hidden_size=4096,
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=4,
        num_experts=128,
        intermediate_size=12288,
        vocab_size=151936,
    )
    model = _uninitialized_model(Qwen3MoeForCausalLM, config)

    assert model.get_hidden_dim("qkv_proj", 0) == (4096, 9216)
    assert model.get_hidden_dim("o_proj", 0) == (8192, 4096)
    assert model.get_hidden_dim("gate", 0) == (4096, 128)


def test_qwen35_moe_all_target_dimensions_use_shared_expert_size():
    config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=4096,
        head_dim=256,
        num_attention_heads=32,
        num_key_value_heads=2,
        attn_output_gate=True,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
        shared_expert_intermediate_size=1024,
        num_experts=512,
        vocab_size=151936,
    )
    model = _uninitialized_model(Qwen3_5ForCausalLM, config)

    assert model.get_hidden_dim("gate", 0) == (4096, 512)
    assert model.get_hidden_dim("shared_expert_gate", 0) == (4096, 1)
    assert model.get_hidden_dim("gate_up_proj", 0) == (4096, 2048)
    assert model.get_hidden_dim("down_proj", 0) == (1024, 4096)


def test_qwen35_conditional_filter_accepts_every_nonexpert_training_target():
    model = object.__new__(Qwen3_5MoeForConditionalGeneration)
    accepted = {
        "model.language_model.layers.0.self_attn.qkv_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.in_proj_z",
        "model.language_model.layers.1.linear_attn.in_proj_b",
        "model.language_model.layers.1.linear_attn.in_proj_a",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.2.mlp.gate",
        "model.language_model.layers.2.mlp.shared_expert_gate",
        "model.language_model.layers.2.mlp.shared_expert.gate_up_proj",
        "model.language_model.layers.2.mlp.shared_expert.down_proj",
    }
    for module_name in accepted:
        assert model.should_apply_lora(module_name), module_name

    rejected = {
        "model.visual.blocks.0.attn.qkv_proj",
        "model.language_model.layers.2.mlp.experts",
        "model.language_model.layers.2.mlp.experts.0.gate_proj",
        "model.language_model.layers.2.input_layernorm",
    }
    for module_name in rejected:
        assert not model.should_apply_lora(module_name), module_name
