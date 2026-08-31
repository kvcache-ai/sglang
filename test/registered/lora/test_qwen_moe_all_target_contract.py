from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.lora.layers import (
    TensorOutputLinearWithLoRA,
    get_lora_layer,
)
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.triton_ops import chunked_sgmv_lora_expand_forward
from sglang.srt.lora.triton_ops.chunked_sgmv_expand import (
    _chunked_lora_expand_kernel,
)
from sglang.srt.lora.triton_ops.chunked_sgmv_shrink import (
    _chunked_lora_shrink_kernel,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM


def test_csgmv_cache_keys_cover_shape_dependent_constants():
    expand = {
        "NUM_SLICES": 1,
        "OUTPUT_DIM": 4096,
        "MAX_RANK": 8,
        "BLOCK_M": 16,
        "BLOCK_N": 64,
        "BLOCK_K": 16,
    }
    expand_key = _chunked_lora_expand_kernel.key_fn((), expand)
    for name, other in (
        ("NUM_SLICES", 2),
        ("OUTPUT_DIM", 32),
        ("MAX_RANK", 16),
        ("BLOCK_M", 32),
        ("BLOCK_N", 32),
        ("BLOCK_K", 32),
    ):
        assert (
            _chunked_lora_expand_kernel.key_fn((), {**expand, name: other})
            != expand_key
        )

    shrink = {
        "N": 8,
        "K": 512,
        "NUM_SLICES": 1,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "BLOCK_K": 256,
    }
    shrink_key = _chunked_lora_shrink_kernel.key_fn((), shrink)
    for name, other in (
        ("N", 16),
        ("K", 1024),
        ("NUM_SLICES", 2),
        ("BLOCK_M", 32),
        ("BLOCK_N", 32),
        ("BLOCK_K", 128),
    ):
        assert (
            _chunked_lora_shrink_kernel.key_fn((), {**shrink, name: other})
            != shrink_key
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_csgmv_expand_does_not_reuse_kernel_across_output_dimensions():
    _chunked_lora_expand_kernel._clear_cache()
    tokens, rank = 19, 8
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=2,
        seg_indptr=torch.tensor([0, 16, 19], dtype=torch.int32, device="cuda"),
        weight_indices=torch.zeros(2, dtype=torch.int32, device="cuda"),
        lora_ranks=torch.tensor([rank], dtype=torch.int32, device="cuda"),
        scalings=torch.tensor([2.0], dtype=torch.float32, device="cuda"),
        max_len=16,
        seg_lens=None,
        permutation=torch.arange(tokens, dtype=torch.int64, device="cuda"),
    )
    hidden = torch.randn(tokens, rank, dtype=torch.bfloat16, device="cuda")

    for output_dim in (6144, 32):
        weights = torch.randn(
            1, output_dim, rank, dtype=torch.bfloat16, device="cuda"
        )
        base = torch.randn(
            tokens, output_dim, dtype=torch.bfloat16, device="cuda"
        )
        expected = base + (hidden @ weights[0].T) * 2.0
        actual = chunked_sgmv_lora_expand_forward(
            hidden,
            weights,
            batch_info,
            torch.tensor([0, output_dim], dtype=torch.int32, device="cuda"),
            output_dim,
            base_output=base,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    assert len(_chunked_lora_expand_kernel.kernel_cache) == 2
    _chunked_lora_expand_kernel._clear_cache()


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


def test_qwen35_conditional_filter_accepts_runtime_nonexpert_targets():
    model = object.__new__(Qwen3_5MoeForConditionalGeneration)
    accepted = {
        # Full-attention projections are direct decoder-layer children.
        "model.language_model.layers.0.qkv_proj",
        "model.language_model.layers.0.o_proj",
        "model.layers.3.qkv_proj",
        "model.layers.3.o_proj",
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


def test_qwen35_flat_attention_modules_are_wrapped_and_audited():
    class _RuntimeModel(nn.Module):
        should_apply_lora = Qwen3_5MoeForConditionalGeneration.should_apply_lora

        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
            self.model.layers[3].qkv_proj = nn.Linear(4, 6, bias=False)
            self.model.layers[3].o_proj = nn.Linear(6, 4, bias=False)

    manager = LoRAManager.__new__(LoRAManager)
    manager.base_model = _RuntimeModel()
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=4)
    manager.lora_backend = _TorchLoRABackend()
    manager.target_modules = {"qkv_proj", "o_proj"}
    manager.init_lora_modules()

    assert set(manager.lora_modules[3]) == {
        "model.layers.3.qkv_proj",
        "model.layers.3.o_proj",
    }
    adapter = SimpleNamespace(
        layers=[
            SimpleNamespace(weights={}),
            SimpleNamespace(weights={}),
            SimpleNamespace(weights={}),
            SimpleNamespace(
                weights={
                    "model.layers.3.self_attn.qkv_proj.lora_A.weight": torch.empty(
                        2, 4
                    ),
                    "model.layers.3.self_attn.qkv_proj.lora_B.weight": torch.empty(
                        6, 2
                    ),
                    "model.layers.3.self_attn.o_proj.lora_A.weight": torch.empty(
                        2, 6
                    ),
                    "model.layers.3.self_attn.o_proj.lora_B.weight": torch.empty(
                        4, 2
                    ),
                }
            ),
        ],
        embedding_layers={},
        _source_lora_weight_names=(),
    )
    manager.validate_lora_weight_consumption(adapter, "qwen35")
