"""Native April checkpoints preserve all supported LoRA tensors without export."""

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.srt.lora import kt_dsv4


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    model = dict(
        num_hidden_layers=43,
        n_routed_experts=256,
        n_shared_experts=1,
        hidden_size=16,
        moe_intermediate_size=16,
        q_lora_rank=8,
        num_attention_heads=2,
        head_dim=4,
        o_groups=2,
        o_lora_rank=4,
        compress_ratios=[0, 0] + [4, 128] * 20 + [4],
        swiglu_limit=10,
    )
    config = dict(
        peft_type="LORA",
        r=8,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=list(kt_dsv4.TARGETS),
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    ordinary, experts = {}, {}
    for layer in range(43):
        for target, (n, k) in kt_dsv4._shapes(model, layer).items():
            prefix = f"base_model.model.model.layers.{layer}.{target}"
            for kind, shape in {"A": (8, k), "B": (n, 8)}.items():
                ordinary[f"{prefix}.lora_{kind}.weight"] = torch.ones(shape)
        for projection in ("gate", "up", "down"):
            for kind, shape in {"a": (256, 8, 16), "b": (256, 16, 8)}.items():
                experts[f"layers.{layer}.experts.{projection}_lora_{kind}"] = (
                    torch.full(shape, layer / 43, dtype=torch.bfloat16)
                )
    save_file(ordinary, str(tmp_path / "adapter_model.safetensors"))
    save_file(experts, str(tmp_path / kt_dsv4.FUSED))
    manifest = dict(
        version=1,
        status="ready",
        expert_weight_format="mxfp4",
        lora=dict(rank=8, alpha=16),
        base=dict(fingerprint="same-base"),
    )

    def seal():
        manifest["artifacts"] = {
            name: dict(
                size=(tmp_path / name).stat().st_size,
                sha256=hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            )
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                kt_dsv4.FUSED,
            )
        }
        (tmp_path / kt_dsv4.MANIFEST).write_text(json.dumps(manifest))

    seal()
    import kt_kernel.sft.deepseek_v4 as native

    monkeypatch.setattr(
        native,
        "inspect_native_checkpoint",
        lambda _: dict(fingerprint="same-base", config=model),
    )
    return tmp_path, model, manifest, seal


def test_native_checkpoint_maps_targets_and_silently_skips_compressor(adapter, caplog):
    path, model, _, _ = adapter
    kt_dsv4.validate_adapter(path, "/same-base")
    result = kt_dsv4.load_nonexpert(path, model)
    assert len(result) == 602
    assert all("compressor" not in key and "indexer" not in key for key in result)
    assert "model.layers.0.self_attn.wq_a.lora_A.weight" in result
    assert "model.layers.42.mlp.shared_experts.up_proj.lora_B.weight" in result
    assert len(kt_dsv4.runtime_config(path)["target_modules"]) == 7
    assert not caplog.records


def test_native_experts_are_exact_cpu_views_with_expected_layout(adapter):
    path, _, _, _ = adapter
    result = kt_dsv4.load_expert_layer(path, 7, 256, 16, 16, torch.bfloat16)
    saved = load_file(str(path / kt_dsv4.FUSED))
    assert result.pop("rank") == 8 and result.pop("alpha") == 16
    for name, tensor in result.items():
        assert tensor.device.type == "cpu" and tensor.is_contiguous()
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, saved[f"layers.7.experts.{name}"])


@pytest.mark.parametrize("mutation", ["missing", "indexer", "shape", "nan"])
def test_unconsumed_or_malformed_ordinary_weights_are_rejected(adapter, mutation):
    path, model, _, _ = adapter
    tensors = load_file(str(path / "adapter_model.safetensors"))
    key = next(key for key in tensors if ".self_attn.q_b_proj.lora_A." in key)
    if mutation == "missing":
        tensors.pop(key)
    elif mutation == "indexer":
        tensors[key.replace("self_attn.", "self_attn.indexer.")] = tensors.pop(key)
    elif mutation == "shape":
        tensors[key] = tensors[key][:1].clone()
    else:
        tensors[key][0, 0] = float("nan")
    save_file(tensors, str(path / "adapter_model.safetensors"))
    with pytest.raises(ValueError):
        kt_dsv4.load_nonexpert(path, model)


def test_bad_integrity_and_base_are_rejected(adapter):
    path, _, manifest, seal = adapter
    with (path / "adapter_config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Invalid April adapter file"):
        kt_dsv4.validate_adapter(path, "/same-base")
    seal()
    manifest["base"]["fingerprint"] = "another-base"
    seal()
    with pytest.raises(ValueError, match="base fingerprint"):
        kt_dsv4.validate_adapter(path, "/same-base")


def test_server_accepts_one_native_pair_and_is_idempotent(adapter, monkeypatch):
    from sglang.srt.server_args import ServerArgs

    path, _, _, _ = adapter
    monkeypatch.setenv("SGLANG_OPT_FUSE_WQA_WKV", "0")
    args = ServerArgs(model_path="dummy")
    args.model_path = args.kt_weight_path = "/same-base"
    args.kt_method = "MXFP4"
    args.kt_num_gpu_experts = args.kt_gpu_prefill_token_threshold = 0
    args.disable_shared_experts_fusion = True
    args.lora_paths = [f"trained={path}"]
    args.check_lora_server_args()
    pair = args.lora_paths[0]
    assert args.kt_dsv4_lora_path == args.kt_expert_lora_path == str(path)
    assert args.kt_composite_lora_id == pair.lora_id
    assert args.disable_cuda_graph
    # Worker processes receive the serialized LoRARef representation.
    args.lora_paths = [asdict(pair)]
    args.check_lora_server_args()
    assert args.lora_paths[0].lora_id == pair.lora_id
    args.lora_paths = []
    with pytest.raises(ValueError, match="complete static adapter pair"):
        args._validate_kt_lora_serving_paths()


def test_v4_target_selection_excludes_indexer_and_router():
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

    accepts = DeepseekV4ForCausalLM.should_apply_lora
    assert accepts("model.layers.3.self_attn.wq_b")
    assert accepts("model.layers.3.mlp.shared_experts.gate_up_proj")
    for target in (
        "self_attn.indexer.wq_b",
        "self_attn.compressor.wkv_gate",
        "mlp.gate",
        "self_attn.wo_a",
    ):
        assert not accepts(f"model.layers.3.{target}")
    projection = SimpleNamespace(input_size=16, output_size=32)
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        wq_b=SimpleNamespace(base_layer=projection)
                    )
                )
            ]
        )
    )
    assert DeepseekV4ForCausalLM.get_hidden_dim(model, "wq_b", 0) == (16, 32)
