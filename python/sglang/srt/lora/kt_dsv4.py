"""Static April V4 training adapters; original base weights remain quantized."""

import hashlib
import json
import re
from pathlib import Path


MANIFEST = "kt_adapter_manifest.json"
FUSED = "fused_expert_lora.safetensors"
TARGETS = {
    "self_attn.q_a_proj": "self_attn.wq_a",
    "self_attn.q_b_proj": "self_attn.wq_b",
    "self_attn.kv_proj": "self_attn.wkv",
    "self_attn.o_b_proj": "self_attn.wo_b",
    **{
        f"mlp.shared_experts.{p}_proj": f"mlp.shared_experts.{p}_proj"
        for p in ("gate", "up", "down")
    },
    "self_attn.compressor.kv_proj": None,
    "self_attn.compressor.gate_proj": None,
}
_KEY = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(\d+)\.(.+)"
    r"\.lora_([AB])(?:\.default)?\.weight$"
)


def _read(path):
    return json.loads(Path(path).read_text())


def is_native_adapter(path):
    manifest = Path(path) / MANIFEST
    return manifest.is_file() and _read(manifest).get("expert_weight_format") == "mxfp4"


def _config(path):
    config = _read(Path(path) / "adapter_config.json")
    if (
        config.get("peft_type") != "LORA"
        or config.get("r") != 8
        or config.get("lora_alpha") != 16
        or config.get("lora_dropout", 0) != 0
        or config.get("bias", "none") != "none"
        or any(
            config.get(k)
            for k in (
                "use_dora",
                "use_rslora",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "target_parameters",
                "layer_replication",
                "lora_bias",
                "fan_in_fan_out",
                "use_qalora",
                "layers_to_transform",
                "exclude_modules",
            )
        )
    ):
        raise ValueError("April KT serving requires standard r8/alpha16/dropout0 LoRA")
    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not set(targets) <= TARGETS.keys()
    ):
        raise ValueError("Unsupported April KT LoRA target_modules")
    return config


def runtime_config(path):
    config = _config(path)
    config["target_modules"] = sorted(
        {TARGETS[target] for target in config["target_modules"] if TARGETS[target]}
    )
    if not config["target_modules"]:
        raise ValueError("April KT serving requires a supported non-expert LoRA target")
    return config


def _shapes(config, layer):
    h, q = config["hidden_size"], config["q_lora_rank"]
    inter = config["moe_intermediate_size"] * config["n_shared_experts"]
    shapes = {
        "self_attn.q_a_proj": (q, h),
        "self_attn.q_b_proj": (config["num_attention_heads"] * config["head_dim"], q),
        "self_attn.kv_proj": (config["head_dim"], h),
        "self_attn.o_b_proj": (h, config["o_groups"] * config["o_lora_rank"]),
        "mlp.shared_experts.gate_proj": (inter, h),
        "mlp.shared_experts.up_proj": (inter, h),
        "mlp.shared_experts.down_proj": (h, inter),
    }
    ratio = config["compress_ratios"][layer]
    if ratio:
        dim = config["head_dim"] * (2 if ratio == 4 else 1)
        shapes.update(
            {f"self_attn.compressor.{p}_proj": (dim, h) for p in ("kv", "gate")}
        )
    return shapes


def load_nonexpert(path, model_config):
    """Validate original names before filtering and SGLang's gate/up packing."""
    import torch
    from safetensors import safe_open

    config = _config(path)
    expected = {}
    for layer in range(model_config["num_hidden_layers"]):
        for target, (n, k) in _shapes(model_config, layer).items():
            if target in config["target_modules"]:
                expected[layer, target, "A"] = (config["r"], k)
                expected[layer, target, "B"] = (n, config["r"])
    observed, result = set(), {}
    with safe_open(
        str(Path(path) / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        for key in handle.keys():
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Unsupported April LoRA tensor: {key}")
            layer, target, kind = match.groups()
            slot = (int(layer), target, kind)
            tensor = handle.get_tensor(key)
            if (
                slot not in expected
                or slot in observed
                or tuple(tensor.shape) != expected[slot]
                or tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid April LoRA tensor: {key}")
            observed.add(slot)
            mapped = TARGETS[target]
            if mapped is not None:
                result[f"model.layers.{layer}.{mapped}.lora_{kind}.weight"] = tensor
    if observed != set(expected) or not result:
        raise ValueError("Incomplete April non-expert LoRA A/B inventory")
    return result


def validate_adapter(path, model_path):
    """Validate the existing same-base manifest without the training cache."""
    from kt_kernel.sft.deepseek_v4 import inspect_native_checkpoint
    from kt_kernel.sft.export_dsv4_sglang_adapter import _validate_fused_experts

    path = Path(path)
    manifest, config = _read(path / MANIFEST), _config(path)
    if (
        manifest.get("version") != 1
        or manifest.get("status") != "ready"
        or manifest.get("expert_weight_format") != "mxfp4"
        or manifest.get("lora") != {"rank": 8, "alpha": 16.0}
    ):
        raise ValueError("Expected a ready April MXFP4 training adapter manifest")
    required = {"adapter_config.json", "adapter_model.safetensors", FUSED}
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != required:
        raise ValueError("Incomplete April training adapter files")
    for name in sorted(required):
        file = path / name
        record = artifacts[name]
        if (
            not file.is_file()
            or file.is_symlink()
            or file.stat().st_size != record.get("size")
        ):
            raise ValueError(f"Invalid April adapter file: {name}")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record.get("sha256"):
            raise ValueError(f"April adapter checksum mismatch: {name}")
    source = inspect_native_checkpoint(model_path)
    if manifest.get("base", {}).get("fingerprint") != source["fingerprint"]:
        raise ValueError(
            "April adapter base fingerprint mismatch; use the original training base files"
        )
    model = source["config"]
    if (
        model["num_hidden_layers"] != 43
        or model["n_routed_experts"] != 256
        or model.get("swiglu_limit") != 10
    ):
        raise ValueError("Native KT LoRA serving supports April DeepSeek-V4-Flash only")
    _validate_fused_experts(path / FUSED, model, config["r"])
    load_nonexpert(path, model)
    runtime_config(path)


def load_expert_layer(path, layer, num_experts, hidden, intermediate, dtype):
    import torch
    from safetensors import safe_open

    config = _config(path)
    result = {}
    with safe_open(str(Path(path) / FUSED), framework="pt", device="cpu") as handle:
        for projection, (n, k) in {
            "gate": (intermediate, hidden),
            "up": (intermediate, hidden),
            "down": (hidden, intermediate),
        }.items():
            for kind, shape in {
                "a": (num_experts, config["r"], k),
                "b": (num_experts, n, config["r"]),
            }.items():
                name = f"{projection}_lora_{kind}"
                tensor = handle.get_tensor(f"layers.{layer}.experts.{name}")
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Invalid April expert LoRA: layer {layer}, {name}"
                    )
                result[name] = tensor.to(device="cpu", dtype=dtype).contiguous()
    return dict(result, rank=config["r"], alpha=config["lora_alpha"])
