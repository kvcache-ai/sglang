#!/usr/bin/env python3
"""Session-C text correctness oracle for GLM-5-Next-0808.

The harness deliberately has no dependency on SGLang internals.  It runs two
short, token-id based requests against the patched Transformers model and an
SGLang HTTP server, then compares generated token ids and top-log-probability
values.  The Transformers side also retains the complete vocabulary logits so
that regressions can be investigated without reloading the 300+ GiB model.

Typical usage on qj5090::

    PYTHONPATH=/path/to/patched-transformers/src \
      /path/to/reference-venv/bin/python scripts/glm5_next_session_c_oracle.py hf \
      --model-path /mnt/.../GLM-5-Next-0808/hf_fp8 \
      --output-dir /mnt/.../session_c/oracle

    python scripts/glm5_next_session_c_oracle.py server \
      --base-url http://127.0.0.1:30000 \
      --output-dir /mnt/.../session_c/eager

    python scripts/glm5_next_session_c_oracle.py compare \
      --hf-result /mnt/.../session_c/oracle/hf_oracle.json \
      --server-result /mnt/.../session_c/eager/sglang_result.json \
      --output-dir /mnt/.../session_c/eager
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
FIXTURE_SEED = 20260811
PROMPT_LENGTHS = (16, 128)
DEFAULT_NEW_TOKENS = 4
DEFAULT_TOP_K = 256
COMPARISON_TOP_K = 64
HF_CACHE_CONTRACT = {
    "version": 1,
    "implementation": "transformers.cache_utils.DynamicCache(config=text_config)",
    "use_cache": True,
    "explicit_cache_position": True,
    "explicit_position_ids": True,
    "strict_exact_growth": True,
}
EXPECTED_HF_LAYER_COUNT = 45
EXPECTED_HF_KDA_LAYER_IDS = tuple(
    layer_id for layer_id in range(EXPECTED_HF_LAYER_COUNT) if layer_id % 4 != 3
)
EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE = (3 * 64 * 128, 1, 4)
HF_KDA_CONV_FP32_CONTRACT = {
    "version": 1,
    "expected_layer_count": len(EXPECTED_HF_KDA_LAYER_IDS),
    "expected_layer_ids": list(EXPECTED_HF_KDA_LAYER_IDS),
    "expected_weight_shape": list(EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE),
    "runtime_dtype": "torch.float32",
    "activation": "silu",
    "activation_mode": "fused_causal_conv1d",
    "authoritative_runtime": (
        "official SGLang Glm5NextLinearAttention.qkv_conv1d params_dtype=torch.float32"
    ),
    "correction_reason": (
        "the pinned Transformers prequantized renamed-key loader bypasses "
        "_keep_in_fp32_modules_strict and leaves self_attn.conv1d.weight BF16"
    ),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixtures(vocab_size: int) -> list[dict[str, Any]]:
    """Return deterministic, tokenizer-independent text fixtures."""
    if vocab_size < 4096:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    fixtures: list[dict[str, Any]] = []
    for prompt_length in PROMPT_LENGTHS:
        rng = random.Random(FIXTURE_SEED + prompt_length)
        # Keep clear of low control ids and the high special-token band.
        token_ids = [
            rng.randrange(1024, vocab_size - 1024) for _ in range(prompt_length)
        ]
        encoded = b"".join(
            int(token_id).to_bytes(4, "little") for token_id in token_ids
        )
        fixtures.append(
            {
                "name": f"tokens_{prompt_length}",
                "prompt_length": prompt_length,
                "input_ids": token_ids,
                "input_sha256": _sha256_bytes(encoded),
            }
        )
    return fixtures


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _git_head(source_root: str | None) -> str | None:
    if not source_root:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", source_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _model_manifest(model_path: Path) -> dict[str, Any]:
    interesting = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    return {
        "path": str(model_path.resolve()),
        "files": {
            name: _sha256_file(model_path / name)
            for name in interesting
            if (model_path / name).exists()
        },
    }


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _top_logprobs(logits: Any, top_k: int) -> list[dict[str, float | int]]:
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logprobs, k=min(top_k, logprobs.shape[-1]))
    return [
        {"token_id": int(token_id), "logprob": float(logprob)}
        for token_id, logprob in zip(indices.tolist(), values.tolist())
    ]


def _repair_hf_kda_conv_weights(model: Any) -> dict[str, Any]:
    """Repair and attest the pinned HF loader's GLM KDA-conv dtype bug.

    The checkpoint contains separate q/k/v BF16 causal-convolution weights.  The
    Transformers conversion mapping merges them into ``self_attn.conv1d``, but
    its prequantized renamed-key branch skips the model's strict-FP32 module
    rule.  Official SGLang constructs the corresponding merged runtime weight
    in FP32 and applies SiLU inside the causal-convolution operation.  Validate
    the complete fixed-model structure before mutating any parameter, then cast
    every affected weight and revalidate it before CPU-offload hooks are added.
    """
    import torch

    try:
        text_config = model.config.text_config
        layers = model.model.language_model.layers
    except AttributeError as error:
        raise RuntimeError(
            "invalid patched-Transformers oracle: missing GLM text model structure"
        ) from error

    if len(layers) != EXPECTED_HF_LAYER_COUNT:
        raise RuntimeError(
            "invalid patched-Transformers oracle: expected "
            f"{EXPECTED_HF_LAYER_COUNT} text layers, found {len(layers)}"
        )
    layer_types = getattr(text_config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or len(layer_types) != len(layers):
        raise RuntimeError(
            "invalid patched-Transformers oracle: layer_types does not describe "
            "all runtime text layers"
        )
    kda_layer_ids = tuple(
        layer_id
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    )
    if kda_layer_ids != EXPECTED_HF_KDA_LAYER_IDS:
        raise RuntimeError(
            "invalid patched-Transformers oracle: KDA layer IDs differ from the "
            f"frozen GLM-5-Next-0808 contract: {list(kda_layer_ids)}"
        )

    expected_config = {
        "linear_num_heads": 64,
        "linear_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    for name, expected in expected_config.items():
        actual = getattr(text_config, name, None)
        if actual != expected:
            raise RuntimeError(
                "invalid patched-Transformers oracle: "
                f"text_config.{name}={actual!r}, expected {expected}"
            )

    validated: list[tuple[int, Any]] = []
    source_dtype_by_layer: dict[str, str] = {}
    allowed_source_dtypes = {torch.bfloat16, torch.float32}
    expected_channels = EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE[0]
    for layer_id in kda_layer_ids:
        self_attn = getattr(layers[layer_id], "self_attn", None)
        conv1d = getattr(self_attn, "conv1d", None)
        module_name = f"model.language_model.layers.{layer_id}.self_attn.conv1d"
        if not isinstance(conv1d, torch.nn.Conv1d):
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name} is not Conv1d"
            )
        structural_contract = {
            "in_channels": expected_channels,
            "out_channels": expected_channels,
            "groups": expected_channels,
            "kernel_size": (4,),
            "stride": (1,),
            "padding": (3,),
            "dilation": (1,),
        }
        for name, expected in structural_contract.items():
            actual = getattr(conv1d, name, None)
            if actual != expected:
                raise RuntimeError(
                    f"invalid patched-Transformers oracle: {module_name}.{name}="
                    f"{actual!r}, expected {expected!r}"
                )
        if conv1d.bias is not None:
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name} has a bias"
            )
        if tuple(conv1d.weight.shape) != EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE:
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name}.weight shape "
                f"{tuple(conv1d.weight.shape)} != "
                f"{EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE}"
            )
        if conv1d.weight.dtype not in allowed_source_dtypes:
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name}.weight dtype "
                f"{conv1d.weight.dtype} is neither BF16 nor FP32"
            )
        if getattr(self_attn, "layer_idx", None) != layer_id:
            raise RuntimeError(
                f"invalid patched-Transformers oracle: layer {layer_id} reports "
                f"self_attn.layer_idx={getattr(self_attn, 'layer_idx', None)!r}"
            )
        if getattr(self_attn, "activation", None) != "silu":
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name} does not use "
                "fused SiLU"
            )
        source_dtype_by_layer[str(layer_id)] = str(conv1d.weight.dtype)
        validated.append((layer_id, conv1d))

    converted_layer_ids = [
        layer_id
        for layer_id, conv1d in validated
        if conv1d.weight.dtype == torch.bfloat16
    ]
    already_fp32_layer_ids = [
        layer_id
        for layer_id, conv1d in validated
        if conv1d.weight.dtype == torch.float32
    ]
    with torch.no_grad():
        for _, conv1d in validated:
            conv1d.weight.data = conv1d.weight.data.to(dtype=torch.float32)

    runtime_dtype_by_layer: dict[str, str] = {}
    for layer_id, conv1d in validated:
        if (
            tuple(conv1d.weight.shape) != EXPECTED_HF_KDA_CONV_WEIGHT_SHAPE
            or conv1d.weight.dtype != torch.float32
        ):
            raise RuntimeError(
                "invalid patched-Transformers oracle: KDA-conv FP32 repair did "
                f"not hold for layer {layer_id}"
            )
        runtime_dtype_by_layer[str(layer_id)] = str(conv1d.weight.dtype)

    return {
        "contract": HF_KDA_CONV_FP32_CONTRACT,
        "source_dtype_by_layer": source_dtype_by_layer,
        "converted_layer_ids": converted_layer_ids,
        "already_fp32_layer_ids": already_fp32_layer_ids,
        "runtime_dtype_by_layer": runtime_dtype_by_layer,
        "verified_layer_ids": list(kda_layer_ids),
    }


def _load_hf_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    # DeepGEMM is intentionally disabled for this one-device CPU-offload oracle.
    # The patched FP8 implementation falls back to its Triton path.
    os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    import torch
    from accelerate import cpu_offload
    from transformers import AutoConfig, Glm5NextForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("the FP8 Transformers oracle requires a CUDA device")
    execution_device = torch.device(args.device)
    torch.cuda.set_device(execution_device)
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    quantization_config = config.quantization_config
    skip_modules = quantization_config.get("modules_to_not_convert", [])
    # The official checkpoint uses the pre-modularized names
    # ``self_attn.f_{a,b}_proj`` while the patched model nests both projections
    # under ``self_attn.forget_gate``.  The generic conversion map correctly
    # renames checkpoint keys, but the FP8 skip-list normalizer in the supplied
    # patch currently misses this nesting.  Repair the two skip names before
    # module replacement; otherwise Transformers turns BF16 forget-gate linears
    # into FP8Linear and silently initializes nonexistent scale tensors.
    repaired_skip_modules = []
    repaired_count = 0
    for name in skip_modules:
        if ".self_attn.f_a_proj" in name:
            name = name.replace(
                ".self_attn.f_a_proj", ".self_attn.forget_gate.f_a_proj"
            )
            repaired_count += 1
        elif ".self_attn.f_b_proj" in name:
            name = name.replace(
                ".self_attn.f_b_proj", ".self_attn.forget_gate.f_b_proj"
            )
            repaired_count += 1
        repaired_skip_modules.append(name)
    if repaired_count != 68:
        raise RuntimeError(
            "expected to repair 68 GLM linear-attention forget-gate skip-list "
            f"entries, repaired {repaired_count}"
        )
    quantization_config["modules_to_not_convert"] = repaired_skip_modules

    model = Glm5NextForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        attn_implementation="eager",
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    kda_conv_correction = _repair_hf_kda_conv_weights(model)
    cpu_offload(model, execution_device=execution_device, offload_buffers=True)
    for module_name in (
        "model.language_model.layers.0.self_attn.forget_gate.f_a_proj",
        "model.language_model.layers.0.self_attn.forget_gate.f_b_proj",
    ):
        module = model.get_submodule(module_name)
        if module.__class__.__name__ != "Linear" or hasattr(module, "weight_scale_inv"):
            raise RuntimeError(
                f"invalid patched-Transformers oracle: {module_name} was "
                f"quantized as {module.__class__.__name__}"
            )
    return model, {"glm5_next_kda_conv_fp32": kda_conv_correction}


def _cache_counter(value: Any, *, label: str) -> int:
    """Convert a scalar cache counter to a fail-closed non-negative integer."""
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise RuntimeError(f"{label} is not scalar: shape={tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is boolean, not a token counter")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{label} is not an integer: {value!r}") from error
    if converted < 0 or converted != value:
        raise RuntimeError(f"{label} is not a non-negative integer: {value!r}")
    return converted


def _cache_progress(cache: Any, *, label: str) -> dict[str, int]:
    """Read every public/legacy token counter exposed by a Transformers cache."""
    if cache is None:
        raise RuntimeError(f"{label}: use_cache=True returned past_key_values=None")
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise RuntimeError(f"{label}: cache has no callable get_seq_length()")

    progress = {
        "sequence_length": _cache_counter(
            get_seq_length(), label=f"{label}.get_seq_length()"
        )
    }
    # Transformers versions differ on whether this counter is public, private,
    # or entirely represented by get_seq_length().  If either form exists, it
    # is part of the contract and must agree with the authoritative length.
    for attribute in ("seen_tokens", "_seen_tokens"):
        try:
            value = getattr(cache, attribute)
        except AttributeError:
            continue
        if callable(value):
            value = value()
        progress[attribute] = _cache_counter(value, label=f"{label}.{attribute}")
    return progress


def _assert_cache_progress(
    progress: dict[str, int], *, expected: int, label: str
) -> None:
    for counter_name, value in progress.items():
        if value != expected:
            raise RuntimeError(
                f"{label}: cache {counter_name}={value}, expected exactly {expected}"
            )


def _new_hf_dynamic_cache(model: Any) -> Any:
    """Construct the cache class used by the official patched GLM5 model."""
    from transformers.cache_utils import DynamicCache

    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    text_config = (
        get_text_config()
        if callable(get_text_config)
        else getattr(config, "text_config", config)
    )
    cache = DynamicCache(config=text_config)
    initial = _cache_progress(cache, label="new HF DynamicCache")
    _assert_cache_progress(initial, expected=0, label="new HF DynamicCache")
    return cache


def _run_hf_case(
    model: Any,
    fixture: dict[str, Any],
    *,
    device: str,
    max_new_tokens: int,
    top_k: int,
) -> tuple[dict[str, Any], Any]:
    import torch

    input_ids = torch.tensor([fixture["input_ids"]], dtype=torch.long, device=device)
    past_key_values = _new_hf_dynamic_cache(model)
    initial_progress = _cache_progress(
        past_key_values, label=f"{fixture['name']} initial cache"
    )
    _assert_cache_progress(
        initial_progress, expected=0, label=f"{fixture['name']} initial cache"
    )
    generated_ids: list[int] = []
    full_logits: list[Any] = []
    steps: list[dict[str, Any]] = []
    cache_steps: list[dict[str, Any]] = []

    with torch.inference_mode():
        for step_index in range(max_new_tokens):
            expected_before = (
                0 if step_index == 0 else fixture["prompt_length"] + step_index - 1
            )
            expected_after = expected_before + int(input_ids.shape[1])
            before_progress = _cache_progress(
                past_key_values,
                label=f"{fixture['name']} step {step_index} cache before",
            )
            _assert_cache_progress(
                before_progress,
                expected=expected_before,
                label=f"{fixture['name']} step {step_index} cache before",
            )

            # These are the same absolute positions used by GenerationMixin,
            # made explicit so the hand-written rollout cannot silently restart
            # at position zero on a decode step.
            cache_position = torch.arange(
                expected_before,
                expected_after,
                dtype=torch.long,
                device=device,
            )
            position_ids = cache_position.unsqueeze(0)
            attention_mask = torch.ones(
                (1, expected_after), dtype=torch.long, device=device
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            returned_cache = getattr(outputs, "past_key_values", None)
            after_progress = _cache_progress(
                returned_cache,
                label=f"{fixture['name']} step {step_index} cache after",
            )
            _assert_cache_progress(
                after_progress,
                expected=expected_after,
                label=f"{fixture['name']} step {step_index} cache after",
            )
            if after_progress["sequence_length"] <= before_progress["sequence_length"]:
                raise RuntimeError(
                    f"{fixture['name']} step {step_index}: cache did not strictly grow: "
                    f"{before_progress['sequence_length']} -> "
                    f"{after_progress['sequence_length']}"
                )

            step_logits = outputs.logits[0, -1].float().cpu()
            if not bool(torch.isfinite(step_logits).all()):
                raise RuntimeError(
                    f"non-finite HF logits in {fixture['name']} step {step_index}"
                )
            next_token = int(torch.argmax(step_logits))
            generated_ids.append(next_token)
            full_logits.append(step_logits)
            steps.append(
                {
                    "step": step_index,
                    "token_id": next_token,
                    "top_logprobs": _top_logprobs(step_logits, top_k),
                }
            )
            cache_steps.append(
                {
                    "step": step_index,
                    "input_length": int(input_ids.shape[1]),
                    "attention_mask_length": int(attention_mask.shape[1]),
                    "cache_position": cache_position.tolist(),
                    "position_ids": position_ids[0].tolist(),
                    "before": before_progress,
                    "after": after_progress,
                }
            )
            past_key_values = returned_cache
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return (
        {
            "name": fixture["name"],
            "prompt_length": fixture["prompt_length"],
            "input_sha256": fixture["input_sha256"],
            "generated_ids": generated_ids,
            "steps": steps,
            "cache_audit": {
                "cache_class": (
                    f"{past_key_values.__class__.__module__}."
                    f"{past_key_values.__class__.__qualname__}"
                ),
                "use_cache": True,
                "initial": initial_progress,
                "steps": cache_steps,
            },
        },
        torch.stack(full_logits),
    )


def run_hf(args: argparse.Namespace) -> int:
    import torch

    started = time.time()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model, oracle_corrections = _load_hf_model(args)
    text_config = model.config.text_config
    fixtures = build_fixtures(int(text_config.vocab_size))
    repetitions: list[dict[str, Any]] = []
    tensors_by_repeat: list[dict[str, Any]] = []

    for repeat_index in range(args.repeats):
        repeat_cases: list[dict[str, Any]] = []
        repeat_tensors: dict[str, Any] = {}
        for fixture in fixtures:
            case, logits = _run_hf_case(
                model,
                fixture,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
            )
            repeat_cases.append(case)
            repeat_tensors[fixture["name"]] = logits
            torch.cuda.empty_cache()
        repetitions.append({"repeat": repeat_index, "cases": repeat_cases})
        tensors_by_repeat.append(repeat_tensors)

    envelope: dict[str, Any] = {
        "max_abs": 0.0,
        "max_relative": 0.0,
        "tokens_exact": True,
    }
    if args.repeats > 1:
        reference = tensors_by_repeat[0]
        reference_cases = {case["name"]: case for case in repetitions[0]["cases"]}
        for repeat_index in range(1, args.repeats):
            candidate_cases = {
                case["name"]: case for case in repetitions[repeat_index]["cases"]
            }
            for name, expected in reference.items():
                candidate = tensors_by_repeat[repeat_index][name]
                delta = (expected - candidate).abs()
                relative = delta / expected.abs().clamp_min(1e-6)
                envelope["max_abs"] = max(envelope["max_abs"], float(delta.max()))
                envelope["max_relative"] = max(
                    envelope["max_relative"], float(relative.max())
                )
                envelope["tokens_exact"] = bool(envelope["tokens_exact"]) and (
                    reference_cases[name]["generated_ids"]
                    == candidate_cases[name]["generated_ids"]
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / "hf_full_logits.pt"
    # Persist all repetitions.  A full run is still only a few MiB.
    torch.save(tensors_by_repeat, logits_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "patched_transformers_oracle",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "model": _model_manifest(model_path),
        "source_head": _git_head(args.source_root),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": args.device,
        },
        "fixture_seed": FIXTURE_SEED,
        "cache_contract": HF_CACHE_CONTRACT,
        "oracle_corrections": oracle_corrections,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "repeat_envelope": envelope,
        "cases": repetitions[0]["cases"],
        "repetitions": args.repeats,
        "full_logits_file": logits_path.name,
        "pass": bool(envelope["tokens_exact"]),
    }
    _write_json(output_dir / "hf_oracle.json", payload)
    print(json.dumps({"pass": payload["pass"], "repeat_envelope": envelope}, indent=2))
    return 0 if payload["pass"] else 1


def _parse_sglang_top_logprobs(raw: Any) -> list[dict[str, float | int]]:
    parsed: list[dict[str, float | int]] = []
    for entry in raw or []:
        if isinstance(entry, dict):
            token_id = entry.get("token_id")
            logprob = entry.get("logprob")
        else:
            logprob, token_id = entry[0], entry[1]
        parsed.append({"token_id": int(token_id), "logprob": float(logprob)})
    return parsed


def run_server(args: argparse.Namespace) -> int:
    import requests

    started = time.time()
    base_url = args.base_url.rstrip("/")
    if not args.no_flush_cache:
        response = requests.post(
            f"{base_url}/flush_cache", timeout=args.request_timeout
        )
        response.raise_for_status()

    # Match the model-config vocabulary exactly.  Fixture ids deliberately stay
    # below the tokenizer's high special-token band, but the RNG upper bound must
    # still equal the Transformers oracle or the two sides receive different ids.
    fixtures = build_fixtures(args.vocab_size)
    cases: list[dict[str, Any]] = []
    for fixture in fixtures:
        request_body = {
            "input_ids": fixture["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "top_logprobs_num": args.top_k,
            "return_text_in_logprobs": False,
        }
        response = requests.post(
            f"{base_url}/generate", json=request_body, timeout=args.request_timeout
        )
        response.raise_for_status()
        body = response.json()
        if isinstance(body, list):
            if len(body) != 1:
                raise RuntimeError(f"unexpected batched SGLang response: {len(body)}")
            body = body[0]
        meta = body.get("meta_info", {})
        generated_ids = [int(token_id) for token_id in body.get("output_ids", [])]
        raw_steps = meta.get("output_top_logprobs", [])
        if (
            len(generated_ids) != args.max_new_tokens
            or len(raw_steps) != args.max_new_tokens
        ):
            raise RuntimeError(
                f"incomplete SGLang result for {fixture['name']}: "
                f"ids={len(generated_ids)} top_logprobs={len(raw_steps)}"
            )
        steps: list[dict[str, Any]] = []
        for index, (token_id, raw_top) in enumerate(zip(generated_ids, raw_steps)):
            top_logprobs = _parse_sglang_top_logprobs(raw_top)
            if len(top_logprobs) < min(args.top_k, args.vocab_size):
                raise RuntimeError(
                    f"short top-logprob list in {fixture['name']} step {index}: "
                    f"{len(top_logprobs)}"
                )
            if not _finite([float(item["logprob"]) for item in top_logprobs]):
                raise RuntimeError(
                    f"non-finite SGLang logprobs in {fixture['name']} step {index}"
                )
            steps.append(
                {"step": index, "token_id": token_id, "top_logprobs": top_logprobs}
            )
        cases.append(
            {
                "name": fixture["name"],
                "prompt_length": fixture["prompt_length"],
                "input_sha256": fixture["input_sha256"],
                "generated_ids": generated_ids,
                "steps": steps,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sglang_http_result",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "base_url": base_url,
        "source_head": _git_head(args.source_root),
        "fixture_seed": FIXTURE_SEED,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "cases": cases,
        "pass": True,
    }
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "sglang_result.json", payload)
    print(
        json.dumps({"pass": True, "cases": [case["name"] for case in cases]}, indent=2)
    )
    return 0


def _validate_serialized_cache_audit(
    case: dict[str, Any], *, max_new_tokens: int
) -> list[str]:
    """Validate the persisted evidence before accepting an HF oracle file."""
    failures: list[str] = []
    audit = case.get("cache_audit")
    if not isinstance(audit, dict):
        return ["missing cache_audit"]
    if audit.get("use_cache") is not True:
        failures.append("cache_audit does not attest use_cache=True")
    if not isinstance(audit.get("cache_class"), str) or not audit["cache_class"]:
        failures.append("cache_audit has no cache class")

    initial = audit.get("initial")
    if not isinstance(initial, dict) or initial.get("sequence_length") != 0:
        failures.append("cache did not start at sequence length zero")
    elif any(value != 0 for value in initial.values()):
        failures.append("an initial cache token counter was nonzero")

    cache_steps = audit.get("steps")
    if not isinstance(cache_steps, list) or len(cache_steps) != max_new_tokens:
        failures.append(
            "cache audit step count differs from max_new_tokens: "
            f"{len(cache_steps) if isinstance(cache_steps, list) else 'invalid'} "
            f"!= {max_new_tokens}"
        )
        return failures

    prompt_length = case.get("prompt_length")
    if not isinstance(prompt_length, int) or prompt_length <= 0:
        failures.append("invalid prompt_length in cache audit")
        return failures

    for step_index, step in enumerate(cache_steps):
        if not isinstance(step, dict):
            failures.append(f"cache step {step_index} is not an object")
            continue
        expected_before = 0 if step_index == 0 else prompt_length + step_index - 1
        expected_input_length = prompt_length if step_index == 0 else 1
        expected_after = expected_before + expected_input_length
        expected_positions = list(range(expected_before, expected_after))
        expected_scalars = {
            "step": step_index,
            "input_length": expected_input_length,
            "attention_mask_length": expected_after,
        }
        for key, expected in expected_scalars.items():
            if step.get(key) != expected:
                failures.append(
                    f"cache step {step_index} {key}={step.get(key)!r}, "
                    f"expected {expected}"
                )
        for key in ("cache_position", "position_ids"):
            if step.get(key) != expected_positions:
                failures.append(
                    f"cache step {step_index} {key}={step.get(key)!r}, "
                    f"expected {expected_positions}"
                )
        for phase, expected in (("before", expected_before), ("after", expected_after)):
            progress = step.get(phase)
            if not isinstance(progress, dict) or "sequence_length" not in progress:
                failures.append(f"cache step {step_index} has invalid {phase} counters")
                continue
            for counter_name, value in progress.items():
                if value != expected:
                    failures.append(
                        f"cache step {step_index} {phase}.{counter_name}={value!r}, "
                        f"expected {expected}"
                    )
            if phase == "after" and progress["sequence_length"] <= expected_before:
                failures.append(f"cache step {step_index} did not strictly grow")
    return failures


def _validate_serialized_hf_kda_conv_correction(
    oracle: dict[str, Any],
) -> list[str]:
    """Reject oracle evidence that lacks the fixed FP32 fused-conv contract."""
    failures: list[str] = []
    corrections = oracle.get("oracle_corrections")
    if not isinstance(corrections, dict):
        return ["missing oracle_corrections"]
    correction = corrections.get("glm5_next_kda_conv_fp32")
    if not isinstance(correction, dict):
        return ["missing glm5_next_kda_conv_fp32 correction"]
    if correction.get("contract") != HF_KDA_CONV_FP32_CONTRACT:
        failures.append("KDA-conv correction contract differs from the frozen contract")

    expected_keys = {str(layer_id) for layer_id in EXPECTED_HF_KDA_LAYER_IDS}
    source_dtypes = correction.get("source_dtype_by_layer")
    if not isinstance(source_dtypes, dict) or set(source_dtypes) != expected_keys:
        failures.append("KDA-conv correction source dtype layer set is invalid")
        source_dtypes = {}
    elif not all(
        dtype in {"torch.bfloat16", "torch.float32"} for dtype in source_dtypes.values()
    ):
        failures.append("KDA-conv correction contains an unexpected source dtype")

    converted = correction.get("converted_layer_ids")
    already_fp32 = correction.get("already_fp32_layer_ids")
    if not isinstance(converted, list) or not all(
        isinstance(layer_id, int) for layer_id in converted
    ):
        failures.append("KDA-conv converted layer IDs are invalid")
        converted = []
    if not isinstance(already_fp32, list) or not all(
        isinstance(layer_id, int) for layer_id in already_fp32
    ):
        failures.append("KDA-conv already-FP32 layer IDs are invalid")
        already_fp32 = []
    expected_converted = [
        layer_id
        for layer_id in EXPECTED_HF_KDA_LAYER_IDS
        if source_dtypes.get(str(layer_id)) == "torch.bfloat16"
    ]
    expected_already_fp32 = [
        layer_id
        for layer_id in EXPECTED_HF_KDA_LAYER_IDS
        if source_dtypes.get(str(layer_id)) == "torch.float32"
    ]
    if converted != expected_converted:
        failures.append("KDA-conv converted layers disagree with source dtypes")
    if already_fp32 != expected_already_fp32:
        failures.append("KDA-conv already-FP32 layers disagree with source dtypes")

    runtime_dtypes = correction.get("runtime_dtype_by_layer")
    if (
        not isinstance(runtime_dtypes, dict)
        or set(runtime_dtypes) != expected_keys
        or set(runtime_dtypes.values()) != {"torch.float32"}
    ):
        failures.append("KDA-conv runtime dtype attestation is not all FP32")
    if correction.get("verified_layer_ids") != list(EXPECTED_HF_KDA_LAYER_IDS):
        failures.append("KDA-conv verified layer IDs are incomplete")
    return failures


def compare_payloads(
    oracle: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    comparison_top_k: int = COMPARISON_TOP_K,
) -> dict[str, Any]:
    """Compare independent greedy rollouts while their prefixes are identical.

    Logits at the first token-mismatch step are still comparable because both
    implementations consumed the same prefix.  Once that token differs, later
    logits are conditioned on different prefixes and must not be used as a
    numerical correctness signal.  The token mismatch itself remains a hard
    failure.
    """
    failures: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if oracle.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported oracle schema")
    if candidate.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported candidate schema")
    if oracle.get("kind") != "patched_transformers_oracle":
        failures.append("oracle payload kind is not patched_transformers_oracle")
    if candidate.get("kind") != "sglang_http_result":
        failures.append("candidate payload kind is not sglang_http_result")
    if oracle.get("pass") is not True:
        failures.append("oracle payload did not pass")
    if candidate.get("pass") is not True:
        failures.append("candidate payload did not pass")
    if oracle.get("fixture_seed") != FIXTURE_SEED:
        failures.append("oracle fixture seed differs from the frozen contract")
    if candidate.get("fixture_seed") != FIXTURE_SEED:
        failures.append("candidate fixture seed differs from the frozen contract")
    if oracle.get("max_new_tokens") != candidate.get("max_new_tokens"):
        failures.append("oracle and candidate max_new_tokens differ")
    oracle_max_new_tokens = oracle.get("max_new_tokens")
    if not isinstance(oracle_max_new_tokens, int) or oracle_max_new_tokens <= 0:
        failures.append("invalid max_new_tokens contract")
        oracle_max_new_tokens = 0
    for role, payload in (("oracle", oracle), ("candidate", candidate)):
        top_k = payload.get("top_k")
        if not isinstance(top_k, int) or top_k < comparison_top_k:
            failures.append(
                f"{role} top_k is smaller than comparison_top_k={comparison_top_k}"
            )
    if oracle.get("repetitions", 0) < 2:
        failures.append("oracle payload lacks two deterministic repetitions")
    if oracle.get("repeat_envelope", {}).get("tokens_exact") is not True:
        failures.append("oracle repeated generated tokens were not exact")
    if oracle.get("cache_contract") != HF_CACHE_CONTRACT:
        failures.append("oracle payload lacks the strict HF cache contract")
    failures.extend(_validate_serialized_hf_kda_conv_correction(oracle))
    oracle_cases = {case["name"]: case for case in oracle.get("cases", [])}
    candidate_cases = {case["name"]: case for case in candidate.get("cases", [])}
    if set(oracle_cases) != set(candidate_cases):
        failures.append(
            f"case set mismatch: oracle={sorted(oracle_cases)} "
            f"candidate={sorted(candidate_cases)}"
        )

    for name in sorted(set(oracle_cases) & set(candidate_cases)):
        expected_case = oracle_cases[name]
        actual_case = candidate_cases[name]
        failures.extend(
            f"{name}: invalid HF cache audit: {failure}"
            for failure in _validate_serialized_cache_audit(
                expected_case,
                max_new_tokens=oracle_max_new_tokens,
            )
        )
        if expected_case.get("input_sha256") != actual_case.get("input_sha256"):
            failures.append(f"{name}: input token digest mismatch")
            continue
        expected_generated = expected_case.get("generated_ids", [])
        actual_generated = actual_case.get("generated_ids", [])
        if expected_generated != actual_generated:
            failures.append(
                f"{name}: generated token mismatch: "
                f"{expected_generated} != {actual_generated}"
            )
        first_divergent_step = next(
            (
                step_index
                for step_index, (expected_token, actual_token) in enumerate(
                    zip(expected_generated, actual_generated)
                )
                if expected_token != actual_token
            ),
            None,
        )
        if first_divergent_step is None and len(expected_generated) != len(
            actual_generated
        ):
            first_divergent_step = min(len(expected_generated), len(actual_generated))
        expected_steps = expected_case.get("steps", [])
        actual_steps = actual_case.get("steps", [])
        if len(expected_steps) != len(actual_steps):
            failures.append(
                f"{name}: step count mismatch: {len(expected_steps)} != {len(actual_steps)}"
            )
        for step_index, (expected_step, actual_step) in enumerate(
            zip(expected_steps, actual_steps)
        ):
            if first_divergent_step is not None and step_index > first_divergent_step:
                diagnostics.append(
                    {
                        "case": name,
                        "step": step_index,
                        "status": "skipped_due_to_prefix_divergence",
                        "first_divergent_step": first_divergent_step,
                    }
                )
                continue
            expected_top = expected_step["top_logprobs"][:comparison_top_k]
            actual_by_id = {
                int(item["token_id"]): float(item["logprob"])
                for item in actual_step["top_logprobs"]
            }
            missing = [
                int(item["token_id"])
                for item in expected_top
                if int(item["token_id"]) not in actual_by_id
            ]
            max_abs = 0.0
            max_relative = 0.0
            numerical_failures = 0
            for item in expected_top:
                token_id = int(item["token_id"])
                if token_id not in actual_by_id:
                    continue
                expected_value = float(item["logprob"])
                actual_value = actual_by_id[token_id]
                delta = abs(expected_value - actual_value)
                relative = delta / max(abs(expected_value), 1e-6)
                max_abs = max(max_abs, delta)
                max_relative = max(max_relative, relative)
                if delta > atol + rtol * abs(expected_value):
                    numerical_failures += 1
            diagnostics.append(
                {
                    "case": name,
                    "step": step_index,
                    "status": "compared",
                    "max_abs": max_abs,
                    "max_relative": max_relative,
                    "missing_reference_top_ids": missing,
                    "numerical_failures": numerical_failures,
                }
            )
            if missing:
                failures.append(
                    f"{name} step {step_index}: {len(missing)} oracle top-"
                    f"{comparison_top_k} ids absent from candidate top list"
                )
            if numerical_failures:
                failures.append(
                    f"{name} step {step_index}: {numerical_failures} logprobs outside "
                    f"atol={atol}, rtol={rtol}"
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5_next_correctness_comparison",
        "pass": not failures,
        "atol": atol,
        "rtol": rtol,
        "comparison_top_k": comparison_top_k,
        "failures": failures,
        "diagnostics": diagnostics,
    }


def run_compare(args: argparse.Namespace) -> int:
    with Path(args.hf_result).open(encoding="utf-8") as handle:
        oracle = json.load(handle)
    with Path(args.server_result).open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    result = compare_payloads(
        oracle,
        candidate,
        atol=args.atol,
        rtol=args.rtol,
        comparison_top_k=args.comparison_top_k,
    )
    result["oracle"] = str(Path(args.hf_result).resolve())
    result["candidate"] = str(Path(args.server_result).resolve())
    result["created_unix"] = time.time()
    _write_json(Path(args.output_dir) / "comparison.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("hf", help="run the patched Transformers oracle")
    hf_parser.add_argument("--model-path", required=True)
    hf_parser.add_argument("--output-dir", required=True)
    hf_parser.add_argument("--source-root")
    hf_parser.add_argument("--device", default="cuda:0")
    hf_parser.add_argument("--repeats", type=int, default=2)
    hf_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    hf_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    hf_parser.set_defaults(func=run_hf)

    server_parser = subparsers.add_parser("server", help="query an SGLang server")
    server_parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    server_parser.add_argument("--output-dir", required=True)
    server_parser.add_argument("--source-root")
    server_parser.add_argument("--vocab-size", type=int, default=154880)
    server_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    server_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    server_parser.add_argument("--request-timeout", type=float, default=7200)
    server_parser.add_argument("--no-flush-cache", action="store_true")
    server_parser.set_defaults(func=run_server)

    compare_parser = subparsers.add_parser(
        "compare", help="compare oracle and server JSON"
    )
    compare_parser.add_argument("--hf-result", required=True)
    compare_parser.add_argument("--server-result", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--atol", type=float, default=5e-2)
    compare_parser.add_argument("--rtol", type=float, default=5e-2)
    compare_parser.add_argument(
        "--comparison-top-k", type=int, default=COMPARISON_TOP_K
    )
    compare_parser.set_defaults(func=run_compare)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if hasattr(args, "repeats") and args.repeats < 2:
        raise ValueError("the oracle requires at least two repetitions")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
