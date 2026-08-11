"""CPU-only construction contracts for GLM-5-Next DSA attention."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next_dsa.py"


class _Glm5NextTextConfig:
    model_type = "glm5_next_text"

    def __init__(self, **overrides):
        values = dict(
            hidden_size=4096,
            num_attention_heads=64,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            v_head_dim=256,
            q_lora_rank=1536,
            kv_lora_rank=512,
            rope_theta=800000.0,
            rope_scaling=None,
            max_position_embeddings=1048576,
            rms_norm_eps=1e-5,
            index_head_dim=128,
            index_topk=2048,
            index_kpool=4,
            index_kpool_compress=True,
            index_kpool_always_select_tail=True,
            index_n_heads=32,
            index_topk_freq=1,
            index_topk_pattern=None,
            index_skip_topk_offset=None,
            index_share_for_mtp_iteration=False,
            indexer_rope_interleave=True,
            architectures=["Glm5NextForConditionalGeneration"],
        )
        values.update(overrides)
        self.__dict__.update(values)


class _RecordingBaseAttention(nn.Module):
    init_calls = []

    def __init__(self, **kwargs):
        super().__init__()
        type(self).init_calls.append(kwargs)
        self.base_kwargs = kwargs
        self.base_forward_dependency = object()
        self.base_weight = nn.Parameter(torch.ones(1))
        self.use_nsa = False
        self.nsa_enable_prefill_cp = False
        self.is_nextn = kwargs["is_nextn"]
        self.rotary_emb = object()


class _RecordingIndexerKPool(nn.Module):
    init_calls = []

    def __init__(self, **kwargs):
        super().__init__()
        type(self).init_calls.append(kwargs)
        self.kwargs = kwargs
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(4, 1))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(1, 1))
        self.wk = nn.Linear(1, 1, bias=False)


def _load_module(*, cp_enabled=False):
    packages = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.layers",
        "sglang.srt.layers.attention",
        "sglang.srt.layers.attention.nsa",
        "sglang.srt.layers.quantization",
        "sglang.srt.models",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    config_module = types.ModuleType("sglang.srt.configs.glm5_next")
    config_module.Glm5NextTextConfig = _Glm5NextTextConfig
    packages[config_module.__name__] = config_module

    indexer_module = types.ModuleType(
        "sglang.srt.layers.attention.nsa.nsa_indexer_kpool"
    )
    indexer_module.IndexerKPool = _RecordingIndexerKPool
    packages[indexer_module.__name__] = indexer_module

    nsa_utils = types.ModuleType("sglang.srt.layers.attention.nsa.utils")
    nsa_utils.is_nsa_enable_prefill_cp = lambda: cp_enabled
    packages[nsa_utils.__name__] = nsa_utils

    quantization = types.ModuleType("sglang.srt.layers.quantization.base_config")
    quantization.QuantizationConfig = type("QuantizationConfig", (), {})
    packages[quantization.__name__] = quantization

    deepseek = types.ModuleType("sglang.srt.models.deepseek_v2")
    deepseek.DeepseekV2AttentionMLA = _RecordingBaseAttention
    packages[deepseek.__name__] = deepseek

    utils = types.ModuleType("sglang.srt.utils")
    utils.add_prefix = lambda name, prefix: f"{prefix}.{name}" if prefix else name
    packages[utils.__name__] = utils

    module_name = f"_glm5_next_dsa_test_{int(cp_enabled)}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


def _build(attention_cls, config=None, **overrides):
    config = config or _Glm5NextTextConfig()
    kwargs = dict(
        config=config,
        hidden_size=4096,
        num_heads=64,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        rope_theta=800000.0,
        max_position_embeddings=1048576,
        layer_id=3,
        prefix="model.layers.3.self_attn",
        skip_rope=True,
        is_nextn=False,
    )
    kwargs.update(overrides)
    return attention_cls(**kwargs)


class TestGlm5NextDSAAttention(unittest.TestCase):
    def setUp(self):
        _RecordingBaseAttention.init_calls.clear()
        _RecordingIndexerKPool.init_calls.clear()
        self.module = _load_module()

    def test_rejects_non_glm_config_before_base_construction(self):
        non_glm = SimpleNamespace(model_type="deepseek_v3")
        self.assertFalse(self.module.is_glm5_next_dsa_config(non_glm))
        with self.assertRaisesRegex(TypeError, "only accepts Glm5NextTextConfig"):
            _build(self.module.Glm5NextDSAAttention, config=non_glm)
        self.assertEqual(_RecordingBaseAttention.init_calls, [])

    def test_constructs_zero_rope_kpool_from_real_text_config(self):
        config = _Glm5NextTextConfig()
        attention = _build(self.module.Glm5NextDSAAttention, config=config)

        self.assertTrue(self.module.is_glm5_next_dsa_config(config))
        self.assertTrue(attention.use_nsa)
        self.assertTrue(attention.skip_rope)
        self.assertIsNone(attention.rotary_emb)
        self.assertFalse(attention.nsa_enable_prefill_cp)
        self.assertFalse(attention.is_nextn)
        self.assertFalse(attention.skip_topk)
        self.assertFalse(attention.next_skip_topk)
        self.assertIsNotNone(attention.base_forward_dependency)

        base_kwargs = attention.base_kwargs
        self.assertIsNot(base_kwargs["config"], config)
        self.assertIsNone(base_kwargs["config"].index_topk)
        self.assertEqual(config.index_topk, 2048)
        self.assertTrue(base_kwargs["skip_rope"])
        self.assertFalse(base_kwargs["is_nextn"])

        indexer_kwargs = attention.indexer.kwargs
        self.assertIs(indexer_kwargs["config"], config)
        self.assertTrue(indexer_kwargs["skip_rope"])
        self.assertEqual(indexer_kwargs["hidden_size"], 4096)
        self.assertEqual(indexer_kwargs["index_n_heads"], 32)
        self.assertEqual(indexer_kwargs["index_head_dim"], 128)
        self.assertEqual(indexer_kwargs["rope_head_dim"], 0)
        self.assertEqual(indexer_kwargs["index_topk"], 2048)
        self.assertEqual(indexer_kwargs["q_lora_rank"], 1536)
        self.assertEqual(indexer_kwargs["scale_fmt"], "ue8m0")
        self.assertEqual(indexer_kwargs["block_size"], 128)
        self.assertFalse(indexer_kwargs["is_neox_style"])
        self.assertEqual(indexer_kwargs["prefix"], "model.layers.3.self_attn.indexer")
        self.assertEqual(attention.index_topk_output_width, 2048 + 3)

    def test_zero_rope_never_enters_base_fused_rope_metadata_path(self):
        attention = _build(self.module.Glm5NextDSAAttention)

        self.assertIsNone(attention.rotary_emb)
        self.assertFalse(attention._fuse_rope_for_trtllm_mla(object()))

    def test_checkpoint_parameters_stay_under_indexer_namespace(self):
        attention = _build(self.module.Glm5NextDSAAttention)
        names = set(dict(attention.named_parameters()))

        self.assertIn("indexer.index_kpool_compress_ape", names)
        self.assertIn("indexer.index_kpool_compress_gate", names)
        self.assertIn("indexer.wk.weight", names)
        self.assertFalse(any(name.startswith("dsa_indexer.") for name in names))

    def test_kpool4_checkpoint_invariants_are_asserted(self):
        cases = (
            ("index_kpool", 2, "index_kpool=4"),
            ("index_kpool_compress", False, "index_kpool_compress=True"),
            (
                "index_kpool_always_select_tail",
                False,
                "index_kpool_always_select_tail=True",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                config = _Glm5NextTextConfig(**{field: value})
                with self.assertRaisesRegex(AssertionError, message):
                    _build(self.module.Glm5NextDSAAttention, config=config)

    def test_rejects_cp_mtp_and_index_sharing_modes(self):
        invalid_configs = (
            ("index_topk_freq", 2),
            ("index_topk_pattern", "N"),
            ("index_skip_topk_offset", 1),
            ("index_share_for_mtp_iteration", True),
        )
        for field, value in invalid_configs:
            with self.subTest(field=field):
                config = _Glm5NextTextConfig(**{field: value})
                with self.assertRaises(ValueError):
                    _build(self.module.Glm5NextDSAAttention, config=config)

        with self.assertRaisesRegex(NotImplementedError, "MTP"):
            _build(self.module.Glm5NextDSAAttention, is_nextn=True)

        cp_module = _load_module(cp_enabled=True)
        with self.assertRaisesRegex(NotImplementedError, "context parallel"):
            _build(cp_module.Glm5NextDSAAttention)

    def test_rejects_rope_or_dimension_drift(self):
        with self.assertRaisesRegex(ValueError, "skip_rope=True"):
            _build(self.module.Glm5NextDSAAttention, skip_rope=False)
        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim"):
            _build(self.module.Glm5NextDSAAttention, qk_rope_head_dim=16)
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            _build(self.module.Glm5NextDSAAttention, hidden_size=8192)


if __name__ == "__main__":
    unittest.main()
