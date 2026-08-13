"""CPU-only tests for the GLM-5-Next phase-3 model boundary."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "python/sglang/srt/configs/glm5_next.py"
MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next.py"
NORM_PATH = REPO_ROOT / "python/sglang/srt/models/glm5_next_norm.py"
MODEL_CONFIG_PATH = REPO_ROOT / "python/sglang/srt/configs/model_config.py"


class _PretrainedConfigStub:
    model_type = ""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_config_module():
    packages = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.configs"):
        module = types.ModuleType(name)
        module.__path__ = []
        packages[name] = module

    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    configuration_utils = types.ModuleType("transformers.configuration_utils")
    configuration_utils.PretrainedConfig = _PretrainedConfigStub
    transformers.configuration_utils = configuration_utils
    packages["transformers"] = transformers
    packages[configuration_utils.__name__] = configuration_utils

    mamba_utils = types.ModuleType("sglang.srt.configs.mamba_utils")
    mamba_utils.KimiLinearCacheParams = type("KimiLinearCacheParams", (), {})
    mamba_utils.KimiLinearStateShape = type("KimiLinearStateShape", (), {})
    packages[mamba_utils.__name__] = mamba_utils

    module_name = "_glm5_next_stage3_config"
    spec = importlib.util.spec_from_file_location(module_name, CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    packages[module_name] = module
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


def _load_model_module(config_module):
    packages = {}
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.distributed",
        "sglang.srt.distributed.device_communicators",
        "sglang.srt.eplb",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.model_executor",
        "sglang.srt.models",
        "sglang.srt.models.deepseek_common",
    )
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        packages[name] = module

    packages["sglang.srt.configs.glm5_next"] = config_module

    norm_name = "sglang.srt.models.glm5_next_norm"
    norm_spec = importlib.util.spec_from_file_location(norm_name, NORM_PATH)
    norm_module = importlib.util.module_from_spec(norm_spec)
    assert norm_spec is not None and norm_spec.loader is not None
    norm_spec.loader.exec_module(norm_module)
    packages[norm_name] = norm_module

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )
    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.__path__ = []
    distributed.get_pp_group = lambda: pp_group
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    distributed.get_tp_group = lambda: object()
    packages[distributed.__name__] = distributed

    @contextmanager
    def use_symmetric_memory(*args, **kwargs):
        del args, kwargs
        yield

    pynccl_allocator = types.ModuleType(
        "sglang.srt.distributed.device_communicators.pynccl_allocator"
    )
    pynccl_allocator.use_symmetric_memory = use_symmetric_memory
    packages[pynccl_allocator.__name__] = pynccl_allocator

    expert_distribution = types.ModuleType("sglang.srt.eplb.expert_distribution")
    expert_distribution.get_global_expert_distribution_recorder = lambda: None
    packages[expert_distribution.__name__] = expert_distribution

    class _ScatterMode:
        SCATTERED = object()
        TP_ATTN_FULL = object()
        FULL = object()

    class _LayerScatterModes:
        @classmethod
        def init_new(cls, **kwargs):
            del kwargs
            return SimpleNamespace(
                layer_input_mode=_ScatterMode.FULL,
                attn_mode=_ScatterMode.FULL,
                mlp_mode=_ScatterMode.FULL,
                middle_residual_mode=_ScatterMode.FULL,
                layer_output_mode=_ScatterMode.FULL,
            )

    class _AttnTpContext:
        def __init__(self):
            self.active = False
            self.init_calls = []
            self.entries = 0
            self.exits = 0

        def init_context(self, q_lora_rank, is_nsa):
            self.init_calls.append((q_lora_rank, is_nsa))

        @contextmanager
        def maybe_input_scattered(self, forward_batch):
            del forward_batch
            self.entries += 1
            self.active = True
            try:
                yield
            finally:
                self.active = False
                self.exits += 1

    attn_tp_context = _AttnTpContext()

    communicator = types.ModuleType("sglang.srt.layers.communicator")
    communicator.LayerScatterModes = _LayerScatterModes
    communicator.ScatterMode = _ScatterMode
    communicator.get_attn_tp_context = lambda: attn_tp_context
    packages[communicator.__name__] = communicator

    class _MHCLayerCommunicator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    communicator_mhc = types.ModuleType("sglang.srt.layers.communicator_mhc")
    communicator_mhc.MHCLayerCommunicator = _MHCLayerCommunicator
    packages[communicator_mhc.__name__] = communicator_mhc

    dp_attention = types.ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.get_attention_tp_rank = lambda: 0
    dp_attention.is_allocation_symmetric = lambda: False
    packages[dp_attention.__name__] = dp_attention

    class _ReplicatedLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.kwargs = kwargs

        def forward(self, hidden_states):
            return hidden_states.new_empty(
                (*hidden_states.shape[:-1], self.output_size)
            ), None

    linear = types.ModuleType("sglang.srt.layers.linear")
    linear.ReplicatedLinear = _ReplicatedLinear
    linear.RowParallelLinear = type("RowParallelLinear", (), {})
    packages[linear.__name__] = linear

    class _Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    layernorm = types.ModuleType("sglang.srt.layers.layernorm")
    layernorm.RMSNorm = _Layer
    packages[layernorm.__name__] = layernorm

    logits_processor = types.ModuleType("sglang.srt.layers.logits_processor")
    logits_processor.LogitsProcessor = _Layer
    packages[logits_processor.__name__] = logits_processor

    base_config = types.ModuleType("sglang.srt.layers.quantization.base_config")
    base_config.QuantizationConfig = type("QuantizationConfig", (), {})
    packages[base_config.__name__] = base_config

    unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")
    unquant.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    packages[unquant.__name__] = unquant

    layer_utils = types.ModuleType("sglang.srt.layers.utils")
    layer_utils.PPMissingLayer = _Layer
    packages[layer_utils.__name__] = layer_utils

    embedding = types.ModuleType("sglang.srt.layers.vocab_parallel_embedding")
    embedding.ParallelLMHead = _Layer
    embedding.VocabParallelEmbedding = _Layer
    packages[embedding.__name__] = embedding

    forward_info = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_info.ForwardBatch = type("ForwardBatch", (), {})
    forward_info.PPProxyTensors = dict
    packages[forward_info.__name__] = forward_info

    deepseek_loader = types.ModuleType(
        "sglang.srt.models.deepseek_common.deepseek_weight_loader"
    )
    deepseek_loader.DeepseekV2WeightLoaderMixin = type(
        "DeepseekV2WeightLoaderMixin", (), {}
    )
    packages[deepseek_loader.__name__] = deepseek_loader

    class _KimiDeltaAttention(nn.Module):
        def __init__(self, *, config, **kwargs):
            super().__init__()
            self.config = config
            self.hidden_size = config.hidden_size
            self.num_heads = config.linear_attn_config["num_heads"]
            self.head_dim = config.linear_attn_config["head_dim"]
            self.head_v_dim = config.v_head_dim
            self.local_num_heads = self.num_heads
            self.attn_tp_size = 1
            self.do_fuse_qkvbfg = False
            self.f_a_proj = _ReplicatedLinear(self.hidden_size, self.head_dim)
            self.o_proj = SimpleNamespace(reduce_results=True)
            self.attn = SimpleNamespace(lower_bound=None)

    class _KimiMlaAttention(nn.Module):
        def __init__(self, *, v_head_dim, reduce_results=True, **kwargs):
            super().__init__()
            self.head_v_dim = v_head_dim
            self.o_proj = SimpleNamespace(reduce_results=reduce_results)

        def prepare_qkv_latent(self, hidden_states, forward_batch):
            return hidden_states, forward_batch

    class _KimiDecoderLayer(nn.Module):
        def forward(self, **kwargs):
            self.kimi_forward_kwargs = kwargs
            return kwargs["hidden_states"], kwargs["residual"]

    class _KimiLinearModel(nn.Module):
        def forward(self, **kwargs):
            self.kimi_forward_kwargs = kwargs
            return "kimi-model-forward"

    kimi = types.ModuleType("sglang.srt.models.kimi_linear")
    kimi.KimiDecoderLayer = _KimiDecoderLayer
    kimi.KimiDeltaAttention = _KimiDeltaAttention
    kimi.KimiLinearModel = _KimiLinearModel
    kimi.KimiMLAAttention = _KimiMlaAttention
    kimi.KimiMLP = _Layer
    kimi.KimiMoE = _Layer
    packages[kimi.__name__] = kimi

    glm_dsa = types.ModuleType("sglang.srt.models.glm5_next_dsa")
    glm_dsa.Glm5NextDSAAttention = _KimiMlaAttention
    packages[glm_dsa.__name__] = glm_dsa

    class _Glm5NextMLP(_Layer):
        pass

    class _Glm5NextMoE(_Layer):
        def forward(self, hidden_states, forward_batch=None, **kwargs):
            self.forward_batch = forward_batch
            self.forward_kwargs = kwargs
            return hidden_states + 2

    glm_moe = types.ModuleType("sglang.srt.models.glm5_next_moe")
    glm_moe.Glm5NextMLP = _Glm5NextMLP
    glm_moe.Glm5NextMoE = _Glm5NextMoE
    packages[glm_moe.__name__] = glm_moe

    model_transformers = types.ModuleType("sglang.srt.models.transformers")
    model_transformers.maybe_prefix = lambda prefix, name: (
        f"{prefix}.{name}" if prefix else name
    )
    packages[model_transformers.__name__] = model_transformers

    utils = types.ModuleType("sglang.srt.utils")

    def make_layers(count, factory, *, prefix, **kwargs):
        return (
            nn.ModuleList(factory(i, f"{prefix}.{i}") for i in range(count)),
            0,
            count,
        )

    utils.make_layers = make_layers
    packages[utils.__name__] = utils

    class _BumpAllocator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    common = types.ModuleType("sglang.srt.utils.common")
    common.BumpAllocator = _BumpAllocator
    packages[common.__name__] = common

    module_name = "_glm5_next_stage3_model"
    spec = importlib.util.spec_from_file_location(module_name, MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    packages[module_name] = module
    assert spec.loader is not None
    with patch.dict(sys.modules, packages):
        spec.loader.exec_module(module)
    return module


def _compile_function(path: Path, name: str, class_name: str | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name is not None:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        body = class_node.body
    function = copy.deepcopy(
        next(
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    function.decorator_list = []
    output_body: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    symbol_name = name
    if class_name is None:
        output_body.append(function)
    else:
        symbol_name = "_Harness"
        output_body.append(
            ast.ClassDef(
                name=symbol_name,
                bases=[],
                keywords=[],
                body=[function],
                decorator_list=[],
            )
        )
    compiled = ast.fix_missing_locations(ast.Module(body=output_body, type_ignores=[]))
    namespace = {"copy": copy}
    exec(compile(compiled, str(path), "exec"), namespace)
    value = namespace[symbol_name]
    return value if class_name is None else getattr(value, name)


class TestGlm5NextConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_module = _load_config_module()

    def test_actual_fp8_defaults_partition_45_layers_as_34_kda_11_dsa(self):
        config = self.config_module.Glm5NextTextConfig()

        self.assertEqual(config.num_hidden_layers, 45)
        self.assertEqual(config.linear_num_heads, 64)
        self.assertEqual(config.linear_head_dim, 128)
        self.assertEqual(len(config.linear_layer_ids), 34)
        self.assertEqual(len(config.full_attention_layer_ids), 11)
        self.assertEqual(
            config.full_attention_layer_ids,
            list(range(3, 45, 4)),
        )
        self.assertEqual(
            sorted(config.linear_layer_ids + config.full_attention_layer_ids),
            list(range(45)),
        )

    def test_glm_capabilities_are_exact_and_non_glm_defaults_false(self):
        root = self.config_module.Glm5NextConfig()
        capabilities = self.config_module.get_glm5_next_capabilities(root)

        self.assertTrue(capabilities.is_glm5_next)
        self.assertTrue(capabilities.uses_kpool4_compress)
        self.assertTrue(capabilities.uses_kda_safe_gate)
        self.assertTrue(capabilities.uses_zero_rope_mla)
        self.assertTrue(capabilities.uses_mhc)

        non_glm = SimpleNamespace(
            model_type="kimi_linear",
            architectures=["KimiLinearForCausalLM"],
            index_kpool=4,
            index_kpool_compress=True,
            index_kpool_always_select_tail=True,
            linear_attn_config={"kda_layers": [0], "gate_lower_bound": -5.0},
            qk_rope_head_dim=0,
            full_attention_layer_ids=[1],
            mhc=True,
        )
        self.assertEqual(
            self.config_module.get_glm5_next_capabilities(non_glm),
            self.config_module.Glm5NextCapabilities(False, False, False, False, False),
        )

    def test_nested_vision_config_is_data_only(self):
        root = self.config_module.Glm5NextConfig(
            vision_config={"depth": 24, "hidden_size": 1536}
        )

        self.assertEqual(root.vision_config.model_type, "glm_ocr_vision")
        self.assertEqual(root.vision_config.depth, 24)
        self.assertEqual(root.vision_config.hidden_size, 1536)
        self.assertFalse(
            any(name.startswith("transformers.models.glm_ocr") for name in sys.modules)
        )

    def test_invalid_layer_partition_fails_early(self):
        with self.assertRaisesRegex(ValueError, "layer_types"):
            self.config_module.Glm5NextTextConfig(layer_types=["linear_attention"])


class TestGlm5NextModelBoundary(unittest.TestCase):
    def test_model_config_routes_glm_to_mla_shape_derivation(self):
        tree = ast.parse(
            MODEL_CONFIG_PATH.read_text(encoding="utf-8"),
            filename=str(MODEL_CONFIG_PATH),
        )
        model_config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelConfig"
        )
        derive_shapes = next(
            node
            for node in model_config.body
            if isinstance(node, ast.FunctionDef) and node.name == "_derive_model_shapes"
        )
        source = ast.unparse(derive_shapes)

        self.assertIn("Glm5NextForConditionalGeneration", source)
        self.assertIn("if self.is_glm5_next", source)
        self.assertIn("self.hf_text_config.index_head_dim", source)

    def test_model_import_and_45_layer_construction_are_text_only(self):
        config_module = _load_config_module()
        model_module = _load_model_module(config_module)
        root_config = config_module.Glm5NextConfig()

        # The production dimensions include all mHC parameters.  Meta tensors
        # validate the exact 45-layer construction without allocating them.
        with torch.device("meta"):
            model = model_module.Glm5NextForConditionalGeneration(root_config)
        kda_layers = [
            layer
            for layer in model.model.layers
            if isinstance(layer.self_attn, model_module.Glm5NextLinearAttention)
        ]
        dsa_layers = [
            layer
            for layer in model.model.layers
            if not isinstance(layer.self_attn, model_module.Glm5NextLinearAttention)
        ]

        self.assertEqual(len(model.model.layers), 45)
        self.assertEqual(len(kda_layers), 34)
        self.assertEqual(len(dsa_layers), 11)
        self.assertTrue(all(layer.self_attn.num_heads == 64 for layer in kda_layers))
        self.assertTrue(all(layer.self_attn.head_dim == 128 for layer in kda_layers))
        self.assertTrue(all(layer.self_attn.head_v_dim == 128 for layer in kda_layers))
        self.assertTrue(all(layer.self_attn.head_v_dim == 256 for layer in dsa_layers))
        self.assertEqual(root_config.text_config.v_head_dim, 256)
        self.assertIsNone(model.visual)

    def test_kda_compatibility_view_is_non_mutating_and_128_wide(self):
        build_config = _compile_function(MODEL_PATH, "_kda_construction_config")
        source = SimpleNamespace(
            linear_attn_config={"num_heads": 64, "head_dim": 128},
            linear_num_heads=1,
            linear_head_dim=1,
            v_head_dim=256,
        )

        result = build_config(source)

        self.assertIsNot(result, source)
        self.assertEqual(source.v_head_dim, 256)
        self.assertEqual(result.linear_num_heads, 64)
        self.assertEqual(result.linear_head_dim, 128)
        self.assertEqual(result.v_head_dim, 128)

    def test_kda_beta_projection_is_replicated_then_head_sliced(self):
        config_module = _load_config_module()
        model_module = _load_model_module(config_module)
        config = config_module.Glm5NextTextConfig()

        attention = model_module.Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=config.hidden_size,
            config=config,
            quant_config=object(),
            prefix="model.layers.0.self_attn",
        )

        self.assertEqual(attention.b_proj.input_size, config.hidden_size)
        self.assertEqual(attention.b_proj.output_size, 64)
        self.assertEqual(attention._beta_head_start, 0)
        self.assertEqual(attention._beta_heads_per_rank, 64)
        self.assertEqual(
            attention.b_proj.kwargs["prefix"],
            "model.layers.0.self_attn.b_proj",
        )

        source = MODEL_PATH.read_text(encoding="utf-8")
        self.assertIn("beta = beta.narrow(", source)
        self.assertNotIn("tensor_model_parallel_all_gather(beta", source)

    def test_weight_prefix_and_phase7_visual_whitelist_are_exact(self):
        normalize = _compile_function(MODEL_PATH, "normalize_glm5_next_weight_name")
        normalize.__globals__["GLM5_NEXT_PHASE7_VISUAL_WEIGHT_PREFIXES"] = (
            "visual.",
            "model.visual.",
        )

        self.assertEqual(
            normalize("model.language_model.layers.3.self_attn.q_b_proj.weight"),
            "model.layers.3.self_attn.q_b_proj.weight",
        )
        self.assertEqual(
            normalize("language_model.embed_tokens.weight"),
            "model.embed_tokens.weight",
        )
        self.assertIsNone(normalize("model.visual.blocks.0.attn.qkv.weight"))
        self.assertIsNone(normalize("visual.patch_embed.proj.weight"))
        self.assertEqual(
            normalize("model.language_model.layers.0.visual_gate.weight"),
            "model.layers.0.visual_gate.weight",
        )

    def test_fp8_qkv_scale_uses_the_same_shard_mapping_as_weight(self):
        load_stacked = _compile_function(
            MODEL_PATH,
            "_load_kda_stacked_weight",
            class_name="Glm5NextForConditionalGeneration",
        )
        calls = []

        class Param:
            pass

        param = Param()
        param.weight_loader = lambda param, tensor, shard: calls.append(
            (param, tensor, shard)
        )
        tensor = object()
        params = {
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv": param,
        }

        loaded = load_stacked(
            object(),
            "model.layers.0.self_attn.q_proj.weight_scale_inv",
            tensor,
            params,
        )

        self.assertTrue(loaded)
        self.assertEqual(calls, [(param, tensor, "q")])

    def test_model_registration_and_lazy_vision_import_boundary(self):
        source = MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(MODEL_PATH))
        entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "EntryClass"
                for target in node.targets
            )
        )
        self.assertEqual(ast.unparse(entry.value), "[Glm5NextForConditionalGeneration]")
        self.assertNotIn("sglang.srt.models.glm_ocr", source)
        self.assertNotIn('if "visual" in name', source)


class TestGlm5NextMHCModelBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_module = _load_config_module()
        cls.model_module = _load_model_module(cls.config_module)

    def _config(self, mhc=True):
        return self.config_module.Glm5NextTextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            n_routed_experts=None,
            mhc=mhc,
            hc_mult=2,
            linear_num_heads=2,
            linear_head_dim=2,
            v_head_dim=2,
            qk_nope_head_dim=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
        )

    def test_mhc_parameters_and_communicator_layer_boundaries(self):
        first = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)
        last = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=1)

        expected_shapes = {
            "input_layernorm.weight": (4,),
            "post_attention_layernorm.weight": (4,),
            "hc_attn_base": (8,),
            "hc_attn_scale": (3,),
            "hc_attn_fn": (8, 8),
            "hc_ffn_base": (8,),
            "hc_ffn_scale": (3,),
            "hc_ffn_fn": (8, 8),
        }
        named_parameters = dict(first.named_parameters())
        self.assertEqual(set(named_parameters), set(expected_shapes))
        for name, shape in expected_shapes.items():
            parameter = named_parameters[name]
            self.assertEqual(tuple(parameter.shape), shape)
            self.assertEqual(parameter.dtype, torch.float32)

        self.assertFalse(first.self_attn.o_proj.reduce_results)
        self.assertTrue(first.layer_communicator.kwargs["is_first_layer"])
        self.assertFalse(first.layer_communicator.kwargs["is_last_layer"])
        self.assertFalse(last.layer_communicator.kwargs["is_first_layer"])
        self.assertTrue(last.layer_communicator.kwargs["is_last_layer"])
        self.assertIs(first.layer_communicator.kwargs["hc_attn_pre"].__self__, first)
        self.assertIs(first.layer_communicator.kwargs["hc_ffn_pre"].__self__, first)
        self.assertIs(first.layer_communicator.kwargs["hc_post"].__self__, first)
        self.assertIs(
            first.layer_communicator.kwargs["attn_all_reduce_output_dtype"],
            torch.bfloat16,
        )
        self.assertIsNone(
            last.layer_communicator.kwargs["attn_all_reduce_output_dtype"]
        )
        self.assertIsNone(first.layer_communicator.kwargs["qkv_latent_func"])
        dsa_latent_func = last.layer_communicator.kwargs["qkv_latent_func"]
        self.assertIs(dsa_latent_func.__self__, last.self_attn)
        self.assertIs(
            dsa_latent_func.__func__, last.self_attn.prepare_qkv_latent.__func__
        )

    def test_glm_norm_matches_checkpoint_bf16_rounding_boundary(self):
        norm = self.model_module.Glm5NextRMSNorm(17, eps=1e-6).to(torch.bfloat16)
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(3, 17, generator=generator).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(
                torch.randn(17, generator=generator).to(torch.bfloat16)
            )

        actual = norm(hidden_states)
        hidden_states_fp32 = hidden_states.float()
        normalized_fp32 = hidden_states_fp32 * torch.rsqrt(
            hidden_states_fp32.pow(2).mean(-1, keepdim=True) + 1e-6
        )
        expected = norm.weight * normalized_fp32.to(torch.bfloat16)
        shared_optimized_order = (norm.weight.float() * normalized_fp32).to(
            torch.bfloat16
        )

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, shared_optimized_order))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_glm_norm_cuda_graph_bs_1_2_4_replays_dynamic_inputs_in_place(self):
        hidden_size = 17
        generator = torch.Generator().manual_seed(20260812)
        weight = torch.randn(hidden_size, generator=generator).to(torch.bfloat16)

        def checkpoint_reference(norm, hidden_states):
            hidden_states_fp32 = hidden_states.float()
            normalized_fp32 = hidden_states_fp32 * torch.rsqrt(
                hidden_states_fp32.pow(2).mean(-1, keepdim=True) + 1e-6
            )
            return norm.weight * normalized_fp32.to(hidden_states.dtype)

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                norm = self.model_module.Glm5NextRMSNorm(hidden_size, eps=1e-6).to(
                    device="cuda", dtype=torch.bfloat16
                )
                with torch.no_grad():
                    norm.weight.copy_(weight)

                input_a = torch.randn(batch_size, hidden_size, generator=generator).to(
                    device="cuda", dtype=torch.bfloat16
                )
                poison = torch.randn(batch_size, hidden_size, generator=generator).to(
                    device="cuda", dtype=torch.bfloat16
                )
                static_input = input_a.clone()
                expected_a = checkpoint_reference(norm, input_a)

                # Warm up all kernels and allocator paths before capture.
                norm(static_input)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = norm(static_input)

                stable_pointers = {
                    "input": static_input.data_ptr(),
                    "output": static_output.data_ptr(),
                    "weight": norm.weight.data_ptr(),
                }

                def replay(value):
                    static_input.copy_(value)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertEqual(static_input.data_ptr(), stable_pointers["input"])
                    self.assertEqual(
                        static_output.data_ptr(), stable_pointers["output"]
                    )
                    self.assertEqual(norm.weight.data_ptr(), stable_pointers["weight"])
                    return static_output.clone()

                first_a = replay(input_a)
                poison_output = replay(poison)
                second_a = replay(input_a)

                torch.testing.assert_close(first_a, expected_a, rtol=0, atol=0)
                self.assertFalse(torch.equal(poison_output, first_a))
                torch.testing.assert_close(second_a, first_a, rtol=0, atol=0)

    def test_all_glm_decoder_and_final_norms_are_model_local(self):
        model = self.model_module.Glm5NextModel(self._config())

        for layer in model.layers:
            self.assertIsInstance(
                layer.input_layernorm, self.model_module.Glm5NextRMSNorm
            )
            self.assertIsInstance(
                layer.post_attention_layernorm, self.model_module.Glm5NextRMSNorm
            )
        self.assertIsInstance(model.norm, self.model_module.Glm5NextRMSNorm)

    def test_top_level_forward_owns_dsa_latent_context_lifetime(self):
        model = self.model_module.Glm5NextForConditionalGeneration(self._config())
        attn_context = self.model_module.get_attn_tp_context()

        class _Inner(nn.Module):
            def forward(inner_self, *args):
                del inner_self, args
                self.assertTrue(attn_context.active)
                return torch.ones(1, 4)

        class _Logits(nn.Module):
            def forward(inner_self, input_ids, hidden_states, lm_head, batch):
                del inner_self, input_ids, lm_head, batch
                self.assertFalse(attn_context.active)
                return hidden_states

        model.model = _Inner()
        model.logits_processor = _Logits()
        entries_before = attn_context.entries
        exits_before = attn_context.exits

        output = model(
            input_ids=torch.tensor([0]),
            positions=torch.tensor([0]),
            forward_batch=object(),
        )

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertFalse(attn_context.active)
        self.assertEqual(attn_context.entries, entries_before + 1)
        self.assertEqual(attn_context.exits, exits_before + 1)
        self.assertEqual(
            attn_context.init_calls[-1], (self._config().q_lora_rank, True)
        )

    def test_callbacks_select_the_matching_checkpoint_parameters(self):
        layer = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)
        calls = []

        def hc_pre(**kwargs):
            calls.append(("pre", kwargs))
            return "pre-result"

        packages = {}
        for name in (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.layernorm",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            packages[name] = module
        mhc = types.ModuleType("sglang.kernels.ops.layernorm.mhc")
        mhc.hc_pre = hc_pre
        packages[mhc.__name__] = mhc

        x = torch.zeros(1, 8)
        with patch.dict(sys.modules, packages):
            self.assertEqual(layer.hc_attn_pre(x, "weight", 1e-4), "pre-result")
            self.assertEqual(layer.hc_ffn_pre(x, "weight", 1e-4), "pre-result")

        attn_kwargs = calls[0][1]
        ffn_kwargs = calls[1][1]
        self.assertIs(attn_kwargs["hc_fn"], layer.hc_attn_fn)
        self.assertIs(attn_kwargs["hc_scale"], layer.hc_attn_scale)
        self.assertIs(attn_kwargs["hc_base"], layer.hc_attn_base)
        self.assertIs(ffn_kwargs["hc_fn"], layer.hc_ffn_fn)
        self.assertIs(ffn_kwargs["hc_scale"], layer.hc_ffn_scale)
        self.assertIs(ffn_kwargs["hc_base"], layer.hc_ffn_base)
        self.assertEqual(attn_kwargs["sinkhorn_iters"], 20)

    def test_hc_post_uses_glm_bf16_matmul_semantics(self):
        layer = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)
        generator = torch.Generator().manual_seed(0)
        tokens, hc_mult, hidden_size = 3, layer.config.hc_mult, 17
        hidden_states = (
            torch.randn(tokens, hidden_size, generator=generator) * 3
        ).to(torch.bfloat16)
        residual_streams = (
            torch.randn(tokens, hc_mult, hidden_size, generator=generator) * 3
        ).to(torch.bfloat16)
        residual = residual_streams.flatten(1)
        h_post = torch.randn(tokens, hc_mult, generator=generator)
        h_res_matrix = torch.randn(tokens, hc_mult, hc_mult, generator=generator)
        h_res = h_res_matrix.flatten(1)

        actual = layer.hc_post(hidden_states, residual, h_res, h_post)
        expected = (
            h_post.to(torch.bfloat16).unsqueeze(-1) * hidden_states.unsqueeze(1)
            + torch.matmul(
                h_res_matrix.to(torch.bfloat16).transpose(-1, -2),
                residual_streams,
            )
        ).flatten(1)
        elementwise_sum = (
            h_post.to(torch.bfloat16).unsqueeze(-1) * hidden_states.unsqueeze(1)
            + (
                h_res_matrix.to(torch.bfloat16)
                .transpose(-1, -2)
                .unsqueeze(-1)
                * residual_streams.unsqueeze(1)
            ).sum(dim=2)
        ).flatten(1)
        late_cast = (
            h_post.unsqueeze(-1) * hidden_states.float().unsqueeze(1)
            + torch.matmul(h_res_matrix.transpose(-1, -2), residual_streams.float())
        ).to(torch.bfloat16).flatten(1)

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, elementwise_sum))
        self.assertFalse(torch.equal(actual, late_cast))

    def test_mhc_layer_forward_orders_attention_mlp_and_post(self):
        layer = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)
        calls = []

        class _Communicator:
            def prepare_attn(self, hidden_states, residual, forward_batch):
                calls.append("prepare_attn")
                self_outer.assertEqual(tuple(hidden_states.shape), (1, 4))
                self_outer.assertIsNone(residual)
                return hidden_states, hidden_states.repeat(1, 2)

            def prepare_mlp(self, hidden_states, residual, forward_batch):
                calls.append("prepare_mlp")
                self_outer.assertEqual(tuple(residual.shape), (1, 8))
                return hidden_states, residual

            def postprocess_layer(self, hidden_states, residual, forward_batch):
                calls.append("postprocess")
                return residual + hidden_states.repeat(1, 2), None

        class _Attention(nn.Module):
            def forward(self, **kwargs):
                calls.append("attention")
                return kwargs["hidden_states"] + 1

        class _MLP(nn.Module):
            def forward(self, hidden_states):
                calls.append("mlp")
                return hidden_states + 2

        self_outer = self
        layer.layer_communicator = _Communicator()
        layer.self_attn = _Attention()
        layer.mlp = _MLP()
        output, residual = layer(
            positions=torch.tensor([0]),
            hidden_states=torch.ones(1, 4),
            forward_batch=object(),
            residual=None,
            zero_allocator=object(),
        )

        self.assertEqual(
            calls,
            ["prepare_attn", "attention", "prepare_mlp", "mlp", "postprocess"],
        )
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertIsNone(residual)

    def test_mhc_sparse_moe_receives_forward_batch(self):
        layer = self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)
        forward_batch = object()

        class _Communicator:
            def prepare_attn(self, hidden_states, residual, batch):
                self_outer.assertIs(batch, forward_batch)
                return hidden_states, hidden_states.repeat(1, 2)

            def prepare_mlp(self, hidden_states, residual, batch):
                self_outer.assertIs(batch, forward_batch)
                return hidden_states, residual

            def postprocess_layer(self, hidden_states, residual, batch):
                self_outer.assertIs(batch, forward_batch)
                return hidden_states.repeat(1, 2), None

        class _Attention(nn.Module):
            def forward(self, **kwargs):
                return kwargs["hidden_states"]

        self_outer = self
        layer.layer_communicator = _Communicator()
        layer.self_attn = _Attention()
        layer.mlp = self.model_module.Glm5NextMoE()

        output, residual = layer(
            positions=torch.tensor([0]),
            hidden_states=torch.ones(1, 4),
            forward_batch=forward_batch,
            residual=None,
            zero_allocator=object(),
        )

        self.assertIs(layer.mlp.forward_batch, forward_batch)
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertIsNone(residual)

    def test_non_mhc_overrides_fail_closed_before_execution(self):
        for disabled_value in (False, 1):
            with self.subTest(disabled_value=disabled_value):
                with self.assertRaisesRegex(ValueError, "requires.*mhc=True"):
                    self.model_module.Glm5NextDecoderLayer(
                        self._config(mhc=disabled_value), layer_idx=0
                    )
                with self.assertRaisesRegex(ValueError, "requires.*mhc=True"):
                    self.model_module.Glm5NextModel(self._config(mhc=disabled_value))

    def test_mhc_model_keeps_expanded_state_until_the_last_layer(self):
        model = self.model_module.Glm5NextModel(self._config())
        widths = []

        class _Stage(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

            def forward(self, **kwargs):
                hidden_states = kwargs["hidden_states"]
                widths.append(hidden_states.shape[-1])
                self_outer.assertIsNone(kwargs["residual"])
                if self.layer_idx == 0:
                    return hidden_states.repeat(1, 2), None
                return hidden_states[:, :4], None

        class _Norm(nn.Module):
            def forward(self, hidden_states):
                widths.append(hidden_states.shape[-1])
                return hidden_states + 1

        self_outer = self
        model.layers = nn.ModuleList([_Stage(0), _Stage(1)])
        model.norm = _Norm()
        recorder = SimpleNamespace(with_current_layer=lambda layer_idx: nullcontext())
        model_globals = self.model_module.Glm5NextModel.forward.__globals__
        with patch.dict(
            model_globals,
            {"get_global_expert_distribution_recorder": lambda: recorder},
        ):
            output = model(
                input_ids=None,
                positions=torch.tensor([0]),
                forward_batch=object(),
                inputs_embeds=torch.ones(1, 4),
            )

        self.assertEqual(widths, [4, 8, 4])
        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertFalse(hasattr(model, "kimi_forward_kwargs"))

    def test_scattered_mhc_modes_fail_during_layer_construction(self):
        scattered = self.model_module.ScatterMode.SCATTERED
        modes = SimpleNamespace(
            layer_input_mode=scattered,
            attn_mode=self.model_module.ScatterMode.FULL,
            mlp_mode=self.model_module.ScatterMode.FULL,
            middle_residual_mode=self.model_module.ScatterMode.FULL,
            layer_output_mode=self.model_module.ScatterMode.FULL,
        )
        with patch.object(
            self.model_module.LayerScatterModes,
            "init_new",
            return_value=modes,
        ):
            with self.assertRaisesRegex(NotImplementedError, "scattered"):
                self.model_module.Glm5NextDecoderLayer(self._config(), layer_idx=0)


if __name__ == "__main__":
    unittest.main()
