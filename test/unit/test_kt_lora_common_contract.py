import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import LoRAType, get_target_module_name
from sglang.srt.server_args import (
    ServerArgs,
    _prepare_kt_composite_lora_adapter,
    _validate_kt_expert_lora_adapter_path,
    prepare_server_args,
)
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES


class _ShapeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(1))

    def get_hidden_dim(self, module_name, layer_idx):
        assert module_name == "gate"
        return 8, 6


class _ReplicatedWrapper:
    def slice_lora_a_weights(self, weight, tp_rank):
        return weight

    def slice_lora_b_weights(self, weight, tp_rank):
        return weight


class _ColumnWrapper(_ReplicatedWrapper):
    def slice_lora_b_weights(self, weight, tp_rank):
        shard = weight.shape[0] // 2
        return weight[tp_rank * shard : (tp_rank + 1) * shard]


class _RowWrapper(_ReplicatedWrapper):
    def slice_lora_a_weights(self, weight, tp_rank):
        shard = weight.shape[1] // 2
        return weight[:, tp_rank * shard : (tp_rank + 1) * shard]


class _LocalTP1Wrapper(_ReplicatedWrapper):
    base_layer = SimpleNamespace(tp_rank=0, tp_size=1)

    def slice_lora_b_weights(self, weight, tp_rank):
        assert tp_rank == 0
        return weight


def _shape_pool(wrapper):
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.base_hf_config = SimpleNamespace()
    pool.max_loras_per_batch = 2
    pool.tp_size = 2
    pool.tp_rank = 1
    pool._target_module_wrappers = {"gate": {0: [wrapper]}}
    return pool


def _write_adapter(path, tensors):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
            }
        )
    )
    save_file(tensors, str(path / "adapter_model.safetensors"))


def test_target_resolution_is_exact_and_deterministic():
    targets = {"gate", "gate_up_proj", "shared_expert_gate", "mlp.gate"}

    assert (
        get_target_module_name("model.layers.0.mlp.gate.lora_A.default.weight", targets)
        == "mlp.gate"
    )
    assert (
        get_target_module_name("model.layers.0.mlp.gate_up_proj", targets)
        == "gate_up_proj"
    )
    assert (
        get_target_module_name("model.layers.0.mlp.shared_expert_gate", targets)
        == "shared_expert_gate"
    )


def test_target_resolution_rejects_substrings():
    with pytest.raises(ValueError, match="complete path segments"):
        get_target_module_name("model.layers.0.mlp.gate_up_proj", {"gate"})


@pytest.mark.parametrize(
    ("wrapper", "a_shape", "b_shape"),
    [
        (_ReplicatedWrapper(), (2, 4, 8), (2, 6, 4)),
        (_ColumnWrapper(), (2, 4, 8), (2, 3, 4)),
        (_RowWrapper(), (2, 4, 4), (2, 6, 4)),
    ],
)
def test_memory_pool_shape_uses_actual_module_tp_contract(wrapper, a_shape, b_shape):
    pool = _shape_pool(wrapper)
    model = _ShapeModel()

    assert pool.get_lora_A_shape("gate", model, 4, 0) == a_shape
    assert pool.get_lora_B_shape("gate", model, 4, 0) == b_shape


def test_memory_pool_rejects_incompatible_wrappers_for_same_target():
    pool = _shape_pool(_ReplicatedWrapper())
    pool._target_module_wrappers["gate"][0].append(_ColumnWrapper())

    with pytest.raises(ValueError, match="incompatible TP-local shapes"):
        pool._slice_with_target_wrappers(
            "gate",
            0,
            torch.empty((6, 4), device="meta"),
            LoRAType.LORA_B,
        )


def test_memory_pool_uses_base_layer_local_tp_rank():
    pool = _shape_pool(_LocalTP1Wrapper())

    assert pool.get_lora_B_shape("gate", _ShapeModel(), 4, 0) == (2, 6, 4)


def test_adapter_tensor_without_wrapper_fails_consumption_audit():
    manager = LoRAManager.__new__(LoRAManager)
    manager.target_modules = {"gate"}
    manager.lora_modules = [{}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    adapter = SimpleNamespace(
        layers=[
            SimpleNamespace(
                weights={
                    "base_model.model.model.layers.0.mlp.gate.lora_A.weight": (
                        torch.empty(2, 8)
                    )
                }
            )
        ],
        embedding_layers={},
        _source_lora_weight_names=(),
    )

    with pytest.raises(ValueError, match="no executable LoRA wrapper"):
        manager.validate_lora_weight_consumption(adapter, "unconsumed")


def test_unknown_top_level_lora_tensor_fails_consumption_audit():
    manager = LoRAManager.__new__(LoRAManager)
    manager.target_modules = {"gate"}
    manager.lora_modules = [{}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    adapter = SimpleNamespace(
        layers=[SimpleNamespace(weights={})],
        embedding_layers={},
        _source_lora_weight_names=(
            "base_model.model.router_without_layer.lora_A.weight",
        ),
    )

    with pytest.raises(ValueError, match="would otherwise be silently ignored"):
        manager.validate_lora_weight_consumption(adapter, "dropped")


def test_unknown_top_level_non_lora_tensor_fails_consumption_audit():
    manager = LoRAManager.__new__(LoRAManager)
    manager.target_modules = {"gate"}
    manager.lora_modules = [{}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    adapter = SimpleNamespace(
        layers=[SimpleNamespace(weights={})],
        embedding_layers={},
        added_tokens_embeddings={},
        _source_lora_weight_names=("unsupported_top_level.weight",),
    )

    with pytest.raises(ValueError, match="explicit runtime consumer"):
        manager.validate_lora_weight_consumption(adapter, "dropped")


def test_routed_expert_tensor_rejected_from_ordinary_pool():
    manager = LoRAManager.__new__(LoRAManager)
    manager.target_modules = {"gate_up_proj"}
    manager.lora_modules = [
        {"model.layers.0.mlp.shared_expert.gate_up_proj": _ReplicatedWrapper()}
    ]
    manager.embed_tokens_module = None
    manager.lm_head_module = None
    expert_name = (
        "base_model.model.model.layers.0.mlp.experts.0."
        "gate_up_proj.lora_A.weight"
    )
    adapter = SimpleNamespace(
        layers=[SimpleNamespace(weights={expert_name: torch.empty(2, 8)})],
        embedding_layers={},
        _source_lora_weight_names=(expert_name,),
    )

    with pytest.raises(ValueError, match="--kt-expert-lora-path"):
        manager.validate_lora_weight_consumption(adapter, "expert-via-tensor-api")


def test_cli_all_allocates_only_targets_executable_by_this_model():
    manager = LoRAManager.__new__(LoRAManager)
    manager.target_modules = {
        "gate",
        "gate_up_proj",
        "qkv_proj",
        "shared_expert_gate",
    }
    manager.lora_modules = [{"model.layers.0.mlp.gate": _ReplicatedWrapper()}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None

    assert manager.get_runtime_target_modules() == {"gate"}


def test_expert_only_adapter_rejected_from_lora_paths(tmp_path):
    adapter = tmp_path / "expert"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(16, 2),
        },
    )

    with pytest.raises(ValueError, match="--kt-expert-lora-path"):
        _prepare_kt_composite_lora_adapter(str(adapter))

    _validate_kt_expert_lora_adapter_path(str(adapter), allow_nonexpert=False)


def test_dedicated_expert_path_rejects_merged_adapter(tmp_path):
    adapter = tmp_path / "merged"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(
                2, 8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(
                16, 2
            ),
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )

    with pytest.raises(ValueError, match="--lora-paths"):
        _validate_kt_expert_lora_adapter_path(
            str(adapter), allow_nonexpert=False
        )


def test_composite_cache_hit_does_not_materialize_source_tensors(
    tmp_path, monkeypatch
):
    adapter = tmp_path / "merged"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(
                2, 8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(
                16, 2
            ),
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )
    monkeypatch.setenv("SGLANG_KT_LORA_CACHE_DIR", str(tmp_path / "cache"))
    first = _prepare_kt_composite_lora_adapter(str(adapter))

    def fail_load(*args, **kwargs):
        raise AssertionError("cache hit materialized the source adapter")

    monkeypatch.setattr("safetensors.torch.load_file", fail_load)
    assert _prepare_kt_composite_lora_adapter(str(adapter)) == first


def test_prepared_composite_survives_server_args_asdict_round_trip(
    tmp_path, monkeypatch
):
    adapter = tmp_path / "merged"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(
                2, 8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(
                16, 2
            ),
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )
    monkeypatch.setenv("SGLANG_KT_LORA_CACHE_DIR", str(tmp_path / "cache"))

    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = [f"production={adapter}"]
    args.kt_weight_path = "/base/expert/weights"
    args.enable_piecewise_cuda_graph = True
    args.check_lora_server_args()

    assert args.disable_cuda_graph
    assert not args.enable_piecewise_cuda_graph
    reconstructed = ServerArgs(**dataclasses.asdict(args))
    # ``model_path=dummy`` intentionally skips ServerArgs.__post_init__ checks;
    # run the same normalization that a real reconstructed Engine performs.
    reconstructed.check_lora_server_args()
    assert reconstructed.kt_composite_lora_name == "production"
    assert reconstructed.kt_composite_lora_id == args.kt_composite_lora_id
    assert reconstructed.lora_paths[0].lora_id == args.lora_paths[0].lora_id


def test_composite_lora_fails_fast_for_pipeline_parallelism(tmp_path, monkeypatch):
    adapter = tmp_path / "merged"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(
                2, 8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(
                16, 2
            ),
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )
    monkeypatch.setenv("SGLANG_KT_LORA_CACHE_DIR", str(tmp_path / "cache"))

    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = [f"production={adapter}"]
    args.kt_weight_path = "/base/expert/weights"
    args.pp_size = 2

    with pytest.raises(ValueError, match="--pp-size 1"):
        args.check_lora_server_args()


def test_raw_compact_training_adapter_fails_fast(tmp_path):
    adapter = tmp_path / "raw"
    _write_adapter(
        adapter,
        {
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )
    save_file(
        {"layers.0.gate_lora_a": torch.empty(1)},
        str(adapter / "fused_expert_lora.safetensors"),
    )

    with pytest.raises(ValueError, match="raw KT training adapter"):
        _prepare_kt_composite_lora_adapter(str(adapter))


def test_kt_lora_requires_base_expert_weight_path():
    args = ServerArgs.__new__(ServerArgs)
    args.kt_lora_path = None
    args.kt_expert_lora_path = "/converted/expert"
    args.kt_weight_path = None

    with pytest.raises(ValueError, match="--kt-weight-path"):
        args._validate_kt_lora_serving_paths()


def test_explicit_expert_path_cannot_mix_with_lora_paths():
    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = ["/ordinary/adapter"]
    args.kt_expert_lora_path = "/converted/expert"

    with pytest.raises(ValueError, match="cannot be combined"):
        args.check_lora_server_args()


def test_forged_composite_identity_cannot_pair_unrelated_adapters(tmp_path):
    expert = tmp_path / "expert"
    _write_adapter(
        expert,
        {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.empty(
                2, 8
            ),
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.empty(
                16, 2
            ),
        },
    )
    nonexpert = tmp_path / "nonexpert"
    _write_adapter(
        nonexpert,
        {
            "model.layers.0.mlp.gate.lora_A.weight": torch.empty(2, 8),
            "model.layers.0.mlp.gate.lora_B.weight": torch.empty(6, 2),
        },
    )

    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = [
        {
            "lora_id": "forged-id",
            "lora_name": "forged",
            "lora_path": str(nonexpert),
        }
    ]
    args.kt_weight_path = "/base/expert/weights"
    args.kt_expert_lora_path = str(expert)
    args.kt_composite_lora_name = "forged"
    args.kt_composite_lora_id = "forged-id"

    with pytest.raises(ValueError, match="cannot be combined"):
        args.check_lora_server_args()


def test_ordinary_lora_dict_without_id_gets_generated_identity():
    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = [
        {"lora_name": "ordinary", "lora_path": "/ordinary/adapter"}
    ]

    args.check_lora_server_args()

    assert args.lora_paths[0].lora_name == "ordinary"
    assert args.lora_paths[0].lora_id


def test_deprecated_kt_lora_path_is_expert_only_alias():
    args = ServerArgs.__new__(ServerArgs)
    args.kt_lora_path = "/converted/expert"
    args.kt_expert_lora_path = None

    args._normalize_deprecated_kt_lora_path()

    assert args.kt_lora_path is None
    assert args.kt_expert_lora_path == "/converted/expert"


def test_model_specific_targets_are_public_cli_choices():
    required = {
        "gate",
        "shared_expert_gate",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    }
    assert required.issubset(SUPPORTED_LORA_TARGET_MODULES)

    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--lora-target-modules",
            *sorted(required),
        ]
    )
    assert set(args.lora_target_modules) == required


def test_lora_target_all_expands_to_full_public_contract():
    args = ServerArgs(model_path="dummy")
    args.enable_lora = True
    args.lora_paths = []
    args.max_lora_rank = 8
    args.lora_target_modules = {"all"}
    args.lora_backend = "triton"
    args.kt_lora_path = None
    args.kt_expert_lora_path = None

    args.check_lora_server_args()

    assert set(SUPPORTED_LORA_TARGET_MODULES) == args.lora_target_modules
