from types import SimpleNamespace

import pytest
import torch

from sglang.srt.elastic_ep.daemon_hbm_expert_source import (
    DaemonHBMSourceRegistry,
    select_daemon_hbm_source_slot,
)
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    maybe_rebalance_after_rank_fault,
)
from sglang.srt.eplb import eplb_manager
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.weight_cache.expert_source import ExpertSlotKey
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.mooncake_expert_source import (
    MooncakeExpertSourceDescriptor,
    MooncakeRemoteSlot,
)


def _descriptor(source_id):
    return MooncakeExpertSourceDescriptor(
        source_id=source_id,
        session_id=f"session-{source_id}",
        slot_pointers={
            ExpertSlotKey(0, 1, 0): MooncakeRemoteSlot(pointer=1234, byte_size=16)
        },
    )


def test_select_daemon_hbm_source_uses_pre_fault_membership():
    descriptor = _descriptor("rank-7")
    source_rank, selected, source_slot = select_daemon_hbm_source_slot(
        old_physical_to_logical=[0, 1, 2, 3],
        logical_expert_id=3,
        num_local_physical_experts=2,
        old_ep_world_ranks=[5, 7],
        registry=DaemonHBMSourceRegistry({7: descriptor}),
    )

    assert source_rank == 7
    assert selected is descriptor
    assert source_slot == 1


def test_select_daemon_hbm_source_rejects_missing_daemon():
    with pytest.raises(RuntimeError, match="No retained daemon-HBM source"):
        select_daemon_hbm_source_slot(
            old_physical_to_logical=[0, 1],
            logical_expert_id=1,
            num_local_physical_experts=1,
            old_ep_world_ranks=[4, 5],
            registry=DaemonHBMSourceRegistry({4: _descriptor("rank-4")}),
        )


def test_eplb_daemon_installs_initial_slot_layout_before_model_load(monkeypatch):
    daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
    daemon.server_args = SimpleNamespace(enable_eplb=True)
    model_config = object()
    metadata = object()
    calls = []

    monkeypatch.setattr(
        "sglang.srt.weight_cache.daemon.get_parallel",
        lambda: SimpleNamespace(moe_ep_rank=3),
    )
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_location.compute_initial_expert_location_metadata",
        lambda **kwargs: calls.append(("compute", kwargs)) or metadata,
    )
    monkeypatch.setattr(
        "sglang.srt.eplb.expert_location.set_global_expert_location_metadata",
        lambda value: calls.append(("install", value)),
    )

    daemon._initialize_eplb_expert_location_metadata(model_config)

    assert calls == [
        (
            "compute",
            {
                "server_args": daemon.server_args,
                "model_config": model_config,
                "moe_ep_rank": 3,
            },
        ),
        ("install", metadata),
    ]


def test_ep_timeout_membership_change_rebalances(monkeypatch):
    state = ElasticEPState(
        active_ranks=torch.tensor([True, True]),
        last_active_ranks=torch.tensor([True, True]),
        active_ranks_cpu=torch.tensor([True, True]),
        effective_ep_size=2,
    )
    monkeypatch.setattr(ElasticEPStateManager, "_instance", state)
    rebalance_calls = []

    state.active_ranks[1] = False
    assert maybe_rebalance_after_rank_fault(
        eplb_manager=SimpleNamespace(
            rebalance=lambda: (rebalance_calls.append(True), iter(()))[1]
        ),
    )
    assert rebalance_calls == [True]
    assert state.active_ranks.tolist() == [1, 0]
    assert state.active_ranks_cpu.tolist() == [1, 0]
    assert state.last_active_ranks.tolist() == [1, 0]


class _Updater:
    def __init__(self, missing):
        self.missing = missing
        self.calls = []

    def update(self, *_args, **kwargs):
        self.calls.append(("update", kwargs["commit"]))
        return self.missing

    def commit(self, metadata, layer_ids):
        self.calls.append(("commit", metadata, layer_ids))


class _HBMClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def restore_missing_experts(self, **_kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected restore failure")


class _DiskUpdater:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def update_weights_from_disk(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _run_recovery(monkeypatch, *, hbm_client, disk_updater, updater=None):
    old_metadata = object()
    monkeypatch.setattr(
        eplb_manager, "get_global_expert_location_metadata", lambda: old_metadata
    )
    updater = updater or _Updater({0: [3]})
    new_metadata = object()
    model = SimpleNamespace(
        routed_experts_weights_of_layer={},
        model_path="/models/test",
        load_format="safetensors",
        generate_weight_name_filter=lambda missing: lambda name: missing == {0: [3]},
    )
    eplb_manager.update_expert_location_with_recovery(
        expert_location_updater=updater,
        model=model,
        new_expert_location_metadata=new_metadata,
        update_layer_ids=[0],
        nnodes=2,
        tp_rank=0,
        daemon_hbm_source_client=hbm_client,
        update_weights_from_disk_callable=disk_updater.update_weights_from_disk,
        ep_dispatch_algorithm="none",
        init_lplb_solvers_callable=lambda: None,
    )
    return updater, new_metadata


def test_hbm_recovery_commits_only_after_hbm_restore(monkeypatch):
    hbm = _HBMClient()
    disk = _DiskUpdater()
    updater, new_metadata = _run_recovery(
        monkeypatch, hbm_client=hbm, disk_updater=disk
    )

    assert hbm.calls == 1
    assert disk.calls == []
    assert updater.calls == [("update", False), ("commit", new_metadata, [0])]


def test_hbm_failure_falls_back_to_disk_before_commit(monkeypatch):
    hbm = _HBMClient(fail=True)
    disk = _DiskUpdater()
    updater = _Updater({0: [3]})
    updater, new_metadata = _run_recovery(
        monkeypatch, hbm_client=hbm, disk_updater=disk, updater=updater
    )

    assert hbm.calls == 1
    assert disk.calls[0][0] == ("/models/test", "safetensors")
    assert disk.calls[0][1]["weight_name_filter"]("expert")
    assert disk.calls[0][1]["allow_weight_cache"] is True
    assert updater.calls == [("update", False), ("commit", new_metadata, [0])]


def test_missing_hbm_source_falls_back_to_disk_before_commit(monkeypatch):
    disk = _DiskUpdater()
    updater, new_metadata = _run_recovery(
        monkeypatch, hbm_client=None, disk_updater=disk
    )

    assert disk.calls[0][0] == ("/models/test", "safetensors")
    assert disk.calls[0][1]["allow_weight_cache"] is True
    assert updater.calls == [("update", False), ("commit", new_metadata, [0])]


def test_disk_recovery_failure_does_not_commit(monkeypatch):
    disk = _DiskUpdater((False, "injected disk failure"))
    updater = _Updater({0: [3]})

    with pytest.raises(RuntimeError, match="injected disk failure"):
        _run_recovery(
            monkeypatch, hbm_client=None, disk_updater=disk, updater=updater
        )

    assert updater.calls == [("update", False)]


@pytest.mark.parametrize("is_ep_joiner", [True, False])
def test_hbm_registry_is_collected_only_for_initial_cohort(monkeypatch, is_ep_joiner):
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(
        is_ep_joiner=is_ep_joiner,
        enable_elastic_hbm_expert_source=True,
    )
    runner.load_config = SimpleNamespace(weight_cache_socket="/tmp/weight-cache.sock")
    runner.ps = SimpleNamespace(moe_ep_rank=0, moe_ep_size=1)
    runner.model = object()

    calls = []
    monkeypatch.setattr(
        "sglang.srt.model_executor.model_runner.get_exec",
        lambda: SimpleNamespace(
            moe=SimpleNamespace(
                enable_elastic_expert_backup=False, elastic_ep_backend="mooncake"
            )
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.elastic_ep.daemon_hbm_expert_source.collect_daemon_hbm_source_registry",
        lambda **_kwargs: calls.append("collect") or DaemonHBMSourceRegistry({}),
    )
    monkeypatch.setattr(
        "sglang.srt.elastic_ep.daemon_hbm_expert_source.DaemonHBMExpertSourceClient",
        lambda **_kwargs: "hbm-client",
    )
    monkeypatch.setattr(
        "sglang.srt.distributed.parallel_state.get_world_group",
        lambda: SimpleNamespace(ranks=[0]),
    )
    monkeypatch.setattr(
        "sglang.srt.distributed.parallel_state.get_moe_ep_group",
        lambda: SimpleNamespace(ranks=[0]),
    )

    runner.maybe_init_daemon_hbm_source_client()

    assert calls == ([] if is_ep_joiner else ["collect"])
    assert runner.daemon_hbm_source_client == (None if is_ep_joiner else "hbm-client")
