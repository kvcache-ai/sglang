"""Request-row lifecycle for GLM-5-Next's live KPool compression tail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch

from sglang.srt.managers.coordinator_registry import register_request_coordinator
from sglang.srt.managers.forward_hooks_registry import register_forward_hook

if TYPE_CHECKING:
    from sglang.srt.mem_cache.glm5_next_memory_pool import Glm5NextHybridKVPool


class Glm5NextKPoolCoordinator:
    """Own the boundary at which a request's live KPool tail is reusable.

    ``prepare_kpool_request`` must run after a request row is allocated and
    before its first prefill forward.  The existing SGLang
    ``on_request_admit`` event fires *after* prefill, so it intentionally does
    not clear state: doing so would erase the tail that prefill just produced.
    Finish and retract events occur before request-row release and are safe
    cleanup points.
    """

    def __init__(self, pool: "Glm5NextHybridKVPool") -> None:
        if not getattr(pool, "is_glm5_next_kpool", False):
            raise TypeError("Glm5NextKPoolCoordinator only accepts the exact GLM KPool")
        self.pool = pool
        self._prepared_rows: set[int] = set()

    @staticmethod
    def _as_cpu_rows(
        req_pool_indices: int | Iterable[int] | torch.Tensor,
    ) -> list[int]:
        if isinstance(req_pool_indices, int):
            return [req_pool_indices]
        if isinstance(req_pool_indices, torch.Tensor):
            return [int(row) for row in req_pool_indices.detach().cpu().reshape(-1)]
        return [int(row) for row in req_pool_indices]

    def prepare_kpool_request(
        self, req_pool_indices: int | Iterable[int] | torch.Tensor
    ) -> None:
        """Clear freshly allocated rows immediately before first prefill."""

        rows = self._as_cpu_rows(req_pool_indices)
        if not rows:
            return
        self.pool.prepare_kpool_request(rows)
        self._prepared_rows.update(rows)

    def _release_request(self, req) -> None:
        req_pool_idx = getattr(req, "req_pool_idx", None)
        if req_pool_idx is None:
            return
        row = int(req_pool_idx)
        self.pool.clear_compress_tail_rows(row)
        self._prepared_rows.discard(row)

    def on_request_admit(self, req) -> None:
        """No-op: the current admit hook is post-prefill, not pre-forward."""

        del req

    def on_request_finished(self, req) -> None:
        self._release_request(req)

    def on_request_retract(self, req) -> None:
        self._release_request(req)

    def is_prepared(self, req_pool_idx: int) -> bool:
        """Test/debug visibility; not used to skip mandatory row clearing."""

        return req_pool_idx in self._prepared_rows


class _Glm5NextKPoolHookAdapter:
    """Global hook proxy; inert until the exact GLM pool factory attaches."""

    _coordinator: Optional[Glm5NextKPoolCoordinator] = None

    @classmethod
    def attach(cls, coordinator: Optional[Glm5NextKPoolCoordinator]) -> None:
        cls._coordinator = coordinator

    def prepare_kpool_request(self, req_pool_indices) -> None:
        if self._coordinator is not None:
            self._coordinator.prepare_kpool_request(req_pool_indices)

    def on_request_admit(self, req) -> None:
        if self._coordinator is not None:
            self._coordinator.on_request_admit(req)

    def on_request_finished(self, req) -> None:
        if self._coordinator is not None:
            self._coordinator.on_request_finished(req)

    def on_request_retract(self, req) -> None:
        if self._coordinator is not None:
            self._coordinator.on_request_retract(req)


_GLM5_NEXT_KPOOL_HOOK = _Glm5NextKPoolHookAdapter()


def attach_glm5_next_kpool_lifecycle(
    pool: "Glm5NextHybridKVPool",
) -> Glm5NextKPoolCoordinator:
    coordinator = Glm5NextKPoolCoordinator(pool)
    _Glm5NextKPoolHookAdapter.attach(coordinator)
    return coordinator


def detach_glm5_next_kpool_lifecycle() -> None:
    """Detach process-global state for worker shutdown and isolation tests."""

    _Glm5NextKPoolHookAdapter.attach(None)


def get_glm5_next_kpool_coordinator() -> Optional[Glm5NextKPoolCoordinator]:
    return _Glm5NextKPoolHookAdapter._coordinator


register_request_coordinator("glm5_next_kpool", Glm5NextKPoolCoordinator)
register_forward_hook("glm5_next_kpool", _GLM5_NEXT_KPOOL_HOOK)
