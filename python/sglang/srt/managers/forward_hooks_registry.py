"""Registry for forward / scheduler lifecycle hooks.

Plugin slot used by Scheduler / ScheduleBatch / output processor / runtime
checker to dispatch lifecycle events without referencing model-specific
coordinators (HiSparse, etc.) directly.

Plugins register a hook object (a class with optional `on_*` / `query_*`
methods) at module load time. The base scheduler / batch dispatches
events to all registered hooks; missing methods are no-ops.

The DSV4 plugin (HiSparseCoordinator) self-registers via
`register_forward_hook("hisparse", _HiSparseHookAdapter)` when
hisparse_coordinator.py is imported (triggered via deepseek_v4.py
side-effect).
"""

from typing import Any, Callable, Dict, List


_HOOKS: Dict[str, Any] = {}


def register_forward_hook(name: str, hook: Any) -> None:
    """Register a hook object. `hook` is any object — methods are looked up
    via getattr at dispatch time.

    Idempotent: re-registration with the same name is a no-op (matches the
    other registry modules in this package)."""
    if name in _HOOKS:
        return
    _HOOKS[name] = hook


def get_hook(name: str) -> Any:
    """Return the registered hook object for `name`, or None."""
    return _HOOKS.get(name)


def has_hook(name: str) -> bool:
    return name in _HOOKS


def dispatch(event: str, *args: Any, **kwargs: Any) -> None:
    """Fire `event` (e.g., "on_request_admit") to every registered hook
    that defines that method. Missing methods are no-ops."""
    for hook in _HOOKS.values():
        method = getattr(hook, event, None)
        if method is None:
            continue
        method(*args, **kwargs)


def query_any(event: str, *args: Any, **kwargs: Any) -> bool:
    """Return True if any registered hook's `event` method returns truthy.
    Used for boolean-OR queries like has_ongoing_staging()."""
    for hook in _HOOKS.values():
        method = getattr(hook, event, None)
        if method is None:
            continue
        if method(*args, **kwargs):
            return True
    return False


def query_collect(event: str, *args: Any, **kwargs: Any) -> List[Any]:
    """Concatenate the list returned by every registered hook's `event`.
    Used for queries like collect_ready_reqs() that aggregate from plugins."""
    out: List[Any] = []
    for hook in _HOOKS.values():
        method = getattr(hook, event, None)
        if method is None:
            continue
        result = method(*args, **kwargs)
        if result:
            out.extend(result)
    return out
