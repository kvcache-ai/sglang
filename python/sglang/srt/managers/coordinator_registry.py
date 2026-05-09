"""Registry for request-level coordinator plugins (e.g. HiSparseCoordinator).

Plugin slot used by Scheduler / ModelRunner to construct optional coordinator
objects when a feature flag is enabled, without importing the coordinator
class from base code.

Plugins register themselves at import time. The DSV4 plugin (HiSparse) is
loaded via the DSV4 model entry-point and registers under name "hisparse".
"""

from typing import Any, Callable, Dict


_Factory = Callable[..., Any]
_COORDINATOR_FACTORIES: Dict[str, _Factory] = {}


def register_request_coordinator(name: str, factory: _Factory) -> None:
    """Register a coordinator factory by name.

    Args:
      name: lookup key used by Scheduler.create_coordinator(name, ...).
      factory: callable invoked with the kwargs Scheduler passes; returns
        the coordinator instance.
    """
    if name in _COORDINATOR_FACTORIES:
        return
    _COORDINATOR_FACTORIES[name] = factory


def create_coordinator(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build a coordinator by name, or raise if no plugin is registered."""
    factory = _COORDINATOR_FACTORIES.get(name)
    if factory is None:
        raise RuntimeError(
            f"No coordinator plugin registered for {name!r}. "
            f"Registered: {sorted(_COORDINATOR_FACTORIES)}. "
            "Hint: ensure the model that needs this feature has been imported "
            "(via sglang.srt.models.registry auto-discovery)."
        )
    return factory(*args, **kwargs)


def has_coordinator(name: str) -> bool:
    return name in _COORDINATOR_FACTORIES
