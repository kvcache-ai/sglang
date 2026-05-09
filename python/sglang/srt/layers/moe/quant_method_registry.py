"""Registry for MoE quant-method wrappers.

Plugin slot used by FusedMoE.__init__ to optionally wrap a base GPU
quant_method (Fp8MoEMethod, ModelOptNvFp4FusedMoEMethod, …) with a
model-specific wrapper (e.g. KTEPWrapperMethod for DeepSeek V4 Flash CPU/GPU
expert split).

Plugins register themselves at import time. The DSV4 plugin is pulled in
when sglang.srt.models.deepseek_v4 is auto-discovered by ModelRegistry —
no base file imports the wrapper directly.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
    from sglang.srt.server_args import ServerArgs


_Predicate = Callable[[Any, "ServerArgs"], Optional[Any]]
_Factory = Callable[[Any, "FusedMoEMethodBase", Any], "FusedMoEMethodBase"]

# Each entry: (priority, wrapper_id, predicate, factory). LOWER priority runs
# FIRST (i.e. wraps the innermost method), matching the original PR #38 layout
# where Phase 2 (mxfp4) wrapped before Phase 3 (kt_ep). Iteration is sorted by
# priority on each `maybe_wrap_moe_quant_method` call so registration order
# from import sequence does not silently change wrap order.
_QUANT_WRAPPERS: List[Tuple[int, str, _Predicate, _Factory]] = []


def register_moe_quant_wrapper(
    wrapper_id: str,
    predicate: _Predicate,
    factory: _Factory,
    priority: int = 100,
) -> None:
    """Register a wrapper.

    Args:
      wrapper_id: stable id used by `is_wrapped_method` for isinstance-style
        checks without importing the wrapper class.
      predicate: (layer, server_args) -> Optional[ctx]. Return None to skip,
        otherwise return an opaque ctx that's passed to factory.
      factory: (layer, gpu_method, ctx) -> wrapped quant_method.
      priority: lower numbers wrap first (i.e. innermost). For DSV4: mxfp4
        registers at priority 10 (Phase 2), kt_ep at priority 20 (Phase 3).
    """
    for _, existing_id, _, _ in _QUANT_WRAPPERS:
        if existing_id == wrapper_id:
            return
    _QUANT_WRAPPERS.append((priority, wrapper_id, predicate, factory))


def maybe_wrap_moe_quant_method(
    layer: Any, gpu_method: "FusedMoEMethodBase", server_args: "ServerArgs"
) -> "FusedMoEMethodBase":
    """Iterate predicates in priority order (lower first); chain-wrap with each
    that matches. For DSV4, the final method is
    KTEPWrapperMethod(DeepSeekMxfp4MoEMethod(gpu_method)) because mxfp4
    is registered at priority 10 and kt_ep at priority 20."""
    method = gpu_method
    for _priority, wrapper_id, predicate, factory in sorted(_QUANT_WRAPPERS):
        ctx = predicate(layer, server_args)
        if ctx is not None:
            method = factory(layer, method, ctx)
            if getattr(method, "_quant_wrapper_id", None) is None:
                method._quant_wrapper_id = wrapper_id
    return method


def is_wrapped_method(method: Any, wrapper_id: str) -> bool:
    """isinstance replacement that doesn't require importing the wrapper class."""
    return getattr(method, "_quant_wrapper_id", None) == wrapper_id
