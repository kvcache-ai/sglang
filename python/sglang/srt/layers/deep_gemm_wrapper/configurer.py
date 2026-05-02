import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

logger = logging.getLogger(__name__)


# Capabilities where DeepGEMM's wgmma / tcgen05.mma kernels actually run.
# Hopper (SM_90a) + DC Blackwell (SM_100f, SM_103a). NOT consumer Blackwell
# (SM_120) which lacks both, NOT Ada (SM_89) / Ampere (SM_8x). Outside this
# set, importing DeepGEMM may succeed but invoking its kernels raises
# "Unsupported architecture" (verified on RTX 5090: deep_gemm.attention's
# get_paged_mqa_logits_metadata crashes during CUDA Graph capture).
DEEPGEMM_CAPS = {(9, 0), (10, 0), (10, 3)}


def _compute_enable_deep_gemm():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() not in DEEPGEMM_CAPS:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
