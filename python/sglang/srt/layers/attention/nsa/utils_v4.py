"""DSV4-Flash extensions to layers/attention/nsa/utils.py.

Holds debug helpers + the triton round-robin rerange kernel that PR #38
inlined into nsa/utils.py. Kept separate so non-DSV4 paths don't import
them.

Side-effect imports happen via models/deepseek_v4.py (which imports
this module), so the helpers are available whenever DSV4 archs are
enabled.
"""

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_attention_tp_rank,
    get_attention_tp_size,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def assert_cp_pure_extend(forward_batch: "ForwardBatch") -> None:
    """Assert SGLANG_DEBUG_HACK_CP_ASSERT_PURE_EXTEND invariants.

    Called from nsa.utils.nsa_use_prefill_cp behind the env-var gate.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = forward_batch.forward_mode
    assert mode == ForwardMode.EXTEND, (
        f"SGLANG_DEBUG_HACK_CP_ASSERT_PURE_EXTEND: expected ForwardMode.EXTEND, got {mode}. "
        "CP round-robin may be silently enabled on MIXED batches."
    )

    extend_lens = list(forward_batch.extend_seq_lens_cpu)
    seq_lens = list(forward_batch.seq_lens_cpu.tolist())
    assert len(extend_lens) == len(
        seq_lens
    ), f"extend_seq_lens_cpu ({len(extend_lens)}) != seq_lens_cpu ({len(seq_lens)})"
    mismatched = [
        (i, e, s) for i, (e, s) in enumerate(zip(extend_lens, seq_lens)) if e != s
    ]
    assert not mismatched, (
        f"SGLANG_DEBUG_HACK_CP_ASSERT_PURE_EXTEND: found chunked-prefill continuation "
        f"(extend_seq_lens != seq_lens) at {mismatched[:5]}{'...' if len(mismatched) > 5 else ''}. "
        "A request has prior KV cache; CP round-robin may have domain mismatch."
    )


def assert_tensor_identical_across_cp_ranks(
    t: torch.Tensor, tag: str, forward_batch
) -> None:
    """Cross-CP-rank tensor consistency check used by DSV4-Flash compressor."""
    from sglang.srt.layers.attention.nsa.utils import (
        is_nsa_enable_prefill_cp,
        nsa_use_prefill_cp,
    )

    if not (is_nsa_enable_prefill_cp() and nsa_use_prefill_cp(forward_batch)):
        return
    cp_size = get_attention_tp_size()
    if cp_size <= 1:
        return

    t_contig = t.contiguous()
    gathered = t_contig.new_empty(t_contig.shape[0] * cp_size, *t_contig.shape[1:])
    attn_tp_all_gather_into_tensor(gathered, t_contig)
    chunks = gathered.view(cp_size, *t_contig.shape)
    rank0 = chunks[0]
    for r in range(1, cp_size):
        if torch.equal(rank0, chunks[r]):
            continue
        rank0_f = rank0.float()
        chunks_r_f = chunks[r].float()
        both_nan = torch.isnan(rank0_f) & torch.isnan(chunks_r_f)
        diff = (rank0_f - chunks_r_f).abs()
        diff = torch.where(both_nan, torch.zeros_like(diff), diff)
        if torch.equal(diff, torch.zeros_like(diff)):
            continue
        raise AssertionError(
            f"[CP rank consistency] {tag}: rank {r} disagrees with rank 0. "
            f"max_abs_diff={diff.max().item():.3e}, "
            f"mean_abs_diff={diff.mean().item():.3e}, "
            f"shape={tuple(t_contig.shape)}, dtype={t_contig.dtype}, "
            f"my_rank={get_attention_tp_rank()}"
        )


class CpRoundRobinRerange:
    """Triton round-robin rerange kernel for CP all-gather output.

    Used by nsa.utils.cp_all_gather_rerange_output behind the
    SGLANG_OPT_CP_REARRANGE_TRITON env-var gate.
    """

    @classmethod
    def execute(cls, gathered: torch.Tensor, cp_size: int) -> torch.Tensor:
        return cls.triton(gathered, cp_size)

    @classmethod
    def vanilla(cls, gathered: torch.Tensor, cp_size: int) -> torch.Tensor:
        out_shape = gathered.shape
        return (
            gathered.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )

    @classmethod
    def triton(cls, gathered: torch.Tensor, cp_size: int) -> torch.Tensor:
        assert (
            gathered.is_cuda
        ), f"gathered must be on CUDA, got device={gathered.device}"
        assert gathered.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ), f"unsupported dtype {gathered.dtype}"
        assert (
            gathered.ndim >= 1
        ), f"gathered.ndim must be >=1, got shape={tuple(gathered.shape)}"
        assert (
            gathered.is_contiguous()
        ), f"gathered must be contiguous, got strides={gathered.stride()} shape={tuple(gathered.shape)}"
        assert (
            isinstance(cp_size, int) and cp_size >= 1
        ), f"cp_size must be positive int, got {cp_size!r}"
        total_rows = gathered.shape[0]
        assert (
            total_rows % cp_size == 0
        ), f"total_rows={total_rows} not divisible by cp_size={cp_size}"
        per_rank_len = total_rows // cp_size

        out = torch.empty_like(gathered)
        if total_rows == 0 or gathered.numel() == 0:
            return out
        view_in = gathered.reshape(total_rows, -1)
        view_out = out.view(total_rows, -1)
        hidden = view_in.shape[1]

        BLOCK_H = 1024
        grid = (total_rows, triton.cdiv(hidden, BLOCK_H))
        _cp_round_robin_rerange_kernel[grid](
            view_in,
            view_out,
            per_rank_len,
            hidden,
            cp_size=cp_size,
            BLOCK_H=BLOCK_H,
        )
        return out


@triton.jit
def _cp_round_robin_rerange_kernel(
    in_ptr,
    out_ptr,
    per_rank_len,
    hidden,
    cp_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    rank = row % cp_size
    local = row // cp_size
    src_row = rank * per_rank_len + local
    offs = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden
    x = tl.load(in_ptr + src_row * hidden + offs, mask=mask)
    tl.store(out_ptr + row * hidden + offs, x, mask=mask)
