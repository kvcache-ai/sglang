"""Triton implementation of the DeepSeek-V3.2 indexer MQA logits for GPUs
without deep_gemm support (e.g. sm_120).

    out[q, k] = k_scale[k] * sum_h( relu(q[q, h, :] . k[k, :]) * w[q, h] )
restricted to k in [ks[q], ke[q]); out-of-window entries are -inf.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mqa_logits_kernel(
    q_ptr,        # [n_q, H, D] fp8e4nv
    k_ptr,        # [n_k, D] fp8e4nv
    ksc_ptr,      # [n_k] fp32
    w_ptr,        # [n_q, H] fp32
    ks_ptr,       # [n_q] int32
    ke_ptr,       # [n_q] int32
    out_ptr,      # [n_q, n_k] fp32
    n_q, n_k,
    H: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    q0 = pid_q * BQ
    k0 = pid_k * BK
    offs_q = q0 + tl.arange(0, BQ)
    offs_k = k0 + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    qm = offs_q < n_q
    km = offs_k < n_k

    # K tile [D, BK] loaded once per program
    k_tile = tl.load(
        k_ptr + offs_k[None, :] * D + offs_d[:, None],
        mask=km[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    acc = tl.zeros((BQ, BK), dtype=tl.float32)
    for h in range(H):
        qh = tl.load(
            q_ptr + offs_q[:, None] * (H * D) + h * D + offs_d[None, :],
            mask=qm[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        s = tl.dot(qh, k_tile)              # [BQ, BK] fp32
        s = tl.maximum(s, 0.0)
        wh = tl.load(w_ptr + offs_q * H + h, mask=qm, other=0.0)
        acc += s * wh[:, None]

    ksc = tl.load(ksc_ptr + offs_k, mask=km, other=0.0)
    acc *= ksc[None, :]

    ks = tl.load(ks_ptr + offs_q, mask=qm, other=0)
    ke = tl.load(ke_ptr + offs_q, mask=qm, other=0)
    in_win = (offs_k[None, :] >= ks[:, None]) & (offs_k[None, :] < ke[:, None])
    acc = tl.where(in_win, acc, float("-inf"))

    tl.store(
        out_ptr + offs_q[:, None] * n_k + offs_k[None, :],
        acc,
        mask=qm[:, None] & km[None, :],
    )


def triton_fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=True):
    k_fp8, k_scale = kv
    n_q, H, D = q.shape
    n_k = k_fp8.shape[0]
    out = torch.empty(n_q, n_k, dtype=torch.float32, device=q.device)
    BQ, BK = 16, 128
    grid = (triton.cdiv(n_q, BQ), triton.cdiv(n_k, BK))
    _mqa_logits_kernel[grid](
        q, k_fp8, k_scale.to(torch.float32), weights.to(torch.float32),
        ks.to(torch.int32), ke.to(torch.int32), out,
        n_q, n_k, H=H, D=D, BQ=BQ, BK=BK,
        num_warps=4, num_stages=2,
    )
    return out
