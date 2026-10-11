"""Torch fallbacks for deep_gemm's MQA indexer logits on GPUs deep_gemm
does not support (e.g. sm_120 / RTX PRO 6000 Blackwell workstation).

Formula (DeepSeek-V3.2 lightning indexer, see tilelang_kernel.fp8_index):
    logits[q, k] = k_scale[k] * sum_h( relu(q[q, h, :] . k[k, :]) * weights[q, h] )

deep_gemm.fp8_mqa_logits additionally restricts each row to the window
[ks[q], ke[q]) and may leave the rest uninitialized (clean_logits=False).
These fallbacks always clean: out-of-window entries are set to -inf so any
downstream top-k selection is safe.
"""

from typing import Tuple

import torch

# Per-q-row chunk so the [qc, H, K] intermediate stays a few GB at 256k keys.
_Q_CHUNK = 64


def torch_fp8_mqa_logits(
    q: torch.Tensor,  # [n_q, H, D] fp8_e4m3
    kv: Tuple[torch.Tensor, torch.Tensor],  # ([n_k, D] fp8_e4m3, [n_k] fp32 scale)
    weights: torch.Tensor,  # [n_q, H] fp32
    ks: torch.Tensor,  # [n_q] int32 inclusive start
    ke: torch.Tensor,  # [n_q] int32 exclusive end
    clean_logits: bool = True,
) -> torch.Tensor:
    k_fp8, k_scale = kv
    n_q, H, D = q.shape
    n_k = k_fp8.shape[0]
    q_bf = q.to(torch.bfloat16)
    k_bf = k_fp8.to(torch.bfloat16).transpose(0, 1)  # [D, n_k]
    w = weights.to(torch.float32)
    out = torch.empty(n_q, n_k, dtype=torch.float32, device=q.device)
    for s in range(0, n_q, _Q_CHUNK):
        e = min(s + _Q_CHUNK, n_q)
        # [qc, H, n_k]
        lg = torch.matmul(q_bf[s:e], k_bf).float()
        lg = torch.relu(lg)
        out[s:e] = torch.einsum("qhk,qh->qk", lg, w[s:e])
    out *= k_scale.to(torch.float32).unsqueeze(0)
    pos = torch.arange(n_k, device=q.device).unsqueeze(0)
    valid = (pos >= ks.to(torch.long).unsqueeze(1)) & (
        pos < ke.to(torch.long).unsqueeze(1)
    )
    out.masked_fill_(~valid, float("-inf"))
    return out


def torch_fp8_paged_mqa_logits(
    q: torch.Tensor,  # [B, next_n, H, D] fp8
    kv_cache: torch.Tensor,  # [num_pages, 64, 1, 132] VIEW of page-blocked buffer
    weights: torch.Tensor,  # [B*next_n, H] fp32
    seqlens: torch.Tensor,  # [B*next_n] int32
    block_tables: torch.Tensor,  # [B, max_pages] int32
    schedule_metadata,  # ignored
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Real index-K layout (memory_pool.py): per page of 64 tokens,
    buf[page, :64*128] = fp8 keys, buf[page, 64*128:] = 64 fp32 scales."""
    B, next_n, H, D = q.shape
    PAGE = 64
    rows = B * next_n
    device = q.device
    flat = kv_cache.reshape(kv_cache.shape[0], -1)  # [P, 8448] fp8-typed bytes
    k_bytes = PAGE * D
    q_bf = q.to(torch.bfloat16).view(rows, H, D)
    w = weights.to(torch.float32)
    out = torch.full((rows, max_seq_len), float("-inf"), dtype=torch.float32, device=device)
    n_pages = (max_seq_len + PAGE - 1) // PAGE
    for b in range(B):
        pages = block_tables[b, :n_pages].to(torch.long)
        pf = flat[pages]  # [n, 8448]
        keys = pf[:, :k_bytes].view(torch.float8_e4m3fn).to(torch.bfloat16).view(-1, D)
        scales = pf[:, k_bytes:].contiguous().view(torch.float32).view(-1)  # [n*64]
        kt = keys.transpose(0, 1)
        L = keys.shape[0]
        for i in range(next_n):
            r = b * next_n + i
            lg = torch.relu(torch.matmul(q_bf[r], kt).float())  # [H, L]
            row = torch.einsum("hl,h->l", lg, w[r]) * scales
            # capture-safe: mask by tensor comparison instead of .item()+slice
            pos = torch.arange(min(L, max_seq_len), device=device)
            valid = pos < seqlens[r].to(device)
            out[r, : pos.numel()] = torch.where(
                valid, row[: pos.numel()], torch.full_like(row[: pos.numel()], float("-inf"))
            )
    return out
