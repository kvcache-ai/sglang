from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

BLOCK_SIZE_K = 64
INDEX_HEAD_DIM = 128
KPOOL_SCORE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Each fixed-width shard contributes its local top-k to a second exact top-k.
# The union is mathematically sufficient for the global result and, unlike the
# old shared-memory radix selector, has no candidate-capacity correctness cap.
KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE = 32768


def build_pooled_page_table_64(
    page_table_64: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Pack one logical pool page into the first token page of each page group.

    Uses advanced indexing (gather) rather than strided slicing so the result
    is always a freshly allocated row-major tensor. Strided slicing produces
    a view whose .contiguous() short-circuits for shape==(1, 1), leaving
    stride(-1) == pool_size and breaking downstream kernels that require
    stride(-1) == 1 (e.g. deep_gemm.fp8_paged_mqa_logits).
    """
    assert BLOCK_SIZE_K % pool_size == 0, (
        f"pool_size ({pool_size}) must divide page_size ({BLOCK_SIZE_K})"
    )
    idx = torch.arange(
        0, page_table_64.shape[-1], pool_size, device=page_table_64.device
    )
    return page_table_64[..., idx]


def gather_index_k_scale_prefix_into(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
    scale_out: torch.Tensor,
) -> None:
    assert buf.dtype == torch.uint8
    assert page_indices.dtype in (torch.int32, torch.int64)
    assert k_out.dtype == torch.uint8
    assert scale_out.dtype == torch.float32
    assert pool.page_size == BLOCK_SIZE_K
    assert k_out.shape[0] >= seq_len
    assert k_out.shape[1] == INDEX_HEAD_DIM
    assert scale_out.shape[0] >= seq_len
    assert buf.is_contiguous()
    assert page_indices.is_contiguous()
    assert k_out.is_contiguous()
    assert scale_out.is_contiguous()
    if seq_len == 0:
        return

    _gather_index_k_scale_prefix_into_kernel[(seq_len,)](
        buf,
        buf.view(torch.float32),
        page_indices,
        k_out,
        scale_out,
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=pool.page_size * INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


def gather_index_k_bf16_prefix_into(
    pool,
    buf: torch.Tensor,
    page_indices: torch.Tensor,
    seq_len: int,
    k_out: torch.Tensor,
) -> None:
    """Gather a paged all-BF16 GLM KPool prefix into bounded scratch."""

    assert buf.dtype == torch.bfloat16
    assert page_indices.dtype in (torch.int32, torch.int64)
    assert k_out.dtype == torch.bfloat16
    assert pool.page_size == BLOCK_SIZE_K
    assert k_out.shape[0] >= seq_len
    assert k_out.shape[1] == INDEX_HEAD_DIM
    assert buf.is_contiguous()
    assert page_indices.is_contiguous()
    assert k_out.is_contiguous()
    if seq_len == 0:
        return

    _gather_index_k_bf16_prefix_into_kernel[(seq_len,)](
        buf,
        page_indices,
        k_out,
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        num_warps=4,
    )


@triton.jit
def _gather_index_k_bf16_prefix_into_kernel(
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    page_idx = token_id // PAGE_SIZE
    token_offset_in_page = token_id % PAGE_SIZE
    page = tl.load(page_indices_ptr + page_idx)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    src = page * BUF_NUMEL_PER_PAGE + token_offset_in_page * HEAD_DIM + offs
    dst = token_id * HEAD_DIM + offs
    value = tl.load(buf_ptr + src, mask=mask, other=0.0)
    tl.store(k_out_ptr + dst, value, mask=mask)


@triton.jit
def _gather_index_k_scale_prefix_into_kernel(
    buf_u8_ptr,
    buf_fp32_ptr,
    page_indices_ptr,
    k_out_ptr,
    scale_out_ptr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    page_idx = token_id // PAGE_SIZE
    token_offset_in_page = token_id % PAGE_SIZE
    page = tl.load(page_indices_ptr + page_idx)

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    src_k_offsets = page * BUF_NUMEL_PER_PAGE + token_offset_in_page * HEAD_DIM + offs
    dst_k_offsets = token_id * HEAD_DIM + offs
    k = tl.load(buf_u8_ptr + src_k_offsets, mask=mask)
    tl.store(k_out_ptr + dst_k_offsets, k, mask=mask)

    src_s_offset = (
        page * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + token_offset_in_page
    )
    scale = tl.load(buf_fp32_ptr + src_s_offset)
    tl.store(scale_out_ptr + token_id, scale)


def kpool_build_ragged_layout(
    full_page_table: torch.Tensor,
    cu_pages_excl: torch.Tensor,
    ragged_pool_pages: torch.Tensor,
    cu_q_len_excl: torch.Tensor,
    ragged_q_len: torch.Tensor,
    pooled_seq_lens_expanded: torch.Tensor,
    slots_per_page: int,
    total_pool_pages: int,
    total_q: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one flat pooled-page table plus per-q row starts/ends.

    NSA stores 64 pooled entries per physical page. A pooled page therefore
    corresponds to every ``pool_size``-th 64-token page in the original real
    page table.
    """
    device = full_page_table.device
    n_rag = cu_pages_excl.shape[0]
    concat_page_table = torch.empty(
        (total_pool_pages,), dtype=full_page_table.dtype, device=device
    )
    q_ks = torch.empty((total_q,), dtype=torch.int32, device=device)
    q_ke = torch.empty((total_q,), dtype=torch.int32, device=device)
    if n_rag == 0:
        return concat_page_table, q_ks, q_ke

    max_pool_pages = full_page_table.shape[1]
    _kpool_build_ragged_layout_kernel[(n_rag,)](
        full_page_table,
        cu_pages_excl,
        ragged_pool_pages,
        cu_q_len_excl,
        ragged_q_len,
        pooled_seq_lens_expanded,
        concat_page_table,
        q_ks,
        q_ke,
        max_pool_pages,
        slots_per_page,
        pool_size,
        BLOCK_PAGE=128,
        BLOCK_Q=128,
    )
    return concat_page_table, q_ks, q_ke


@triton.jit
def _kpool_build_ragged_layout_kernel(
    full_page_table_ptr,
    cu_pages_excl_ptr,
    ragged_pool_pages_ptr,
    cu_q_len_excl_ptr,
    ragged_q_len_ptr,
    pooled_seq_lens_ptr,
    concat_page_table_ptr,
    q_ks_ptr,
    q_ke_ptr,
    MAX_POOL_PAGES,
    SLOTS_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    k = tl.program_id(0)
    page_start = tl.load(cu_pages_excl_ptr + k)
    n_pages = tl.load(ragged_pool_pages_ptr + k)
    q_start = tl.load(cu_q_len_excl_ptr + k)
    q_count = tl.load(ragged_q_len_ptr + k)
    ks_val = page_start * SLOTS_PER_PAGE

    for p_off in tl.range(0, BLOCK_PAGE * tl.cdiv(n_pages, BLOCK_PAGE), BLOCK_PAGE):
        p_offs = p_off + tl.arange(0, BLOCK_PAGE)
        p_mask = p_offs < n_pages
        source_cols = p_offs * POOL_SIZE
        pages = tl.load(
            full_page_table_ptr + k * MAX_POOL_PAGES + source_cols,
            mask=p_mask,
            other=0,
        )
        tl.store(concat_page_table_ptr + page_start + p_offs, pages, mask=p_mask)

    for q_off in tl.range(0, BLOCK_Q * tl.cdiv(q_count, BLOCK_Q), BLOCK_Q):
        q_offs = q_off + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < q_count
        plen = tl.load(pooled_seq_lens_ptr + q_start + q_offs, mask=q_mask, other=0)
        ke_val = tl.minimum(
            ks_val + plen,
            (page_start + n_pages) * SLOTS_PER_PAGE,
        )
        tl.store(
            q_ks_ptr + q_start + q_offs,
            tl.full([BLOCK_Q], ks_val, tl.int32),
            mask=q_mask,
        )
        tl.store(q_ke_ptr + q_start + q_offs, ke_val, mask=q_mask)


def compute_pooled_write_locs(
    page_table_64: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Map logical pooled-K ids to packed physical index-cache locations."""
    assert page_table_64.ndim == 1
    pool_ids = pool_ids.to(torch.int64)
    pool_page_group = torch.div(pool_ids, BLOCK_SIZE_K, rounding_mode="floor")
    token_page_row = pool_page_group * pool_size
    packed_page = page_table_64.index_select(0, token_page_row.to(torch.int64))
    return packed_page.to(torch.int64) * BLOCK_SIZE_K + torch.remainder(
        pool_ids, BLOCK_SIZE_K
    )


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pooled_groups_to_topk(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        assert page_table.shape[0] == group_ids.shape[0]
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_kpool_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after selected expanded-history tokens."""
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1
    assert seq_lens.shape[0] == topk_result.shape[0]
    assert pool_lens.shape[0] == topk_result.shape[0]

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    if page_table is None:
        page_table = topk_result
        has_page_table = False
        page_table_cols = 1
    else:
        assert page_table.ndim == 2
        has_page_table = True
        page_table_cols = page_table.shape[1]

    if topk_offsets is None:
        topk_offsets = seq_lens
        has_topk_offsets = False
    else:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        assert topk_offsets.ndim == 1
        has_topk_offsets = True

    block_cols = triton.next_power_of_2(out_cols)
    _append_kpool_tail_to_topk_kernel[(rows,)](
        topk_result,
        seq_lens,
        pool_lens,
        page_table,
        topk_offsets,
        out,
        topk_result.stride(0),
        topk_result.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        N_COLS=n_cols,
        OUT_COLS=out_cols,
        PAGE_TABLE_COLS=page_table_cols,
        POOL_SIZE=pool_size,
        HAS_PAGE_TABLE=has_page_table,
        HAS_TOPK_OFFSETS=has_topk_offsets,
        BLOCK_COLS=block_cols,
    )
    return out


def _exact_topk_from_pooled_history_logits(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    *,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Device-only exact hierarchical top-k over arbitrary row lengths.

    The shard loop depends only on the static logits width captured by CUDA
    graph.  Replay-varying row starts/lengths stay on device, so this routine
    contains no host synchronization or data-dependent Python control flow.
    Keeping the best ``K`` entries from every shard is exact: every global
    top-``K`` value must be in its shard's local top-``K`` set.
    """

    rows, cols = logits.shape
    group_topk = history_group_budget_for_topk(topk, pool_size)
    if row_starts is not None and row_starts.shape != group_lengths.shape:
        raise ValueError(
            "KPool row_starts must match group_lengths, got "
            f"{tuple(row_starts.shape)} and {tuple(group_lengths.shape)}"
        )

    starts = (
        torch.zeros_like(group_lengths, dtype=torch.int64)
        if row_starts is None
        else row_starts.to(torch.int64)
    ).clamp(min=0, max=cols)
    bounded_lengths = torch.minimum(
        group_lengths.to(torch.int64).clamp_min(0),
        torch.full_like(starts, cols) - starts,
    )

    if cols == 0:
        selected_groups = torch.zeros(
            (rows, group_topk), dtype=torch.int64, device=logits.device
        )
    else:
        positions = torch.arange(cols, dtype=torch.int64, device=logits.device)
        valid = (positions.unsqueeze(0) >= starts.unsqueeze(1)) & (
            positions.unsqueeze(0) < (starts + bounded_lengths).unsqueeze(1)
        )
        masked_logits = logits.masked_fill(~valid, float("-inf"))

        candidate_values = []
        candidate_indices = []
        for chunk_start in range(0, cols, KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE):
            chunk_end = min(chunk_start + KPOOL_HIERARCHICAL_TOPK_CHUNK_SIZE, cols)
            chunk_width = chunk_end - chunk_start
            local_k = min(group_topk, chunk_width)
            values, indices = torch.topk(
                masked_logits[:, chunk_start:chunk_end],
                k=local_k,
                dim=1,
                largest=True,
                sorted=False,
            )
            indices = indices.to(torch.int64) + chunk_start
            if local_k != group_topk:
                pad_width = group_topk - local_k
                values = torch.cat(
                    [
                        values,
                        torch.full(
                            (rows, pad_width),
                            float("-inf"),
                            dtype=values.dtype,
                            device=values.device,
                        ),
                    ],
                    dim=1,
                )
                indices = torch.cat(
                    [
                        indices,
                        torch.zeros(
                            (rows, pad_width),
                            dtype=torch.int64,
                            device=indices.device,
                        ),
                    ],
                    dim=1,
                )
            candidate_values.append(values)
            candidate_indices.append(indices)

        all_values = torch.cat(candidate_values, dim=1)
        all_indices = torch.cat(candidate_indices, dim=1)
        _, candidate_rank = torch.topk(
            all_values,
            k=group_topk,
            dim=1,
            largest=True,
            # Keep every finite candidate before the padded ``-inf`` entries.
            # ``group_valid`` below is a prefix mask when a short sequence has
            # fewer than ``group_topk`` groups, so this ordering is part of the
            # correctness contract rather than a presentation preference.
            sorted=True,
        )
        selected_groups = torch.gather(
            all_indices, 1, candidate_rank
        ) - starts.unsqueeze(1)

    valid_counts = torch.minimum(
        bounded_lengths,
        torch.full_like(bounded_lengths, group_topk),
    )

    rank = torch.arange(group_topk, device=logits.device, dtype=torch.int64)
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)

    effective_page_table = page_table
    if page_table_row_index is not None:
        if page_table is None:
            raise ValueError("page_table_row_index requires page_table")
        if page_table_row_index.shape != group_lengths.shape:
            raise ValueError(
                "KPool page_table_row_index must match group_lengths, got "
                f"{tuple(page_table_row_index.shape)} and "
                f"{tuple(group_lengths.shape)}"
            )
        effective_page_table = page_table.index_select(
            0, page_table_row_index.to(device=page_table.device, dtype=torch.int64)
        )

    result = expand_pooled_groups_to_topk(
        selected_groups,
        group_valid,
        topk=topk,
        pool_size=pool_size,
        page_table=effective_page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is not None:
        result = append_kpool_tail_to_topk(
            result,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=effective_page_table,
            topk_offsets=topk_offsets,
        )
    if out_rows is None or out_rows == rows:
        return result
    padded = torch.full(
        (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
    )
    padded[:rows] = result
    return padded


@triton.jit
def _append_kpool_tail_to_topk_kernel(
    topk_ptr,
    seq_lens_ptr,
    pool_lens_ptr,
    page_table_ptr,
    topk_offsets_ptr,
    out_ptr,
    topk_stride_0,
    topk_stride_1,
    page_table_stride_0,
    page_table_stride_1,
    out_stride_0,
    out_stride_1,
    N_COLS: tl.constexpr,
    OUT_COLS: tl.constexpr,
    PAGE_TABLE_COLS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
    HAS_TOPK_OFFSETS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < OUT_COLS

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * POOL_SIZE
    history_len = tl.minimum(tail_start, N_COLS)
    tail_count = seq_len % POOL_SIZE

    is_history = cols < history_len
    safe_history_cols = tl.minimum(cols, N_COLS - 1)
    history_value = tl.load(
        topk_ptr + row * topk_stride_0 + safe_history_cols * topk_stride_1,
        mask=mask & is_history,
        other=-1,
    )

    tail_offset = cols - history_len
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    tail_raw = tail_start + tail_offset
    tail_value = tail_raw
    if HAS_PAGE_TABLE:
        safe_tail = tl.minimum(tl.maximum(tail_raw, 0), PAGE_TABLE_COLS - 1)
        tail_value = tl.load(
            page_table_ptr
            + row * page_table_stride_0
            + safe_tail * page_table_stride_1,
            mask=mask & is_tail,
            other=-1,
        ).to(tl.int32)
    if HAS_TOPK_OFFSETS:
        offset = tl.load(topk_offsets_ptr + row).to(tl.int32)
        tail_value = tail_raw + offset

    value = tl.where(is_history, history_value, -1)
    value = tl.where(is_tail, tail_value, value)
    tl.store(out_ptr + row * out_stride_0 + cols * out_stride_1, value, mask=mask)


def topk_from_pooled_history_logits(
    logits: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    out_rows: int | None = None,
    page_table_row_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select full-pool groups, expand to tokens, and optionally append tail."""
    assert logits.ndim == 2
    assert group_lengths.ndim == 1
    assert logits.shape[0] == group_lengths.shape[0]
    assert topk > 0
    assert topk % pool_size == 0
    assert out_rows is None or out_rows >= logits.shape[0]
    assert page_table_row_index is None or page_table is not None

    _, cols = logits.shape
    group_topk = history_group_budget_for_topk(topk, pool_size)
    if topk_offsets is not None and topk_offsets.ndim == 2:
        assert topk_offsets.shape[1] == 1
        topk_offsets = topk_offsets.squeeze(1)

    if group_topk not in (128, 160, 192, 224, 256, 512, 2048):
        raise NotImplementedError(
            "index_kpool topk only supports pooled group_topk in "
            f"(128, 160, 192, 224, 256, 512, 2048), got {group_topk} "
            f"(topk={topk}, pool_size={pool_size})."
        )
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise NotImplementedError(
            "index_kpool topk requires CUDA float32 logits; PyTorch topk fallback "
            f"is disabled. Got device={logits.device}, dtype={logits.dtype}."
        )

    return _exact_topk_from_pooled_history_logits(
        logits,
        group_lengths,
        pool_size=pool_size,
        topk=topk,
        page_table=page_table,
        topk_offsets=topk_offsets,
        seq_lens=seq_lens,
        row_starts=row_starts,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )


def kpool_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
    return_compressed: bool = False,
    write_cache: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == INDEX_HEAD_DIM
    assert slot_k.dtype == torch.bfloat16
    assert slot_score.dtype in KPOOL_SCORE_DTYPES
    assert ape.dtype == torch.float32
    assert pool.page_size == BLOCK_SIZE_K
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert loc.dtype == torch.int64
    assert write_cache or return_compressed

    if buf.dtype == torch.bfloat16:
        return _kpool_softmax_rotate_write_cache_bf16(
            pool=pool,
            buf=buf,
            slot_k=slot_k,
            slot_score=slot_score,
            ape=ape,
            loc=loc,
            write_mask=write_mask,
            return_compressed=return_compressed,
            write_cache=write_cache,
        )
    assert buf.dtype == torch.uint8

    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        assert not return_compressed
        write_mask = write_mask.contiguous()
        has_write_mask = True

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, slot_k.shape[2]),
                    dtype=torch.float8_e4m3fn,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], slot_k.shape[2]),
            dtype=torch.float8_e4m3fn,
            device=slot_k.device,
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32
    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=slot_k.shape[2],
        S_OFFSET_NBYTES_IN_PAGE=pool.page_size * pool.index_head_dim,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(slot_k.shape[2]),
    )
    if return_compressed:
        return compressed_k, compressed_scale
    return None


def _kpool_softmax_rotate_write_cache_bf16(
    *,
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None,
    return_compressed: bool,
    write_cache: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """SM86 KPool writer with no FP8 quantization or scale sidecar."""

    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        assert not return_compressed
        write_mask = write_mask.contiguous()
        has_write_mask = True

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, slot_k.shape[2]),
                    dtype=torch.bfloat16,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    compressed_k = (
        torch.empty(
            (slot_k.shape[0], slot_k.shape[2]),
            dtype=torch.bfloat16,
            device=slot_k.device,
        )
        if return_compressed
        else buf
    )
    compressed_scale = (
        torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
        if return_compressed
        else buf
    )
    _kpool_softmax_rotate_write_cache_bf16_kernel[(slot_k.shape[0],)](
        buf,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=slot_k.shape[2],
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(slot_k.shape[2]),
        num_warps=4,
        num_stages=2,
    )
    if return_compressed:
        return compressed_k, compressed_scale
    return None


@triton.jit
def _kpool_softmax_rotate_write_cache_bf16_kernel(
    buf_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write
    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    acc = tl.zeros((BLOCK_D,), tl.float32)
    denom = tl.zeros((BLOCK_D,), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0
        ).to(tl.float32)
        prob = tl.exp(score - max_score)
        denom += prob
        key = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += key * prob

    compressed = (acc / denom).to(tl.bfloat16).to(tl.float32)
    compressed = _hadamard128(compressed).to(tl.bfloat16)
    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        page = loc // PAGE_SIZE
        token = loc % PAGE_SIZE
        out = page * BUF_NUMEL_PER_PAGE + token * HEAD_DIM + offs
        tl.store(buf_ptr + out, compressed, mask=mask)
    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            compressed,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, 1.0)


def kpool_decode_update_and_maybe_write_cache(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool = False,
) -> None:
    assert tail_k.ndim == 3
    assert tail_score.shape == tail_k.shape
    assert tail_k.shape[1] == pool.index_kpool + pool.tail_extra_slots
    assert tail_k.shape[2] == INDEX_HEAD_DIM
    assert key.ndim == 2 and key.shape[1] == INDEX_HEAD_DIM
    assert slot_score.shape == key.shape
    assert ape.shape == (pool.index_kpool, INDEX_HEAD_DIM)
    assert tail_k.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    assert tail_score.dtype in KPOOL_SCORE_DTYPES
    assert slot_score.dtype == tail_score.dtype
    assert ape.dtype == torch.float32
    assert pool.page_size == BLOCK_SIZE_K
    assert pool.index_head_dim == INDEX_HEAD_DIM
    assert tail_k.is_contiguous()
    assert tail_score.is_contiguous()

    batch = key.shape[0]
    if batch == 0:
        return

    if buf.dtype == torch.bfloat16:
        _kpool_decode_update_and_maybe_write_cache_bf16(
            pool=pool,
            buf=buf,
            tail_k=tail_k,
            tail_score=tail_score,
            key=key,
            slot_score=slot_score,
            ape=ape,
            block_tables=block_tables,
            req_pool_indices=req_pool_indices,
            positions=positions,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
        )
        return
    assert buf.dtype == torch.uint8

    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()

    assert req_pool_indices.shape[0] >= batch
    assert positions.shape[0] >= batch
    assert seq_lens.shape[0] >= batch
    assert out_cache_loc.shape[0] >= batch
    assert block_tables.ndim == 2
    assert block_tables.shape[0] >= batch

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    _kpool_decode_update_and_maybe_write_cache_kernel[(batch,)](
        buf_fp8,
        buf_fp32,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        REQ_POOL_SIZE=tail_k.shape[0],
        PAGE_SIZE=pool.page_size,
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=tail_k.shape[2],
        BLOCK_TABLE_COLS=block_tables.shape[1],
        S_OFFSET_NBYTES_IN_PAGE=pool.slots_per_page * pool.index_head_dim,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(tail_k.shape[2]),
        SLOTS_PER_PAGE=pool.slots_per_page,
    )


def _kpool_decode_update_and_maybe_write_cache_bf16(
    *,
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    key = key.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    req_pool_indices = req_pool_indices.contiguous()
    positions = positions.contiguous()
    seq_lens = seq_lens.contiguous()
    out_cache_loc = out_cache_loc.contiguous()
    batch = key.shape[0]

    _kpool_decode_update_and_maybe_write_cache_bf16_kernel[(batch,)](
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        tail_k.stride(0),
        tail_k.stride(1),
        tail_score.stride(0),
        tail_score.stride(1),
        key.stride(0),
        slot_score.stride(0),
        ape.stride(0),
        block_tables.stride(0),
        block_tables.stride(1),
        REQ_POOL_SIZE=tail_k.shape[0],
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=tail_k.shape[2],
        BLOCK_TABLE_COLS=block_tables.shape[1],
        BLOCK_D=triton.next_power_of_2(tail_k.shape[2]),
        SLOTS_PER_PAGE=pool.slots_per_page,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        max_score = tl.maximum(max_score, score)

    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)

    if ROUND_SCALE:
        log_val = tl.log2(absmax * fp8_max_inv)
        scale = tl.exp2(tl.ceil(log_val))
    else:
        scale = absmax * fp8_max_inv

    quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, fp8_min), fp8_max)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )

        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)
    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            quantized,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


@triton.jit
def _kpool_decode_update_and_maybe_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_k_ptr,
    tail_score_ptr,
    key_ptr,
    slot_score_ptr,
    ape_ptr,
    block_tables_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    REQ_POOL_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TABLE_COLS: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < REQ_POOL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), REQ_POOL_SIZE - 1)

    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)

    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % TAIL_SIZE

    key = tl.load(
        key_ptr + row * key_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    if pos_valid & (slot == POOL_SIZE - 1):
        pool_logical_start = safe_pos - slot
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            k_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + phys * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            k = tl.where(is_current, key, k_buf)
            acc += k * prob

        x = (acc / denom).to(tl.bfloat16).to(tl.float32)
        x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

        fp8_min = -448.0
        fp8_max = 448.0
        fp8_max_inv = 1.0 / fp8_max
        absmax = tl.max(tl.abs(x), axis=0)
        absmax = tl.maximum(absmax, 1e-4)

        if ROUND_SCALE:
            log_val = tl.log2(absmax * fp8_max_inv)
            scale = tl.exp2(tl.ceil(log_val))
        else:
            scale = absmax * fp8_max_inv

        quantized = x / scale
        quantized = tl.minimum(tl.maximum(quantized, fp8_min), fp8_max)

        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = pool_page_group * POOL_SIZE
        token_page_row = tl.minimum(tl.maximum(token_page_row, 0), BLOCK_TABLE_COLS - 1)
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1,
        )
        loc_page_index = packed_page.to(tl.int64)
        loc_token_offset_in_page = pool_id % SLOTS_PER_PAGE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )

        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale)

    tail_k_offset = req * tail_k_stride_0 + phys_slot * tail_k_stride_1 + offs
    tail_score_offset = (
        req * tail_score_stride_0 + phys_slot * tail_score_stride_1 + offs
    )
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


@triton.jit
def _kpool_decode_update_and_maybe_write_cache_bf16_kernel(
    buf_ptr,
    tail_k_ptr,
    tail_score_ptr,
    key_ptr,
    slot_score_ptr,
    ape_ptr,
    block_tables_ptr,
    req_pool_indices_ptr,
    positions_ptr,
    seq_lens_ptr,
    out_cache_loc_ptr,
    tail_k_stride_0,
    tail_k_stride_1,
    tail_score_stride_0,
    tail_score_stride_1,
    key_stride_0,
    slot_score_stride_0,
    ape_stride_0,
    block_tables_stride_0,
    block_tables_stride_1,
    REQ_POOL_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TABLE_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    req_raw = tl.load(req_pool_indices_ptr + row)
    req_valid = (req_raw >= 0) & (req_raw < REQ_POOL_SIZE)
    req = tl.minimum(tl.maximum(req_raw, 0), REQ_POOL_SIZE - 1)
    pos = tl.load(positions_ptr + row)
    safe_pos = tl.maximum(pos, 0)
    seq_len = tl.load(seq_lens_ptr + row)
    cache_loc = tl.load(out_cache_loc_ptr + row)
    pos_valid = req_valid & (cache_loc != 0) & (pos >= 0) & (pos < seq_len)
    slot = safe_pos % POOL_SIZE
    phys_slot = safe_pos % TAIL_SIZE

    key = tl.load(
        key_ptr + row * key_stride_0 + offs, mask=dim_mask, other=0.0
    ).to(tl.float32)
    score_current = tl.load(
        slot_score_ptr + row * slot_score_stride_0 + offs,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    if pos_valid & (slot == POOL_SIZE - 1):
        pool_logical_start = safe_pos - slot
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            max_score = tl.maximum(max_score, score)

        acc = tl.zeros((BLOCK_D,), tl.float32)
        denom = tl.zeros((BLOCK_D,), tl.float32)
        for pool_slot in tl.static_range(0, POOL_SIZE):
            is_current = pool_slot == slot
            phys = (pool_logical_start + pool_slot) % TAIL_SIZE
            score_buf = tl.load(
                tail_score_ptr
                + req * tail_score_stride_0
                + phys * tail_score_stride_1
                + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.where(is_current, score_current, score_buf)
            score += tl.load(
                ape_ptr + pool_slot * ape_stride_0 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            prob = tl.exp(score - max_score)
            denom += prob
            key_buf = tl.load(
                tail_k_ptr + req * tail_k_stride_0 + phys * tail_k_stride_1 + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.where(is_current, key, key_buf) * prob

        compressed = (acc / denom).to(tl.bfloat16).to(tl.float32)
        compressed = _hadamard128(compressed).to(tl.bfloat16)
        pool_id = safe_pos // POOL_SIZE
        pool_page_group = pool_id // SLOTS_PER_PAGE
        token_page_row = tl.minimum(
            tl.maximum(pool_page_group * POOL_SIZE, 0), BLOCK_TABLE_COLS - 1
        )
        packed_page = tl.load(
            block_tables_ptr
            + row * block_tables_stride_0
            + token_page_row * block_tables_stride_1
        ).to(tl.int64)
        token = pool_id % SLOTS_PER_PAGE
        out = packed_page * BUF_NUMEL_PER_PAGE + token * HEAD_DIM + offs
        tl.store(buf_ptr + out, compressed, mask=dim_mask)

    tail_k_offset = req * tail_k_stride_0 + phys_slot * tail_k_stride_1 + offs
    tail_score_offset = (
        req * tail_score_stride_0 + phys_slot * tail_score_stride_1 + offs
    )
    update_mask = dim_mask & pos_valid
    tl.store(tail_k_ptr + tail_k_offset, key, mask=update_mask)
    tl.store(tail_score_ptr + tail_score_offset, score_current, mask=update_mask)


@triton.jit
def _hadamard_quantize_fp8(acc, denom, ROUND_SCALE: tl.constexpr):
    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    fp8_max_inv = 1.0 / 448.0
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv

    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    return quantized, scale


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_WRITE_MASK:
        if not tl.load(write_mask_ptr + row):
            return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)
    tail_base = tl.load(tail_logical_base_ptr + row)

    m = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            phys = (tail_base + slot) % TAIL_SIZE
            off = req * tail_stride_0 + phys * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(tail_k_ptr + off, mask=mask, other=0.0).to(tl.float32)
        else:
            off = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + off, mask=mask, other=0.0).to(tl.float32)
            k = tl.load(chunk_k_ptr + off, mask=mask, other=0.0).to(tl.float32)

        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        new_m = tl.maximum(m, score)
        rescale = tl.exp(m - new_m)
        prob = tl.exp(score - new_m)
        denom = denom * rescale + prob
        acc = acc * rescale + k * prob
        m = new_m

    quantized, scale = _hadamard_quantize_fp8(acc, denom, ROUND_SCALE)

    loc = tl.load(loc_ptr + row)
    loc_page_index = loc // SLOTS_PER_PAGE
    loc_token_offset_in_page = loc % SLOTS_PER_PAGE
    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE + loc_token_offset_in_page * HEAD_DIM + offs
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
    tl.store(buf_fp32_ptr + out_s_offset, scale)


def kpool_assemble_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
) -> None:
    pool_size = pool.index_kpool
    n_pools = req_pool_idx.shape[0]
    if n_pools == 0:
        return

    if buf.dtype == torch.bfloat16:
        _kpool_assemble_softmax_rotate_write_cache_bf16(
            pool=pool,
            buf=buf,
            chunk_k=chunk_k,
            chunk_score=chunk_score,
            tail_k=tail_k,
            tail_score=tail_score,
            req_pool_idx=req_pool_idx,
            n_from_tail=n_from_tail,
            chunk_src_start=chunk_src_start,
            tail_logical_base=tail_logical_base,
            ape=ape,
            loc=loc,
            write_mask=write_mask,
        )
        return
    assert buf.dtype == torch.uint8

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)
    slots_per_page = pool.slots_per_page

    _kpool_assemble_softmax_rotate_write_cache_kernel[(n_pools,)](
        buf_fp8,
        buf_fp32,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=slots_per_page,
    )


def scatter_kpool_tail_updates(
    pool,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
) -> None:
    pool_size = pool.index_kpool
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    _scatter_kpool_tail_updates_kernel[(n_rows, pool_size)](
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        POOL_SIZE=pool_size,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
    )


def _kpool_assemble_softmax_rotate_write_cache_bf16(
    *,
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None,
) -> None:
    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=chunk_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    _kpool_assemble_softmax_rotate_write_cache_bf16_kernel[
        (req_pool_idx.shape[0],)
    ](
        buf,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        chunk_k.stride(0),
        tail_k.stride(0),
        tail_k.stride(1),
        ape.stride(0),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        POOL_SIZE=pool.index_kpool,
        TAIL_SIZE=tail_k.shape[1],
        HEAD_DIM=INDEX_HEAD_DIM,
        HAS_WRITE_MASK=has_write_mask,
        BLOCK_D=triton.next_power_of_2(INDEX_HEAD_DIM),
        SLOTS_PER_PAGE=pool.slots_per_page,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _kpool_assemble_softmax_rotate_write_cache_bf16_kernel(
    buf_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    ape_stride_0,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_WRITE_MASK:
        if not tl.load(write_mask_ptr + row):
            return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    n_tail = tl.load(n_from_tail_ptr + row)
    req = tl.load(req_pool_idx_ptr + row)
    chunk_src = tl.load(chunk_src_start_ptr + row)
    tail_base = tl.load(tail_logical_base_ptr + row)

    running_max = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)
    denom = tl.zeros((BLOCK_D,), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        if slot < n_tail:
            phys = (tail_base + slot) % TAIL_SIZE
            src = req * tail_stride_0 + phys * tail_stride_1 + offs
            score = tl.load(tail_score_ptr + src, mask=mask, other=0.0).to(
                tl.float32
            )
            key = tl.load(tail_k_ptr + src, mask=mask, other=0.0).to(tl.float32)
        else:
            src = (chunk_src + (slot - n_tail)) * chunk_stride_0 + offs
            score = tl.load(chunk_score_ptr + src, mask=mask, other=0.0).to(
                tl.float32
            )
            key = tl.load(chunk_k_ptr + src, mask=mask, other=0.0).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0
        ).to(tl.float32)
        next_max = tl.maximum(running_max, score)
        rescale = tl.exp(running_max - next_max)
        prob = tl.exp(score - next_max)
        denom = denom * rescale + prob
        acc = acc * rescale + key * prob
        running_max = next_max

    compressed = (acc / denom).to(tl.bfloat16).to(tl.float32)
    compressed = _hadamard128(compressed).to(tl.bfloat16)
    loc = tl.load(loc_ptr + row)
    page = loc // SLOTS_PER_PAGE
    token = loc % SLOTS_PER_PAGE
    out = page * BUF_NUMEL_PER_PAGE + token * HEAD_DIM + offs
    tl.store(buf_ptr + out, compressed, mask=mask)


@triton.jit
def _scatter_kpool_tail_updates_kernel(
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    dst_logical_start_ptr,
    chunk_src_start_ptr,
    n_write_ptr,
    chunk_stride_0,
    tail_stride_0,
    tail_stride_1,
    POOL_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)

    n_w = tl.load(n_write_ptr + row)
    if slot >= n_w:
        return

    req = tl.load(req_pool_idx_ptr + row)
    dst_logical_start = tl.load(dst_logical_start_ptr + row)
    src_off = tl.load(chunk_src_start_ptr + row) + slot

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    k = tl.load(chunk_k_ptr + src_off * chunk_stride_0 + offs, mask=mask)
    s = tl.load(chunk_score_ptr + src_off * chunk_stride_0 + offs, mask=mask)

    dst = (
        req * tail_stride_0
        + ((dst_logical_start + slot) % TAIL_SIZE) * tail_stride_1
        + offs
    )
    tl.store(tail_k_ptr + dst, k, mask=mask)
    tl.store(tail_score_ptr + dst, s, mask=mask)
