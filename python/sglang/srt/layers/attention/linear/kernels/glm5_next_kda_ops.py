"""GLM-5-Next-only KDA gate and recurrent decode operations.

The existing Kimi KDA kernels deliberately remain untouched.  GLM-5-Next uses
raw beta plus a bounded gate, while Kimi uses the unbounded softplus gate when
``lower_bound`` is absent.  Keeping the bounded kernels in this module avoids
changing Kimi's launch signature, autotune cache, TF32 settings, or padding
behavior.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


_MAX_TRITON_GRID_Z = 65535
_BT_AUTOTUNE = (32, 64, 128)
_NUM_WARPS_AUTOTUNE = (2, 4, 8, 16) if torch.version.hip else (4, 8, 16, 32)


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _next_power_of_2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def glm5_next_decode_grid(
    nk: int, nv: int, num_sequences: int, num_value_heads: int
) -> tuple[tuple[int, int, int], bool]:
    """Return a CUDA-valid grid without changing the normal launch shape."""

    if num_sequences * num_value_heads > _MAX_TRITON_GRID_Z:
        # The safe decode kernel supports NK == 1.  Split sequence and head
        # across Y/Z instead of overflowing CUDA's 65535-block Z dimension.
        if nk != 1:
            raise ValueError("GLM-5-Next split decode grid requires NK == 1")
        return (nv, num_sequences, num_value_heads), True
    return (nk, nv, num_sequences * num_value_heads), False


def trim_glm5_next_kda_padding(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Trim graph padding while accepting token-first or batch-first gates."""

    physical_num_tokens = mixed_qkv.shape[0]
    logical_num_tokens = int(query_start_loc[-1])
    if logical_num_tokens > physical_num_tokens:
        raise ValueError(
            "query_start_loc describes more GLM KDA tokens than mixed_qkv: "
            f"{logical_num_tokens} > {physical_num_tokens}"
        )
    if logical_num_tokens == physical_num_tokens:
        return mixed_qkv, raw_gate, raw_beta, physical_num_tokens

    def trim_token_axis(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.shape[0] == physical_num_tokens:
            return tensor[:logical_num_tokens]
        if tensor.ndim > 1 and tensor.shape[1] == physical_num_tokens:
            return tensor[:, :logical_num_tokens]
        raise ValueError(
            f"Cannot identify the token axis for {name} with shape "
            f"{tuple(tensor.shape)} and {physical_num_tokens} physical tokens"
        )

    return (
        mixed_qkv[:logical_num_tokens],
        trim_token_axis(raw_gate, "raw_gate"),
        trim_token_axis(raw_beta, "raw_beta"),
        physical_num_tokens,
    )


def restore_glm5_next_kda_padding(
    output: torch.Tensor, physical_num_tokens: int
) -> torch.Tensor:
    """Restore zero rows expected by a padded hybrid-attention batch."""

    logical_num_tokens = output.shape[1]
    if logical_num_tokens > physical_num_tokens:
        raise ValueError(
            "GLM KDA output has more logical than physical tokens: "
            f"{logical_num_tokens} > {physical_num_tokens}"
        )
    if logical_num_tokens == physical_num_tokens:
        return output
    padding = output.new_zeros(
        (output.shape[0], physical_num_tokens - logical_num_tokens)
        + tuple(output.shape[2:])
    )
    return torch.cat((output, padding), dim=1)


def _torch_safe_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    dt_bias: Optional[torch.Tensor],
    lower_bound: float,
) -> torch.Tensor:
    """CPU reference/fallback for the bounded GLM gate."""

    original_shape = raw_gate.shape[:-1]
    raw_gate = raw_gate.reshape(-1, raw_gate.shape[-1]).float()
    num_heads = A_log.numel()
    if num_heads * head_k_dim != raw_gate.shape[-1]:
        raise ValueError(
            "GLM KDA raw gate width must equal num_heads * head_k_dim: "
            f"{raw_gate.shape[-1]} != {num_heads} * {head_k_dim}"
        )
    gate = raw_gate.view(-1, num_heads, head_k_dim)
    if dt_bias is not None:
        gate = gate + dt_bias.float().reshape(1, num_heads, head_k_dim)
    scale = A_log.float().reshape(1, num_heads, 1).exp()
    gate = lower_bound * torch.sigmoid(scale * gate)
    return gate.view(*original_shape, num_heads, head_k_dim)


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=num_warps, num_stages=num_stages)
        for bt in _BT_AUTOTUNE
        for num_warps in _NUM_WARPS_AUTOTUNE
        for num_stages in (2, 3)
    ],
    key=["H", "D"],
)
@triton.jit
def _glm5_next_safe_gate_kernel(
    raw_gate,
    A_log,
    output,
    dt_bias,
    lower_bound,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    token_offset = i_t * BT

    offsets_d = tl.arange(0, BD)
    bias_mask = offsets_d < D
    gate_ptr = tl.make_block_ptr(
        base=raw_gate + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(token_offset, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    output_ptr = tl.make_block_ptr(
        base=output + i_h * D,
        shape=(T, D),
        strides=(H * D, 1),
        offsets=(token_offset, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    gate = tl.load(gate_ptr, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(dt_bias + i_h * D + offsets_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        gate += bias[None, :]

    scale = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    gate = lower_bound * tl.sigmoid(scale * gate)
    tl.store(output_ptr, gate, boundary_check=(0, 1))


def glm5_next_safe_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    head_k_dim: int,
    *,
    dt_bias: Optional[torch.Tensor],
    lower_bound: float,
) -> torch.Tensor:
    """Activate a raw GLM gate as ``lower_bound * sigmoid(exp(A)*x)``."""

    if lower_bound >= 0:
        raise ValueError(f"GLM KDA lower_bound must be negative, got {lower_bound}")
    if raw_gate.device.type == "cpu":
        return _torch_safe_gate(raw_gate, A_log, head_k_dim, dt_bias, lower_bound)

    original_shape = raw_gate.shape[:-1]
    raw_gate = raw_gate.reshape(-1, raw_gate.shape[-1])
    num_tokens, width = raw_gate.shape
    num_heads = A_log.numel()
    if num_heads * head_k_dim != width:
        raise ValueError(
            "GLM KDA raw gate width must equal num_heads * head_k_dim: "
            f"{width} != {num_heads} * {head_k_dim}"
        )
    output = torch.empty_like(raw_gate, dtype=torch.float32)

    def grid(meta):
        return (_cdiv(num_tokens, meta["BT"]), num_heads)

    _glm5_next_safe_gate_kernel[grid](
        raw_gate,
        A_log,
        output,
        dt_bias,
        lower_bound,
        num_tokens,
        num_heads,
        head_k_dim,
        BD=_next_power_of_2(head_k_dim),
        HAS_BIAS=dt_bias is not None,
    )
    return output.view(*original_shape, num_heads, head_k_dim)


@triton.jit(do_not_specialize=["T"])
def _glm5_next_safe_decode_kernel(
    A_log,
    raw_gate,
    dt_bias,
    lower_bound,
    q,
    k,
    v,
    raw_beta,
    output,
    state_source,
    state_indices,
    query_start_loc,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SPLIT_N_HV_GRID: tl.constexpr,
):
    if SPLIT_N_HV_GRID:
        i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_k = 0
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(query_start_loc + i_n).to(tl.int64),
            tl.load(query_start_loc + i_n + 1).to(tl.int64),
        )
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    offsets_k = i_k * BK + tl.arange(0, BK)
    offsets_v = i_v * BV + tl.arange(0, BV)
    mask_k = offsets_k < K
    mask_v = offsets_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    q_ptr = q + (bos * H + i_h) * K + offsets_k
    k_ptr = k + (bos * H + i_h) * K + offsets_k
    v_ptr = v + (bos * HV + i_hv) * V + offsets_v
    beta_ptr = raw_beta + bos * HV + i_hv
    output_ptr = output + ((i_k * all_tokens + bos) * HV + i_hv) * V + offsets_v
    gate_ptr = raw_gate + (bos * HV + i_hv) * K + offsets_k
    bias_ptr = dt_bias + i_hv * K + offsets_k
    gate_scale = tl.exp(tl.load(A_log + i_hv).to(tl.float32))

    state = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        state_index = tl.load(state_indices + i_n)
        if state_index >= 0:
            state_ptr = (
                state_source
                + state_index * HV * K * V
                + i_hv * K * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            state += tl.load(state_ptr, mask=mask_state, other=0).to(tl.float32)

    for _ in range(0, T):
        q_value = tl.load(q_ptr, mask=mask_k, other=0).to(tl.float32)
        k_value = tl.load(k_ptr, mask=mask_k, other=0).to(tl.float32)
        v_value = tl.load(v_ptr, mask=mask_v, other=0).to(tl.float32)
        beta = tl.load(beta_ptr).to(tl.float32)
        gate = tl.load(gate_ptr, mask=mask_k, other=0).to(tl.float32)
        bias = tl.load(bias_ptr, mask=mask_k, other=0).to(tl.float32)

        gate = lower_bound * tl.sigmoid(gate_scale * (gate + bias))
        beta = tl.sigmoid(beta)
        if USE_QK_L2NORM_IN_KERNEL:
            q_value /= tl.sqrt(tl.sum(q_value * q_value) + 1e-6)
            k_value /= tl.sqrt(tl.sum(k_value * k_value) + 1e-6)
        q_value *= scale

        state *= tl.exp(gate[:, None])
        v_value -= tl.sum(state * k_value[:, None], axis=0)
        v_value *= beta
        state += k_value[:, None] * v_value[None, :]
        result = tl.sum(state * q_value[:, None], axis=0)
        tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=mask_v)

        q_ptr += H * K
        k_ptr += H * K
        v_ptr += HV * V
        beta_ptr += HV
        output_ptr += HV * V
        gate_ptr += HV * K

    if USE_INITIAL_STATE:
        state_index = tl.load(state_indices + i_n)
        if state_index >= 0:
            state_ptr = (
                state_source
                + state_index * HV * K * V
                + i_hv * K * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            tl.store(
                state_ptr,
                state.to(state_ptr.dtype.element_ty),
                mask=mask_state,
            )


def _torch_safe_decode(
    *,
    A_log: torch.Tensor,
    raw_gate: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_beta: torch.Tensor,
    state_source: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    output = torch.empty_like(v, dtype=q.dtype)
    _, _, num_q_heads, head_k_dim = q.shape
    num_value_heads = v.shape[2]
    head_v_dim = v.shape[-1]
    group_size = num_value_heads // num_q_heads
    flat_A = A_log.float().reshape(-1)
    flat_bias = dt_bias.float().reshape(num_value_heads, head_k_dim)

    for sequence_id in range(query_start_loc.numel() - 1):
        bos = int(query_start_loc[sequence_id])
        eos = int(query_start_loc[sequence_id + 1])
        state_index = int(state_indices[sequence_id])
        if state_index >= 0:
            state = state_source[state_index].float().clone()
        else:
            state = torch.zeros(
                num_value_heads,
                head_k_dim,
                head_v_dim,
                dtype=torch.float32,
                device=q.device,
            )

        for token_id in range(bos, eos):
            for value_head in range(num_value_heads):
                query_head = value_head // group_size
                q_value = q[0, token_id, query_head].float()
                k_value = k[0, token_id, query_head].float()
                if use_qk_l2norm_in_kernel:
                    q_value = q_value / torch.sqrt(q_value.square().sum() + 1e-6)
                    k_value = k_value / torch.sqrt(k_value.square().sum() + 1e-6)
                q_value = q_value * scale

                gate = lower_bound * torch.sigmoid(
                    flat_A[value_head].exp()
                    * (
                        raw_gate[0, token_id, value_head].float()
                        + flat_bias[value_head]
                    )
                )
                beta = raw_beta[0, token_id, value_head].float().sigmoid()
                head_state = state[value_head]
                head_state *= gate.exp().unsqueeze(-1)
                value = v[0, token_id, value_head].float()
                value = (value - (head_state * k_value.unsqueeze(-1)).sum(0)) * beta
                head_state += k_value.unsqueeze(-1) * value.unsqueeze(0)
                output[0, token_id, value_head] = (
                    (head_state * q_value.unsqueeze(-1)).sum(0).to(output.dtype)
                )

        if state_index >= 0:
            state_source[state_index].copy_(state.to(state_source.dtype))
    return output


def glm5_next_safe_decode(
    *,
    A_log: torch.Tensor,
    raw_gate: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_beta: torch.Tensor,
    state_source: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """GLM bounded-gate decode; both gate and beta inputs are raw logits."""

    if lower_bound >= 0:
        raise ValueError(f"GLM KDA lower_bound must be negative, got {lower_bound}")
    batch, tokens, num_q_heads, head_k_dim = k.shape
    head_v_dim = v.shape[-1]
    num_value_heads = v.shape[2]
    num_sequences = batch if query_start_loc is None else query_start_loc.numel() - 1
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(triton.next_power_of_2(head_v_dim), 32)
    nk = triton.cdiv(head_k_dim, block_k)
    nv = triton.cdiv(head_v_dim, block_v)
    if nk != 1:
        raise ValueError("GLM KDA decode currently requires NK == 1")
    if scale is None:
        scale = head_k_dim**-0.5
    elif scale <= 0:
        raise ValueError("scale must be positive")

    if q.device.type == "cpu":
        return _torch_safe_decode(
            A_log=A_log,
            raw_gate=raw_gate,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            q=q,
            k=k,
            v=v,
            raw_beta=raw_beta,
            state_source=state_source,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    raw_gate = raw_gate.contiguous()
    raw_beta = raw_beta.contiguous()
    output = q.new_empty(nk, *v.shape)
    grid, split_grid = glm5_next_decode_grid(nk, nv, num_sequences, num_value_heads)
    _glm5_next_safe_decode_kernel[grid](
        A_log=A_log,
        raw_gate=raw_gate,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        q=q,
        k=k,
        v=v,
        raw_beta=raw_beta,
        output=output,
        state_source=state_source,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        scale=scale,
        T=tokens,
        B=batch,
        H=num_q_heads,
        HV=num_value_heads,
        K=head_k_dim,
        V=head_v_dim,
        BK=block_k,
        BV=block_v,
        USE_INITIAL_STATE=state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=query_start_loc is not None,
        SPLIT_N_HV_GRID=split_grid,
        num_warps=1,
        num_stages=3,
    )
    return output.squeeze(0)
