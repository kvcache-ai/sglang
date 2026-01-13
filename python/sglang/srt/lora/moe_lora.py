# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
MoE LoRA Layer Implementation

This module provides the MoE LoRA computation that can be applied alongside
the FusedMoE layer.

IMPORTANT: The correct LoRA computation requires:
1. Separate gate and up LoRA A matrices (they are independent in PEFT)
2. Access to base MoE weights to compute intermediate activations

The computation flow for each token t, for each selected expert e:
1. Compute gate output: gate_out = W_gate[e] @ x + B_gate[e] @ A_gate[e] @ x
2. Compute up output: up_out = W_up[e] @ x + B_up[e] @ A_up[e] @ x
3. Apply activation: activated = silu(gate_out) * up_out
4. Compute down output: expert_out = W_down[e] @ activated + B_down[e] @ A_down[e] @ activated
5. Accumulate: output[t] += routing_weight[t, e] * expert_out

NOTE: gate and up are computed SEPARATELY because in PEFT LoRA, each linear
layer (gate_proj, up_proj) has its own independent lora_A and lora_B matrices.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def moe_lora_forward(
    hidden_states: torch.Tensor,          # (num_tokens, hidden_size)
    topk_ids: torch.Tensor,               # (num_tokens, top_k) expert indices
    topk_weights: torch.Tensor,           # (num_tokens, top_k) routing weights
    gate_a: torch.Tensor,                 # (max_loras, num_experts, max_rank, hidden_size)
    gate_b: torch.Tensor,                 # (max_loras, num_experts, intermediate, max_rank)
    up_a: torch.Tensor,                   # (max_loras, num_experts, max_rank, hidden_size)
    up_b: torch.Tensor,                   # (max_loras, num_experts, intermediate, max_rank)
    down_a: torch.Tensor,                 # (max_loras, num_experts, max_rank, intermediate)
    down_b: torch.Tensor,                 # (max_loras, num_experts, hidden_size, max_rank)
    weight_indices: torch.Tensor,         # (num_seqs,) adapter index per sequence
    seq_lens: torch.Tensor,               # (num_seqs,) length of each sequence
    lora_ranks: torch.Tensor,             # (max_loras,) rank per adapter
    scalings: torch.Tensor,               # (max_loras,) scaling factor per adapter
    base_output: Optional[torch.Tensor] = None,  # (num_tokens, hidden_size) for in-place add
    base_gate_up_weight: Optional[torch.Tensor] = None,  # (num_experts, inter*2, hidden)
    base_down_weight: Optional[torch.Tensor] = None,     # (num_experts, hidden, inter)
) -> torch.Tensor:
    """
    Compute MoE LoRA contribution (multi-adapter version).

    This correctly computes MoE output with LoRA by using base weights
    and computing gate and up LoRA contributions separately.
    """
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Build token-to-sequence mapping
    token_to_seq = _build_token_to_seq_mapping(seq_lens, num_tokens, device)

    # Check if base weights are available
    has_base_weights = base_gate_up_weight is not None and base_down_weight is not None

    if not has_base_weights:
        logger.warning("Base MoE weights not provided, using LoRA-only computation (may be incorrect)")

    # Initialize output
    output = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

    # Get intermediate size from gate_b shape
    inter_size = gate_b.shape[3] if gate_b.dim() == 5 else gate_b.shape[2]

    # Process each token
    for t in range(num_tokens):
        seq_idx = token_to_seq[t].item()
        adapter_idx = weight_indices[seq_idx].item()
        rank = lora_ranks[adapter_idx].item()
        scaling = scalings[adapter_idx].item()

        if rank == 0:
            # No LoRA for this token, use base output if available
            if base_output is not None:
                output[t] = base_output[t].float()
            continue

        x = hidden_states[t].float()  # (hidden_size,)
        token_out = torch.zeros(hidden_size, device=device, dtype=torch.float32)

        for k in range(top_k):
            expert_id = topk_ids[t, k].item()
            routing_weight = topk_weights[t, k].item()

            if routing_weight == 0:
                continue

            # Get LoRA weights for this expert
            ga = gate_a[adapter_idx, expert_id, :rank, :].float()  # (rank, hidden)
            gb = gate_b[adapter_idx, expert_id, :, :rank].float()  # (inter, rank)
            ua = up_a[adapter_idx, expert_id, :rank, :].float()    # (rank, hidden)
            ub = up_b[adapter_idx, expert_id, :, :rank].float()    # (inter, rank)
            da = down_a[adapter_idx, expert_id, :rank, :].float()  # (rank, inter)
            db = down_b[adapter_idx, expert_id, :, :rank].float()  # (hidden, rank)

            if has_base_weights:
                # Correct computation with base weights
                # Weight layout depends on backend:
                # - triton kernels: w13=(hidden, inter*2), w2=(inter, hidden)
                # - non-triton:     w13=(inter*2, hidden), w2=(hidden, inter)
                W_gate_up = base_gate_up_weight[expert_id].float()
                W_down = base_down_weight[expert_id].float()

                hidden_size = x.shape[0]

                # Detect layout and split gate_up correctly
                if W_gate_up.shape[0] == hidden_size:
                    # triton kernels layout: (hidden, inter*2)
                    W_gate = W_gate_up[:, :inter_size]   # (hidden, inter)
                    W_up = W_gate_up[:, inter_size:]     # (hidden, inter)
                    # Compute base outputs (need transpose for mv)
                    base_gate_out = torch.mv(W_gate.T, x)  # (inter, hidden) @ (hidden,)
                    base_up_out = torch.mv(W_up.T, x)
                else:
                    # Standard layout: (inter*2, hidden)
                    W_gate = W_gate_up[:inter_size, :]   # (inter, hidden)
                    W_up = W_gate_up[inter_size:, :]
                    # Compute base outputs
                    base_gate_out = torch.mv(W_gate, x)  # (inter, hidden) @ (hidden,)
                    base_up_out = torch.mv(W_up, x)

                # LoRA contributions
                lora_gate_out = torch.mv(gb, torch.mv(ga, x))  # (inter,)
                gate_out = base_gate_out + lora_gate_out * scaling

                lora_up_out = torch.mv(ub, torch.mv(ua, x))  # (inter,)
                up_out = base_up_out + lora_up_out * scaling

                # Activation
                activated = F.silu(gate_out) * up_out  # (inter,)

                # down with LoRA
                # Detect W_down layout: triton=(inter, hidden), standard=(hidden, inter)
                # torch.mv(M, v) computes M @ v, so we need shape (hidden, inter) @ (inter,) -> (hidden,)
                if W_down.shape[0] == inter_size:
                    # triton kernels layout: W_down is (inter, hidden), need transpose
                    base_down_out = torch.mv(W_down.T, activated)  # (hidden, inter) @ (inter,)
                else:
                    # Standard layout: W_down is (hidden, inter), use directly
                    base_down_out = torch.mv(W_down, activated)  # (hidden, inter) @ (inter,)

                lora_down_out = torch.mv(db, torch.mv(da, activated))  # (hidden,)
                expert_out = base_down_out + lora_down_out * scaling
            else:
                # Fallback: LoRA-only (incorrect)
                gate_out = torch.mv(gb, torch.mv(ga, x))
                up_out = torch.mv(ub, torch.mv(ua, x))
                activated = F.silu(gate_out) * up_out

                intermediate_b = torch.mv(da, activated)
                expert_out = torch.mv(db, intermediate_b) * scaling

            token_out += routing_weight * expert_out

        output[t] = token_out

    output = output.to(dtype)
    return output


def _build_token_to_seq_mapping(
    seq_lens: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a mapping from token index to sequence index."""
    token_to_seq = torch.zeros(num_tokens, dtype=torch.long, device=device)
    offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_len = seq_len.item()
        token_to_seq[offset:offset + seq_len] = seq_idx
        offset += seq_len
    return token_to_seq


def moe_lora_forward_batched(
    hidden_states: torch.Tensor,          # (num_tokens, hidden_size)
    topk_ids: torch.Tensor,               # (num_tokens, top_k)
    topk_weights: torch.Tensor,           # (num_tokens, top_k)
    gate_a: torch.Tensor,                 # (max_loras, num_experts, max_rank, hidden_size)
    gate_b: torch.Tensor,                 # (max_loras, num_experts, intermediate, max_rank)
    up_a: torch.Tensor,                   # (max_loras, num_experts, max_rank, hidden_size)
    up_b: torch.Tensor,                   # (max_loras, num_experts, intermediate, max_rank)
    down_a: torch.Tensor,                 # (max_loras, num_experts, max_rank, intermediate)
    down_b: torch.Tensor,                 # (max_loras, num_experts, hidden_size, max_rank)
    adapter_idx: int,                     # Single adapter for entire batch
    rank: int,                            # LoRA rank
    scaling: float,                       # Scaling factor
    base_output: Optional[torch.Tensor] = None,
    base_gate_up_weight: Optional[torch.Tensor] = None,  # (num_experts, inter*2, hidden)
    base_down_weight: Optional[torch.Tensor] = None,     # (num_experts, hidden, inter)
) -> torch.Tensor:
    """
    Batched MoE LoRA forward for single-adapter batch.

    This computes the MoE output with LoRA applied correctly:
    - gate and up LoRA are computed SEPARATELY (they have independent lora_A)
    - For each expert: output = (W_down + lora_down) @ silu((W_gate + lora_gate) @ x) * ((W_up + lora_up) @ x)
    """
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Debug logging (only once per call)
    if not hasattr(moe_lora_forward_batched, '_debug_logged'):
        moe_lora_forward_batched._debug_logged = True
        logger.info(f"[MoE LoRA DEBUG] Input shapes:")
        logger.info(f"  hidden_states: {hidden_states.shape}, dtype={hidden_states.dtype}")
        logger.info(f"  topk_ids: {topk_ids.shape}, topk_weights: {topk_weights.shape}")
        logger.info(f"  gate_a: {gate_a.shape}, gate_b: {gate_b.shape}")
        logger.info(f"  up_a: {up_a.shape}, up_b: {up_b.shape}")
        logger.info(f"  down_a: {down_a.shape}, down_b: {down_b.shape}")
        logger.info(f"  adapter_idx: {adapter_idx}, rank: {rank}, scaling: {scaling}")
        if base_gate_up_weight is not None:
            logger.info(f"  base_gate_up_weight: {base_gate_up_weight.shape}")
        if base_down_weight is not None:
            logger.info(f"  base_down_weight: {base_down_weight.shape}")
        # Check if LoRA weights are non-zero
        ga_sample = gate_a[adapter_idx, 0, :rank, :]
        logger.info(f"  gate_a[0] non-zero: {(ga_sample != 0).any()}, norm: {ga_sample.norm():.6f}")
        gb_sample = gate_b[adapter_idx, 0, :, :rank]
        logger.info(f"  gate_b[0] non-zero: {(gb_sample != 0).any()}, norm: {gb_sample.norm():.6f}")

    if rank == 0:
        if base_output is not None:
            return base_output
        return torch.zeros(num_tokens, hidden_size, device=device, dtype=dtype)

    # If no base weights provided, fall back to LoRA-only (incorrect but backwards compatible)
    if base_gate_up_weight is None or base_down_weight is None:
        logger.warning("Base MoE weights not provided, using LoRA-only computation (may be incorrect)")
        return _moe_lora_forward_lora_only(
            hidden_states, topk_ids, topk_weights,
            gate_a, gate_b, up_a, up_b, down_a, down_b,
            adapter_idx, rank, scaling, base_output
        )

    # DEBUG: Set this to True to return base_output without LoRA (for testing)
    DEBUG_DISABLE_MOE_LORA = False
    if DEBUG_DISABLE_MOE_LORA and base_output is not None:
        logger.warning("[DEBUG] MoE LoRA disabled, returning base output")
        return base_output

    # Get LoRA weights for this adapter
    ga = gate_a[adapter_idx, :, :rank, :]  # (num_experts, rank, hidden)
    gb = gate_b[adapter_idx, :, :, :rank]  # (num_experts, inter, rank)
    ua = up_a[adapter_idx, :, :rank, :]    # (num_experts, rank, hidden)
    ub = up_b[adapter_idx, :, :, :rank]    # (num_experts, inter, rank)
    da = down_a[adapter_idx, :, :rank, :]  # (num_experts, rank, inter)
    db = down_b[adapter_idx, :, :, :rank]  # (num_experts, hidden, rank)

    # Initialize output (we will compute the full MoE+LoRA output)
    output = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

    num_experts = ga.shape[0]
    # Get intermediate size from gate_b shape
    inter_size = gb.shape[1]

    debug_expert_count = 0
    for expert_id in range(num_experts):
        # Find tokens routed to this expert
        expert_mask = (topk_ids == expert_id)
        if not expert_mask.any():
            continue

        expert_weights = torch.where(
            expert_mask, topk_weights, torch.zeros_like(topk_weights)
        ).sum(dim=1)

        token_mask = expert_weights > 0
        if not token_mask.any():
            continue

        x = hidden_states[token_mask].float()  # (n_tokens, hidden)
        weights = expert_weights[token_mask]   # (n_tokens,)

        # Get base weights for this expert
        # Weight layout depends on backend:
        # - triton kernels: w13=(hidden, inter*2), w2=(inter, hidden)
        # - non-triton:     w13=(inter*2, hidden), w2=(hidden, inter)
        W_gate_up = base_gate_up_weight[expert_id].float()
        W_down = base_down_weight[expert_id].float()

        hidden_size = x.shape[1]

        # Detect layout and split gate_up correctly
        if W_gate_up.shape[0] == hidden_size:
            # triton kernels layout: (hidden, inter*2)
            W_gate = W_gate_up[:, :inter_size]   # (hidden, inter)
            W_up = W_gate_up[:, inter_size:]     # (hidden, inter)
            # Compute base outputs (no transpose needed)
            base_gate_out = torch.mm(x, W_gate)  # (n_tokens, hidden) @ (hidden, inter)
            base_up_out = torch.mm(x, W_up)
        else:
            # Standard layout: (inter*2, hidden)
            W_gate = W_gate_up[:inter_size, :]   # (inter, hidden)
            W_up = W_gate_up[inter_size:, :]
            # Compute base outputs (transpose needed)
            base_gate_out = torch.mm(x, W_gate.T)  # (n_tokens, hidden) @ (hidden, inter)
            base_up_out = torch.mm(x, W_up.T)

        # LoRA contributions for gate and up
        lora_gate_out = torch.mm(torch.mm(x, ga[expert_id].float().T), gb[expert_id].float().T)
        gate_out = base_gate_out + lora_gate_out * scaling

        lora_up_out = torch.mm(torch.mm(x, ua[expert_id].float().T), ub[expert_id].float().T)
        up_out = base_up_out + lora_up_out * scaling

        # Activation
        activated = F.silu(gate_out) * up_out  # (n_tokens, inter)

        # Compute down with LoRA
        # Detect W_down layout: triton=(inter, hidden), standard=(hidden, inter)
        if W_down.shape[0] == inter_size:
            # triton kernels layout: (inter, hidden)
            base_down_out = torch.mm(activated, W_down)  # (n_tokens, inter) @ (inter, hidden)
        else:
            # Standard layout: (hidden, inter)
            base_down_out = torch.mm(activated, W_down.T)  # (n_tokens, inter) @ (inter, hidden)

        # LoRA contribution for down
        lora_down_out = torch.mm(torch.mm(activated, da[expert_id].float().T), db[expert_id].float().T)
        expert_out = base_down_out + lora_down_out * scaling

        # Debug logging for first few experts
        if debug_expert_count < 2:
            debug_expert_count += 1
            logger.info(f"[MoE LoRA DEBUG] Expert {expert_id}:")
            logger.info(f"  x: shape={x.shape}, norm={x.norm():.4f}")
            logger.info(f"  W_gate_up: shape={W_gate_up.shape}, W_down: shape={W_down.shape}")
            logger.info(f"  W_gate: shape={W_gate.shape}, W_up: shape={W_up.shape}")
            logger.info(f"  ga[{expert_id}]: shape={ga[expert_id].shape}, norm={ga[expert_id].norm():.6f}")
            logger.info(f"  gb[{expert_id}]: shape={gb[expert_id].shape}, norm={gb[expert_id].norm():.6f}")
            logger.info(f"  base_gate_out: norm={base_gate_out.norm():.4f}")
            logger.info(f"  lora_gate_out: norm={lora_gate_out.norm():.6f}, scaled={lora_gate_out.norm() * scaling:.6f}")
            logger.info(f"  base_up_out: norm={base_up_out.norm():.4f}")
            logger.info(f"  lora_up_out: norm={lora_up_out.norm():.6f}, scaled={lora_up_out.norm() * scaling:.6f}")
            logger.info(f"  activated: norm={activated.norm():.4f}")
            logger.info(f"  base_down_out: norm={base_down_out.norm():.4f}")
            logger.info(f"  lora_down_out: norm={lora_down_out.norm():.6f}, scaled={lora_down_out.norm() * scaling:.6f}")
            logger.info(f"  expert_out: norm={expert_out.norm():.4f}")
            logger.info(f"  routing_weight: {weights[0].item():.4f}")

        # Accumulate with routing weights
        output[token_mask] += expert_out * weights.unsqueeze(1)

    # Debug: compare with base_output
    if base_output is not None and not hasattr(moe_lora_forward_batched, '_output_logged'):
        moe_lora_forward_batched._output_logged = True
        logger.info(f"[MoE LoRA DEBUG] Output comparison:")
        logger.info(f"  base_output (FusedMoE): norm={base_output.float().norm():.4f}")
        logger.info(f"  our output (with LoRA): norm={output.norm():.4f}")
        diff = (output - base_output.float()).norm()
        logger.info(f"  difference norm: {diff:.6f}")
        # If diff is very large, it might indicate a computation error
        if diff > 100 * base_output.float().norm():
            logger.warning(f"  WARNING: Output difference is very large! Check computation.")

    # Convert back to input dtype
    output = output.to(dtype)

    return output


def _moe_lora_forward_lora_only(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    down_a: torch.Tensor,
    down_b: torch.Tensor,
    adapter_idx: int,
    rank: int,
    scaling: float,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fallback LoRA-only computation (incorrect but backwards compatible)."""
    num_tokens, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    ga = gate_a[adapter_idx, :, :rank, :]
    gb = gate_b[adapter_idx, :, :, :rank]
    ua = up_a[adapter_idx, :, :rank, :]
    ub = up_b[adapter_idx, :, :, :rank]
    da = down_a[adapter_idx, :, :rank, :]
    db = down_b[adapter_idx, :, :, :rank]

    delta = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)
    num_experts = ga.shape[0]

    for expert_id in range(num_experts):
        expert_mask = (topk_ids == expert_id)
        if not expert_mask.any():
            continue

        expert_weights = torch.where(
            expert_mask, topk_weights, torch.zeros_like(topk_weights)
        ).sum(dim=1)

        token_mask = expert_weights > 0
        if not token_mask.any():
            continue

        x = hidden_states[token_mask].float()
        weights = expert_weights[token_mask]

        # Compute gate and up separately
        gate_out = torch.mm(torch.mm(x, ga[expert_id].float().T), gb[expert_id].float().T)
        up_out = torch.mm(torch.mm(x, ua[expert_id].float().T), ub[expert_id].float().T)

        activated = F.silu(gate_out) * up_out

        intermediate_b = torch.mm(activated, da[expert_id].float().T)
        expert_out = torch.mm(intermediate_b, db[expert_id].float().T)

        delta[token_mask] += expert_out * weights.unsqueeze(1) * scaling

    delta = delta.to(dtype)

    if base_output is not None:
        base_output.add_(delta)
        return base_output

    return delta


class MoELoRALayer(nn.Module):
    """
    MoE LoRA layer wrapper that integrates with FusedMoE.

    This layer computes the LoRA contribution for MoE experts and adds
    it to the base MoE output.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        max_loras: int,
        max_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.max_loras = max_loras
        self.max_rank = max_rank
        self.dtype = dtype
        self.device = device

        # Buffers will be set by memory pool (now separate for gate and up)
        self.gate_a: Optional[torch.Tensor] = None
        self.gate_b: Optional[torch.Tensor] = None
        self.up_a: Optional[torch.Tensor] = None
        self.up_b: Optional[torch.Tensor] = None
        self.down_a: Optional[torch.Tensor] = None
        self.down_b: Optional[torch.Tensor] = None

        # Batch info will be set before forward
        self.weight_indices: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.lora_ranks: Optional[torch.Tensor] = None
        self.scalings: Optional[torch.Tensor] = None

        self.set_lora = False

    def set_lora_info(
        self,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        down_a: torch.Tensor,
        down_b: torch.Tensor,
        weight_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        lora_ranks: torch.Tensor,
        scalings: torch.Tensor,
    ):
        """Set LoRA buffers and batch info before forward pass."""
        self.gate_a = gate_a
        self.gate_b = gate_b
        self.up_a = up_a
        self.up_b = up_b
        self.down_a = down_a
        self.down_b = down_b
        self.weight_indices = weight_indices
        self.seq_lens = seq_lens
        self.lora_ranks = lora_ranks
        self.scalings = scalings
        self.set_lora = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply MoE LoRA to base MoE output.

        Args:
            hidden_states: Input to MoE layer
            topk_ids: Expert indices from router
            topk_weights: Routing weights from router
            base_output: Output from base MoE computation

        Returns:
            base_output with LoRA delta added
        """
        if not self.set_lora:
            return base_output

        # Check if single adapter (common case)
        unique_adapters = self.weight_indices.unique()

        if len(unique_adapters) == 1:
            # Optimized path for single adapter
            adapter_idx = unique_adapters[0].item()
            rank = self.lora_ranks[adapter_idx].item()
            scaling = self.scalings[adapter_idx].item()

            return moe_lora_forward_batched(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_a,
                self.gate_b,
                self.up_a,
                self.up_b,
                self.down_a,
                self.down_b,
                adapter_idx,
                rank,
                scaling,
                base_output,
            )
        else:
            # Multi-adapter path
            return moe_lora_forward(
                hidden_states,
                topk_ids,
                topk_weights,
                self.gate_a,
                self.gate_b,
                self.up_a,
                self.up_b,
                self.down_a,
                self.down_b,
                self.weight_indices,
                self.seq_lens,
                self.lora_ranks,
                self.scalings,
                base_output,
            )
