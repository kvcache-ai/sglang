#!/usr/bin/env python3
"""Offline-compile GLM-5-Next's SM86/SM89 Triton kernel matrix.

This does not require a GPU.  It validates that the installed Triton toolchain
can lower every model-local consumer-GPU kernel to the requested CUDA target.
It is intentionally separate from numerical and end-to-end acceptance, which
must run on real hardware.  Pass ``--resource-report`` to force recompilation
and expose ptxas register, stack, and spill counts for static-config review.

After this passes, run the consumer-GPU unit file on the target host, then use
``glm5_next_session_c_benchmark.py collect --decode-input 32768
--decode-output 256`` against a CUDA-Graph-enabled server.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile(
    *,
    label: str,
    capability: int,
    kernel: Any,
    signature: dict[str, str],
    constants: dict[str, Any],
    num_warps: int,
    num_stages: int,
) -> None:
    target = GPUTarget("cuda", capability, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    triton.compile(
        ASTSource(kernel, signature, constants),
        target=target,
        options=options.__dict__,
    )
    print(f"PASS {label}")


def _compile_indexer(indexer: ModuleType, capability: int) -> None:
    is_sm86 = capability == 86
    value_type = "*bf16" if is_sm86 else "*fp8e4nv"
    block_k, num_warps, num_stages = indexer.glm5_next_indexer_launch_config(
        (capability // 10, capability % 10)
    )
    flat_signature = {
        "q_ptr": value_type,
        "k_ptr": value_type,
        "k_scale_ptr": "*fp32",
        "weights_ptr": "*fp32",
        "ks_ptr": "*i32",
        "ke_ptr": "*i32",
        "logits_ptr": "*fp32",
        "num_keys": "i32",
    }
    flat_constants = {
        "stride_q_row": 4096,
        "stride_q_head": 128,
        "stride_q_dim": 1,
        "stride_k_row": 128,
        "stride_k_dim": 1,
        "stride_weights_row": 32,
        "stride_weights_head": 1,
        "stride_logits_row": 32768,
        "NUM_HEADS": 32,
        "HEAD_DIM": 128,
        "USE_K_SCALE": not is_sm86,
        "BLOCK_K": block_k,
    }
    _compile(
        label=f"indexer-flat-sm{capability}",
        capability=capability,
        kernel=indexer._glm5_next_flat_mqa_logits_kernel,
        signature=flat_signature,
        constants=flat_constants,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    paged_signature = {
        "q_ptr": value_type,
        "cache_ptr": value_type,
        "cache_scale_ptr": "*bf16" if is_sm86 else "*fp32",
        "weights_ptr": "*fp32",
        "seq_lens_ptr": "*i32",
        "page_table_ptr": "*i32",
        "logits_ptr": "*fp32",
        "max_seq_len": "i32",
        "num_cache_pages": "i32",
    }
    paged_constants = {
        "stride_q_row": 4096,
        "stride_q_head": 128,
        "stride_q_dim": 1,
        "stride_cache_page": 8192 if is_sm86 else 8448,
        "stride_weights_row": 32,
        "stride_weights_head": 1,
        "stride_page_table_row": 4096,
        "stride_page_table_col": 1,
        "stride_logits_row": 262144,
        "SCALE_OFFSET": 0 if is_sm86 else 2048,
        "SCALE_PAGE_STRIDE": 0 if is_sm86 else 2112,
        "NUM_HEADS": 32,
        "HEAD_DIM": 128,
        "PAGE_SIZE": 64,
        "USE_K_SCALE": not is_sm86,
        "BLOCK_K": block_k,
    }
    _compile(
        label=f"indexer-paged-sm{capability}",
        capability=capability,
        kernel=indexer._glm5_next_paged_mqa_logits_kernel,
        signature=paged_signature,
        constants=paged_constants,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _compile_sparse_mla(sparse: ModuleType, capability: int) -> None:
    is_sm86 = capability == 86
    block_t, num_warps, num_stages = sparse.glm5_next_sparse_mla_launch_config(
        (capability // 10, capability % 10)
    )
    signature = {
        "query_ptr": "*bf16",
        "kv_ptr": "*bf16" if is_sm86 else "*fp8e4nv",
        "kv_scale_ptr": "*bf16" if is_sm86 else "*fp32",
        "indices_ptr": "*i32",
        "output_ptr": "*bf16",
        "sm_scale": "fp32",
        "num_kv_tokens": "i64",
        "topk": "i32",
    }
    constants = {
        "stride_query_row": 16384,
        "stride_query_head": 512,
        "stride_query_dim": 1,
        "stride_kv_row": 512,
        "stride_kv_dim": 1,
        "stride_kv_scale_row": 512 if is_sm86 else 4,
        "stride_kv_scale_group": 1,
        "stride_indices_row": 2051,
        "stride_indices_col": 1,
        "stride_output_row": 16384,
        "stride_output_head": 512,
        "stride_output_dim": 1,
        "HEAD_DIM": 512,
        "SCALE_GROUP_SIZE": 128,
        "USE_KV_SCALE": not is_sm86,
        "BLOCK_T": block_t,
    }
    _compile(
        label=f"sparse-mla-sm{capability}",
        capability=capability,
        kernel=sparse._glm5_next_sparse_mla_decode_kernel,
        signature=signature,
        constants=constants,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _compile_kda(kda: ModuleType, capability: int) -> None:
    capability_tuple = (capability // 10, capability % 10)
    gate_config, decode_config = kda.glm5_next_kda_launch_config(capability_tuple)
    gate_block_t, gate_warps, gate_stages = gate_config
    decode_warps, decode_stages = decode_config
    _compile(
        label=f"kda-gate-sm{capability}",
        capability=capability,
        kernel=kda._glm5_next_safe_gate_kernel,
        signature={
            "raw_gate": "*bf16",
            "A_log": "*fp32",
            "output": "*fp32",
            "dt_bias": "*bf16",
            "lower_bound": "fp32",
            "T": "i32",
            "H": "i32",
        },
        constants={
            "D": 128,
            "BT": gate_block_t,
            "BD": 128,
            "HAS_BIAS": True,
        },
        num_warps=gate_warps,
        num_stages=gate_stages,
    )
    _compile(
        label=f"kda-decode-sm{capability}",
        capability=capability,
        kernel=kda._glm5_next_safe_decode_kernel,
        signature={
            "A_log": "*fp32",
            "raw_gate": "*bf16",
            "dt_bias": "*bf16",
            "lower_bound": "fp32",
            "q": "*bf16",
            "k": "*bf16",
            "v": "*bf16",
            "raw_beta": "*bf16",
            "output": "*bf16",
            "state_source": "*fp32",
            "state_indices": "*i32",
            "query_start_loc": "*i32",
            "scale": "fp32",
            "T": "i32",
        },
        constants={
            "B": 1,
            "H": 16,
            "HV": 64,
            "K": 128,
            "V": 128,
            "BK": 128,
            "BV": 32,
            "USE_INITIAL_STATE": True,
            "USE_QK_L2NORM_IN_KERNEL": True,
            "IS_VARLEN": True,
            "SPLIT_N_HV_GRID": False,
        },
        num_warps=decode_warps,
        num_stages=decode_stages,
    )


def _runtime_stride_signature(prefixes: tuple[str, ...]) -> dict[str, str]:
    return {name: "i32" for name in prefixes}


def _compile_kpool_sm86(kpool: ModuleType) -> None:
    capability = 86
    _compile(
        label="kpool-gather-sm86",
        capability=capability,
        kernel=kpool._gather_index_k_bf16_prefix_into_kernel,
        signature={
            "buf_ptr": "*bf16",
            "page_indices_ptr": "*i32",
            "k_out_ptr": "*bf16",
        },
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8192,
            "HEAD_DIM": 128,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=2,
    )
    writer_signature = {
        "buf_ptr": "*bf16",
        "slot_k_ptr": "*bf16",
        "slot_score_ptr": "*bf16",
        "ape_ptr": "*fp32",
        "loc_ptr": "*i64",
        "write_mask_ptr": "*i1",
        "compressed_k_ptr": "*bf16",
        "compressed_scale_ptr": "*bf16",
        **_runtime_stride_signature(
            (
                "slot_k_stride_0",
                "slot_k_stride_1",
                "slot_score_stride_0",
                "slot_score_stride_1",
                "ape_stride_0",
            )
        ),
    }
    _compile(
        label="kpool-writer-sm86",
        capability=capability,
        kernel=kpool._kpool_softmax_rotate_write_cache_bf16_kernel,
        signature=writer_signature,
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8192,
            "POOL_SIZE": 4,
            "HEAD_DIM": 128,
            "HAS_WRITE_MASK": False,
            "RETURN_COMPRESSED": False,
            "WRITE_CACHE": True,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=2,
    )
    _compile(
        label="kpool-writer-return-sm86",
        capability=capability,
        kernel=kpool._kpool_softmax_rotate_write_cache_bf16_kernel,
        signature={**writer_signature, "compressed_scale_ptr": "*fp32"},
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8192,
            "POOL_SIZE": 4,
            "HEAD_DIM": 128,
            "HAS_WRITE_MASK": False,
            "RETURN_COMPRESSED": True,
            "WRITE_CACHE": False,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=2,
    )
    _compile(
        label="kpool-writer-write-return-sm86",
        capability=capability,
        kernel=kpool._kpool_softmax_rotate_write_cache_bf16_kernel,
        signature={**writer_signature, "compressed_scale_ptr": "*fp32"},
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8192,
            "POOL_SIZE": 4,
            "HEAD_DIM": 128,
            "HAS_WRITE_MASK": False,
            "RETURN_COMPRESSED": True,
            "WRITE_CACHE": True,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=2,
    )
    common_decode_signature = {
        "buf_ptr": "*bf16",
        "tail_k_ptr": "*bf16",
        "tail_score_ptr": "*bf16",
        "key_ptr": "*bf16",
        "slot_score_ptr": "*bf16",
        "ape_ptr": "*fp32",
        "block_tables_ptr": "*i32",
        "req_pool_indices_ptr": "*i32",
        "positions_ptr": "*i64",
        "seq_lens_ptr": "*i32",
        "out_cache_loc_ptr": "*i64",
        **_runtime_stride_signature(
            (
                "tail_k_stride_0",
                "tail_k_stride_1",
                "tail_score_stride_0",
                "tail_score_stride_1",
                "key_stride_0",
                "slot_score_stride_0",
                "ape_stride_0",
                "block_tables_stride_0",
                "block_tables_stride_1",
            )
        ),
    }
    _compile(
        label="kpool-decode-sm86",
        capability=capability,
        kernel=kpool._kpool_decode_update_and_maybe_write_cache_bf16_kernel,
        signature=common_decode_signature,
        constants={
            "REQ_POOL_SIZE": 2048,
            "BUF_NUMEL_PER_PAGE": 8192,
            "POOL_SIZE": 4,
            "TAIL_SIZE": 4,
            "HEAD_DIM": 128,
            "BLOCK_TABLE_COLS": 16384,
            "BLOCK_D": 128,
            "SLOTS_PER_PAGE": 64,
        },
        num_warps=4,
        num_stages=2,
    )
    _compile(
        label="kpool-assemble-sm86",
        capability=capability,
        kernel=kpool._kpool_assemble_softmax_rotate_write_cache_bf16_kernel,
        signature={
            "buf_ptr": "*bf16",
            "chunk_k_ptr": "*bf16",
            "chunk_score_ptr": "*bf16",
            "tail_k_ptr": "*bf16",
            "tail_score_ptr": "*bf16",
            "req_pool_idx_ptr": "*i32",
            "n_from_tail_ptr": "*i32",
            "chunk_src_start_ptr": "*i64",
            "tail_logical_base_ptr": "*i64",
            "ape_ptr": "*fp32",
            "loc_ptr": "*i64",
            "write_mask_ptr": "*i1",
            **_runtime_stride_signature(
                (
                    "chunk_stride_0",
                    "tail_stride_0",
                    "tail_stride_1",
                    "ape_stride_0",
                )
            ),
        },
        constants={
            "BUF_NUMEL_PER_PAGE": 8192,
            "POOL_SIZE": 4,
            "TAIL_SIZE": 4,
            "HEAD_DIM": 128,
            "HAS_WRITE_MASK": True,
            "BLOCK_D": 128,
            "SLOTS_PER_PAGE": 64,
        },
        num_warps=4,
        num_stages=2,
    )


def _compile_kpool_sm89(kpool: ModuleType) -> None:
    capability = 89
    _compile(
        label="kpool-gather-sm89",
        capability=capability,
        kernel=kpool._gather_index_k_scale_prefix_into_kernel,
        signature={
            "buf_u8_ptr": "*u8",
            "buf_fp32_ptr": "*fp32",
            "page_indices_ptr": "*i32",
            "k_out_ptr": "*u8",
            "scale_out_ptr": "*fp32",
        },
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8448,
            "HEAD_DIM": 128,
            "S_OFFSET_NBYTES_IN_PAGE": 8192,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=3,
    )
    writer_signature = {
        "buf_fp8_ptr": "*fp8e4nv",
        "buf_fp32_ptr": "*fp32",
        "slot_k_ptr": "*bf16",
        "slot_score_ptr": "*bf16",
        "ape_ptr": "*fp32",
        "loc_ptr": "*i64",
        "write_mask_ptr": "*i1",
        "compressed_k_ptr": "*fp8e4nv",
        "compressed_scale_ptr": "*fp32",
        **_runtime_stride_signature(
            (
                "slot_k_stride_0",
                "slot_k_stride_1",
                "slot_score_stride_0",
                "slot_score_stride_1",
                "ape_stride_0",
            )
        ),
    }
    _compile(
        label="kpool-writer-sm89",
        capability=capability,
        kernel=kpool._kpool_softmax_rotate_write_cache_kernel,
        signature=writer_signature,
        constants={
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8448,
            "POOL_SIZE": 4,
            "HEAD_DIM": 128,
            "S_OFFSET_NBYTES_IN_PAGE": 8192,
            "ROUND_SCALE": False,
            "HAS_WRITE_MASK": False,
            "RETURN_COMPRESSED": False,
            "WRITE_CACHE": True,
            "BLOCK_D": 128,
        },
        num_warps=4,
        num_stages=3,
    )
    decode_signature = {
        "buf_fp8_ptr": "*fp8e4nv",
        "buf_fp32_ptr": "*fp32",
        "tail_k_ptr": "*bf16",
        "tail_score_ptr": "*bf16",
        "key_ptr": "*bf16",
        "slot_score_ptr": "*bf16",
        "ape_ptr": "*fp32",
        "block_tables_ptr": "*i32",
        "req_pool_indices_ptr": "*i32",
        "positions_ptr": "*i64",
        "seq_lens_ptr": "*i32",
        "out_cache_loc_ptr": "*i64",
        **_runtime_stride_signature(
            (
                "tail_k_stride_0",
                "tail_k_stride_1",
                "tail_score_stride_0",
                "tail_score_stride_1",
                "key_stride_0",
                "slot_score_stride_0",
                "ape_stride_0",
                "block_tables_stride_0",
                "block_tables_stride_1",
            )
        ),
    }
    _compile(
        label="kpool-decode-sm89",
        capability=capability,
        kernel=kpool._kpool_decode_update_and_maybe_write_cache_kernel,
        signature=decode_signature,
        constants={
            "REQ_POOL_SIZE": 2048,
            "PAGE_SIZE": 64,
            "BUF_NUMEL_PER_PAGE": 8448,
            "POOL_SIZE": 4,
            "TAIL_SIZE": 4,
            "HEAD_DIM": 128,
            "BLOCK_TABLE_COLS": 16384,
            "S_OFFSET_NBYTES_IN_PAGE": 8192,
            "ROUND_SCALE": False,
            "BLOCK_D": 128,
            "SLOTS_PER_PAGE": 64,
        },
        num_warps=4,
        num_stages=3,
    )
    _compile(
        label="kpool-assemble-sm89",
        capability=capability,
        kernel=kpool._kpool_assemble_softmax_rotate_write_cache_kernel,
        signature={
            "buf_fp8_ptr": "*fp8e4nv",
            "buf_fp32_ptr": "*fp32",
            "chunk_k_ptr": "*bf16",
            "chunk_score_ptr": "*bf16",
            "tail_k_ptr": "*bf16",
            "tail_score_ptr": "*bf16",
            "req_pool_idx_ptr": "*i32",
            "n_from_tail_ptr": "*i32",
            "chunk_src_start_ptr": "*i64",
            "tail_logical_base_ptr": "*i64",
            "ape_ptr": "*fp32",
            "loc_ptr": "*i64",
            "write_mask_ptr": "*i1",
            **_runtime_stride_signature(
                (
                    "chunk_stride_0",
                    "tail_stride_0",
                    "tail_stride_1",
                    "ape_stride_0",
                )
            ),
        },
        constants={
            "BUF_NUMEL_PER_PAGE": 8448,
            "POOL_SIZE": 4,
            "TAIL_SIZE": 4,
            "HEAD_DIM": 128,
            "S_OFFSET_NBYTES_IN_PAGE": 8192,
            "ROUND_SCALE": False,
            "HAS_WRITE_MASK": True,
            "BLOCK_D": 128,
            "SLOTS_PER_PAGE": 64,
        },
        num_warps=4,
        num_stages=3,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("86", "89"),
        default=("86", "89"),
    )
    parser.add_argument(
        "--resource-report",
        action="store_true",
        help="force recompilation and print ptxas register/spill usage",
    )
    args = parser.parse_args()

    if args.resource_report:
        from triton import knobs

        knobs.compilation.always_compile = True
        knobs.nvidia.dump_ptxas_log = True

    indexer = _load(
        "_glm5_next_indexer_compile",
        "python/sglang/srt/layers/attention/nsa/glm5_next_indexer_triton.py",
    )
    sparse = _load(
        "_glm5_next_sparse_compile",
        "python/sglang/srt/layers/attention/nsa/glm5_next_sparse_attention.py",
    )
    kda = _load(
        "_glm5_next_kda_compile",
        "python/sglang/srt/layers/attention/linear/kernels/glm5_next_kda_ops.py",
    )
    kpool = _load(
        "_glm5_next_kpool_compile",
        "python/sglang/srt/layers/attention/nsa/kpool_fp8_index.py",
    )

    for target in args.targets:
        capability = int(target)
        _compile_indexer(indexer, capability)
        _compile_sparse_mla(sparse, capability)
        _compile_kda(kda, capability)
        if capability == 86:
            _compile_kpool_sm86(kpool)
        else:
            _compile_kpool_sm89(kpool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
