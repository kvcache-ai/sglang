"""Standalone DSV4 MXFP4 kernel correctness/performance smoke.

Run each backend in a separate process so its prepared weights and allocator
state cannot bias the other backend::

  python bench_v4_mxfp4_marlin.py --backend marlin
  python bench_v4_mxfp4_marlin.py --backend triton
"""

import argparse

import torch
import triton


def make_inputs(args):
    torch.manual_seed(17)
    e, m, k, n, topk = (
        args.experts,
        args.tokens,
        args.hidden,
        args.intermediate,
        args.topk,
    )
    w13 = torch.randint(0, 256, (e, 2 * n, k // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((e, 2 * n, k // 32), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((e, k, n // 32), 127, dtype=torch.uint8, device="cuda")
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.01
    ids = torch.randint(0, e, (m, topk), dtype=torch.int32, device="cuda")
    gates = torch.rand((m, topk), dtype=torch.float32, device="cuda")
    return w13, s13, w2, s2, x, ids, gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("marlin", "triton"), required=True)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--rep-ms", type=int, default=2000)
    args = parser.parse_args()

    w13, s13, w2, s2, x, ids, gates = make_inputs(args)
    if args.backend == "marlin":
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            apply_v4_marlin_moe,
            prepare_v4_mxfp4_marlin,
        )

        prepared = prepare_v4_mxfp4_marlin(w13, s13, w2, s2)

        def fn():
            return apply_v4_marlin_moe(
                hidden_states=x,
                prepared=prepared,
                topk_weights=gates,
                topk_ids=ids,
            )

    else:
        from sglang.srt.layers.quantization.v4_triton_kernels_moe import (
            apply_v4_triton_kernels_moe,
            convert_v4_weights_to_triton_kernels,
        )

        w13_tk, w13_pcg, w2_tk, w2_pcg = convert_v4_weights_to_triton_kernels(
            w13, s13.view(torch.float8_e8m0fnu), w2, s2.view(torch.float8_e8m0fnu)
        )

        def fn():
            return apply_v4_triton_kernels_moe(
                hidden_states=x,
                w13_swiz=w13_tk,
                w13_pcg=w13_pcg,
                w2_swiz=w2_tk,
                w2_pcg=w2_pcg,
                topk_weights=gates,
                topk_ids=ids,
                intermediate_size=args.intermediate,
                num_experts=args.experts,
            )

    fn()
    torch.cuda.synchronize()
    p50, p95 = triton.testing.do_bench(
        fn,
        warmup=args.warmup_ms,
        rep=args.rep_ms,
        quantiles=[0.5, 0.95],
    )
    print(
        f"backend={args.backend} M={args.tokens} E={args.experts} "
        f"K={args.hidden} N={args.intermediate} topk={args.topk} "
        f"p50_ms={p50:.4f} p95_ms={p95:.4f} jitter={p95 / p50:.4f} "
        f"allocated_gb={torch.cuda.memory_allocated() / 1e9:.3f}"
    )


if __name__ == "__main__":
    main()
