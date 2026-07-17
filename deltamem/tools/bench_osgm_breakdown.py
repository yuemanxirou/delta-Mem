from __future__ import annotations

import argparse
import time

import torch

from deltamem.kernels.affine_scan_osgm import affine_scan_osgm_fwd
from deltamem.kernels.fla.ops.os_delta_rule.chunk_osgm_phase import (
    compute_osgm_phase1_fwd,
)


def measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    started_at = time.perf_counter()

    for _ in range(iters):
        fn()

    torch.cuda.synchronize()
    return (time.perf_counter() - started_at) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    device = "cuda"
    B = args.batch_size
    T = args.seq_len
    R = args.rank

    state = torch.randn(B, R, R, device=device, dtype=dtype)
    q = torch.randn(B, T, R, device=device, dtype=dtype)
    k = torch.randn(B, T, R, device=device, dtype=dtype)
    v = torch.randn(B, T, R, device=device, dtype=dtype)

    keep = torch.sigmoid(
        torch.randn(B, T, R, device=device, dtype=dtype)
    )
    erase = torch.sigmoid(
        torch.randn(B, T, R, device=device, dtype=dtype)
    )
    write = torch.sigmoid(
        torch.randn(B, T, R, device=device, dtype=dtype)
    )

    k_4d = k.unsqueeze(2)

    def run_phase1():
        return compute_osgm_phase1_fwd(
            k=k_4d,
            eta=0.03,
            use_denominator=True,
            d_min=1.0,
            d_max=2.0,
            cu_seqlens=None,
            initial_d=None,
            output_final_state=True,
        )

    # 先生成固定 d，避免 mapping/phase2 测试中重复包含 Phase 1。
    d_4d, final_d = run_phase1()
    d = d_4d.squeeze(2)

    def run_mapping():
        return (d * k.to(torch.float32)).to(dtype=k.dtype)

    k_write = run_mapping()

    def run_phase2():
        return affine_scan_osgm_fwd(
            state0=state,
            q=q,
            k_read=k,
            k_write=k_write,
            v=v,
            keep=keep,
            erase=erase,
            write=write,
            token_mask=None,
        )

    phase1_ms = measure(run_phase1, args.warmup, args.iters)
    mapping_ms = measure(run_mapping, args.warmup, args.iters)
    phase2_ms = measure(run_phase2, args.warmup, args.iters)

    print(
        {
            "batch_size": B,
            "seq_len": T,
            "rank": R,
            "dtype": args.dtype,
            "phase1_ms": round(phase1_ms, 4),
            "mapping_ms": round(mapping_ms, 4),
            "phase2_ms": round(phase2_ms, 4),
            "sum_ms": round(
                phase1_ms + mapping_ms + phase2_ms,
                4,
            ),
            "final_d_shape": tuple(final_d.shape),
            "final_d_dtype": str(final_d.dtype),
        }
    )


if __name__ == "__main__":
    main()
