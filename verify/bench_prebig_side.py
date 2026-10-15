"""Perf side of the pre_big_fuse ASC/VMI A/B: flush-L2 timing over the bench grid.

Run once per side; SIDE=asc|vmi selects the kernel module.  Writes
bench_prebig_<side>.json.

Timing: per iteration, a 512 MB NPU scratch buffer is memset to flush L2,
then the kernel is timed between NPU events; the flush is outside the timed
window.  5 warmup + 20 timed iterations; min and mean reported.
"""

import json
import math
import os
import sys

SIDE = sys.argv[1] if len(sys.argv) > 1 else "vmi"
sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores

if SIDE == "asc":
    from tile_kernels.mhc.pre_big_fuse_asc import mhc_pre_big_fuse_fwd_asc as fused
else:
    from tile_kernels.mhc.pre_big_fuse_vmi import mhc_pre_big_fuse_fwd_vmi as fused

DEV = "npu"
NVC = get_num_vec_cores()
MHC = 4
WARMUP, ITERS = 5, 20
FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)

CASES = [
    # (n1, h) grid from tests/mhc/test_pre_big_fuse.py _PRE_BIG_FUSE_BENCH_CASES
    (512, 4096),
    (512, 7168),
    (8192, 4096),
    (8192, 5120),
    (8192, 7168),
]


def host_h_blk(hidden: int) -> int:
    return min(math.gcd(512, hidden), 256)


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(ITERS):
        FLUSH.fill_(0.0)
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    return min(times), sum(times) / len(times)


def amortized(fn, batch=50):
    """50 back-to-back launches / 50: kernel body without per-launch noise."""
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(batch):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1e3 / batch  # ms -> us


results = []
for n1, h in CASES:
    g = torch.Generator(device="cpu").manual_seed(0)
    mhc3 = MHC * 2 + MHC * MHC
    residual = (torch.randn((1, n1, MHC, h), generator=g) * (1 + torch.arange(MHC).mul(0.01).view(1, 1, -1, 1))).bfloat16().to(DEV)
    out_mul = (torch.randn((n1, mhc3), generator=g) * 1e-2).to(DEV)
    sqrsum = (torch.randn((n1,), generator=g).abs() * n1).to(DEV)
    mhc_scale = (torch.randn((3,), generator=g) * 0.1).to(DEV)
    mhc_base = (torch.randn((mhc3,), generator=g) * 0.1).to(DEV)

    post_mix = torch.empty((n1, MHC), dtype=torch.float32, device=DEV)
    comb_mix = torch.empty((n1, MHC * MHC), dtype=torch.float32, device=DEV)
    layer_input = torch.empty((n1, h), dtype=torch.bfloat16, device=DEV)

    def run():
        fused(
            out_mul.view(1, n1, -1),
            sqrsum.view(1, n1),
            mhc_scale,
            mhc_base,
            residual.view(n1, MHC, h),
            post_mix,
            comb_mix,
            layer_input,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            10,
            h_blk=host_h_blk(h),
            n_splits=1,
            token_block=16,
            num_stages=4,
            num_vec_cores=NVC,
        )

    tmin, tmean = timed(run)
    tamort = amortized(run)
    # io bytes mirror test_pre_big_fuse.py's count_bytes over the same tensors
    io_gb = (
        out_mul.numel() * 4 + sqrsum.numel() * 4 + mhc_scale.numel() * 4 + mhc_base.numel() * 4
        + residual.numel() * 2 + post_mix.numel() * 4 + comb_mix.numel() * 4 + layer_input.numel() * 2
    ) / 1e9
    results.append({"op": "fwd", "n1": n1, "h": h, "us_min": tmin, "us_mean": tmean, "us_amort": tamort, "io_gb": io_gb})
    print(f"{SIDE} n1={n1} h={h}: min={tmin:.2f}us mean={tmean:.2f}us amort50={tamort:.2f}us io={io_gb:.3f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)
    del residual, out_mul, sqrsum, post_mix, comb_mix, layer_input
    torch.npu.empty_cache()

out_path = f"/data/sunkexin/tkvmi/work/bench_prebig_{SIDE}.json"
with open(out_path, "w") as f:
    json.dump({"side": SIDE, "results": results}, f, indent=2)
print(f"wrote {out_path}")
