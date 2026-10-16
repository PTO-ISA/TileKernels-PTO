"""Perf side of the pre_apply_mix ASC/VMI A/B: flush-L2 timing over the test bench grid.

Run once per side (ASC kernels with the tl-run env, VMI kernels with the
tkvmi env); SIDE=asc|vmi selects the kernel module.  Writes bench_pre_<side>.json.

Timing: per iteration, a 512 MB NPU scratch buffer is memset to flush L2,
then the kernel is timed between NPU events; the flush is outside the timed
window.  5 warmup + 20 timed iterations; min and mean reported.
"""

import json
import os
import sys

SIDE = sys.argv[1] if len(sys.argv) > 1 else "vmi"
sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.utils import align, ceil_div

if SIDE == "asc":
    from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_bwd_asc as bwd, mhc_pre_apply_mix_fwd_asc as fwd
else:
    from tile_kernels.mhc.pre_apply_mix_vmi import mhc_pre_apply_mix_bwd_vmi as bwd, mhc_pre_apply_mix_fwd_vmi as fwd

DEV = "npu"
NVC = get_num_vec_cores()
WARMUP, ITERS = 5, 20
FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)

CASES = [
    # (n0, n1, h) grid from tests/mhc/test_pre_apply_mix.py _PRE_FWD_BENCH_CASES
    (1, 512, 4096),
    (1, 512, 5120),
    (1, 512, 7168),
    (1, 8192, 4096),
    (1, 8192, 5120),
    (1, 8192, 7168),
]


def host_h_blk_fwd(hidden: int) -> int:
    h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
    return align(h_blk, 64)


def host_h_blk_bwd(hidden: int) -> int:
    return hidden if hidden <= 4096 else align(min(ceil_div(hidden, 4), 3072), 64)


def host_num_stages_bwd(hidden: int) -> int:
    return 3 if hidden <= 4096 else 4


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


results = []
for n0, n1, h in CASES:
    mhc = 4
    n = n0 * n1

    x = torch.randn((n0, n1, mhc, h), dtype=torch.float32).to(torch.bfloat16).to(DEV)
    mix = torch.randn((n0, n1, mhc, 1), dtype=torch.float32).to(DEV)
    o_grad = torch.randn((n0, n1, h), dtype=torch.float32).to(torch.bfloat16).to(DEV)
    o = torch.empty((n0, n1, h), dtype=torch.bfloat16, device=DEV)

    def run_fwd():
        fwd(
            x.view(n, mhc, h), mix.view(n, mhc), o.view(n, h),
            h_blk=host_h_blk_fwd(h), num_vec_cores=NVC,
        )

    tmin, tmean = timed(run_fwd)
    io_gb = (x.numel() * 2 + mix.numel() * 4 + o.numel() * 2) / 1e9
    results.append({"op": "fwd", "n0": n0, "n1": n1, "h": h, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} fwd n1={n1} h={h}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.3f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)

    x_grad = torch.zeros_like(x)

    def run_bwd():
        bwd(
            o_grad.view(n, h), x.view(n, mhc, h), mix.view(n, mhc), x_grad.view(n, mhc, h),
            h_blk=host_h_blk_bwd(h), num_stages=host_num_stages_bwd(h), num_vec_cores=NVC,
        )

    tmin, tmean = timed(run_bwd)
    io_gb = (o_grad.numel() * 2 + x.numel() * 2 + mix.numel() * 4 + 2 * x_grad.numel() * 2 + n0 * n1 * mhc * 4) / 1e9
    results.append({"op": "bwd", "n0": n0, "n1": n1, "h": h, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} bwd n1={n1} h={h}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.3f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)
    del x, mix, o_grad, o, x_grad
    torch.npu.empty_cache()

out_path = f"/data/sunkexin/tkvmi/work/bench_pre_{SIDE}.json"
with open(out_path, "w") as f:
    json.dump({"side": SIDE, "results": results}, f, indent=2)
print(f"wrote {out_path}")
