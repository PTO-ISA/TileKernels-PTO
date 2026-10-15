"""Perf side of the pre_split_mixes ASC/VMI A/B: flush-L2 timing over the bench grid.

Run once per side (ASC kernels with the tl-run env, VMI kernels with the
tkvmi env); SIDE=asc|vmi selects the kernel module.  Writes
bench_pre_split_<side>.json.

Timing: per iteration, a 512 MB NPU scratch buffer is memset to flush L2,
then the kernel is timed between NPU events; the flush is outside the timed
window.  5 warmup + 20 timed iterations; min and mean reported.

Grid = tests/mhc/test_pre_split_mixes.py _PRE_SPLIT_BENCH_CASES (mhc_mult=4).
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

if SIDE == "asc":
    from tile_kernels.mhc.pre_split_mixes_asc import mhc_pre_split_mixes_bwd_mhc4_asc as bwd, mhc_pre_split_mixes_fwd_simd_mhc4_asc as fwd
else:
    from tile_kernels.mhc.pre_split_mixes_vmi import mhc_pre_split_mixes_bwd_mhc4_vmi as bwd, mhc_pre_split_mixes_fwd_simd_mhc4_vmi as fwd

DEV = "npu"
NVC = get_num_vec_cores()
MHC_MULT = 4
MHC_MULT2 = 16
MHC_MULT3 = 24
POST_MULT = 2.0
PRE_EPS = 1e-2
WARMUP, ITERS = 5, 20
FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)

CASES = [
    # (n0, n1) from tests/mhc/test_pre_split_mixes.py _PRE_SPLIT_BENCH_CASES
    (1, 512),
    (1, 8192),
]


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
for n0, n1 in CASES:
    n = n0 * n1

    input_mixes = torch.randn((n, MHC_MULT3), dtype=torch.float32, device=DEV)
    mhc_scale = torch.randn((3,), dtype=torch.float32, device=DEV)
    mhc_base = torch.randn((MHC_MULT3,), dtype=torch.float32, device=DEV)
    pre_grad = torch.randn((n, MHC_MULT), dtype=torch.float32, device=DEV)
    post_grad = torch.randn((n, MHC_MULT), dtype=torch.float32, device=DEV)
    comb_grad = torch.randn((n, MHC_MULT2), dtype=torch.float32, device=DEV)
    post_mix = torch.rand((n, MHC_MULT), dtype=torch.float32, device=DEV)

    pre_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    post_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    comb_out = torch.empty((n, MHC_MULT2), dtype=torch.float32, device=DEV)

    def run_fwd():
        fwd(
            input_mixes, mhc_scale, mhc_base, pre_out, post_out, comb_out,
            POST_MULT, PRE_EPS,
            token_block_size=64, num_stages=2, num_vec_cores=NVC,
        )

    tmin, tmean = timed(run_fwd)
    io_gb = (input_mixes.numel() * 4 + mhc_scale.numel() * 4 + mhc_base.numel() * 4 + (pre_out.numel() + post_out.numel() + comb_out.numel()) * 4) / 1e9
    results.append({"op": "fwd", "n0": n0, "n1": n1, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} fwd n1={n1}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.3f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)

    input_mixes_grad = torch.empty((n, MHC_MULT3), dtype=torch.float32, device=DEV)
    scale_grad_partial = torch.empty((NVC, 3), dtype=torch.float32, device=DEV)
    base_grad_partial = torch.empty((NVC, MHC_MULT3), dtype=torch.float32, device=DEV)

    def run_bwd():
        bwd(
            pre_grad, post_grad, comb_grad, input_mixes, post_mix, mhc_scale, mhc_base,
            input_mixes_grad, scale_grad_partial, base_grad_partial,
            POST_MULT,
            token_block_size=64, num_stages=2, num_vec_cores=NVC,
        )

    tmin, tmean = timed(run_bwd)
    io_gb = (
        (pre_grad.numel() + post_grad.numel() + comb_grad.numel() + input_mixes.numel() + post_mix.numel() + input_mixes_grad.numel()) * 4
        + mhc_scale.numel() * 4 + mhc_base.numel() * 4
        + (scale_grad_partial.numel() + base_grad_partial.numel()) * 4
    ) / 1e9
    results.append({"op": "bwd", "n0": n0, "n1": n1, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} bwd n1={n1}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.3f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)
    del input_mixes, pre_grad, post_grad, comb_grad, post_mix, pre_out, post_out, comb_out, input_mixes_grad
    torch.npu.empty_cache()

out_path = f"/data/sunkexin/tkvmi/work/bench_pre_split_{SIDE}.json"
with open(out_path, "w") as f:
    json.dump({"side": SIDE, "results": results}, f, indent=2)
print(f"wrote {out_path}")
