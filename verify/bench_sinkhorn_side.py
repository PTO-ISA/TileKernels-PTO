"""Perf side of the sinkhorn ASC/VMI A/B: flush-L2 timing over the test bench grid.

Run once per side (ASC kernels with the tl-run env, VMI kernels with the
tkvmi env); SIDE=asc|vmi selects the kernel module.  Writes bench_sinkhorn_<side>.json.

Bench grid from tests/mhc/test_sinkhorn.py _SINKHORN_BENCH_CASES
((1, 512, 4) and (1, 8192, 4) -> num_tokens 512 / 8192) plus a mid point
(n=2048); repeat=10, eps=1e-6 as declared by the test.

Timing: per iteration, a 512 MB NPU scratch buffer is memset to flush L2,
then the kernel is timed between NPU events; the flush is outside the timed
window.  5 warmup + 20 timed iterations; min and mean reported.
"""

import json
import sys

SIDE = sys.argv[1] if len(sys.argv) > 1 else "vmi"
sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores

if SIDE == "asc":
    from tile_kernels.mhc.sinkhorn_asc import mhc_sinkhorn_bwd_asc as bwd, mhc_sinkhorn_fwd_asc as fwd
else:
    from tile_kernels.mhc.sinkhorn_vmi import mhc_sinkhorn_bwd_vmi as bwd, mhc_sinkhorn_fwd_vmi as fwd

DEV = "npu"
NVC = get_num_vec_cores()
WARMUP, ITERS = 5, 20
FLUSH = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
REPEAT, EPS = 10, 1e-6

CASES = [512, 2048, 8192]


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
for n in CASES:
    x = torch.randn((n, 4, 4), dtype=torch.float32, device=DEV)
    o_grad = torch.randn((n, 4, 4), dtype=torch.float32, device=DEV)
    out = torch.empty_like(x)
    grad_input = torch.empty_like(x)

    def run_fwd():
        fwd(x, out, REPEAT, EPS, token_block_size=4, num_vec_cores=NVC)

    tmin, tmean = timed(run_fwd)
    io_gb = (x.numel() + out.numel()) * 4 / 1e9  # read x + write out
    results.append({"op": "fwd", "n": n, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} fwd n={n}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.6f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)

    def run_bwd():
        bwd(o_grad, x, grad_input, REPEAT, EPS, token_block_size=8, num_vec_cores=NVC)

    tmin, tmean = timed(run_bwd)
    io_gb = (o_grad.numel() + x.numel() + grad_input.numel()) * 4 / 1e9  # read g,x + write dx
    results.append({"op": "bwd", "n": n, "us_min": tmin, "us_mean": tmean, "io_gb": io_gb})
    print(f"{SIDE} bwd n={n}: min={tmin:.2f}us mean={tmean:.2f}us io={io_gb:.6f}GB bw={io_gb / tmin * 1e6:.0f}GB/s", flush=True)
    del x, o_grad, out, grad_input
    torch.npu.empty_cache()

out_path = f"/data/sunkexin/tkvmi/work/bench_sinkhorn_{SIDE}.json"
with open(out_path, "w") as f:
    json.dump({"side": SIDE, "results": results}, f, indent=2)
print(f"wrote {out_path}")
