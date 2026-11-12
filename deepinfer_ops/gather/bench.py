"""ASC vs PTO gather on one NPU.

Shape: ``m = n = 131072``, logical float32 rows of length 128 viewed as int8
so the kernel sees ``D = 512``. ``index[::4] = -1`` and ``index[1::4] = n``,
so half the rows are out of range and stay zero.

``ratio = ASC_us / PTO_us``. Timing is the msprof median after 5 warmup
launches. ``BENCH_REP`` (default 10) is the repeat count. ``BENCH_DEVICES``
(default 0) selects the NPU.

Release ptoas 0.72 must already be on ``PATH``. This script does not build
ptoas. Styles ``single`` and ``double`` are the release comparison. Pass
``--styles patched`` only when ``PATH`` points at a ptoas that cancels the
loop-invariant version term.

Example, from the worktree root, under the CANN environment::

    BENCH_DEVICES=0 BENCH_REP=10 python deepinfer_ops/gather/bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "PTO_FLAGS",
    "--pto-backend=vpto --pto-level=level3 "
    "--vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144",
)
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / ".tilelang_cache"))

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
from tilelang.profiler import do_bench  # noqa: E402

from deepinfer_ops.gather.gather_asc import gather_tl as gather_asc  # noqa: E402
from deepinfer_ops.gather.gather_pto import gather_tl as gather_pto  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "gather.jsonl"
M = N = 131072
LOGICAL_D = 128


def _inputs(device: int):
    torch.manual_seed(4)
    x = torch.randn(N, LOGICAL_D, dtype=torch.float32, device=f"npu:{device}").contiguous()
    index = torch.randint(0, N, (M,), dtype=torch.int32, device=f"npu:{device}")
    index[::4] = -1
    index[1::4] = N
    y = torch.empty(M, LOGICAL_D, dtype=torch.float32, device=f"npu:{device}")
    return x, index, y


def _view_args(x, index, y):
    x8 = x.view(torch.int8)
    y8 = y.view(torch.int8)
    return (x8, index, y8), x8.shape[1], x8.stride(0)


def _time(fn, device: int) -> float:
    ms = do_bench(
        fn,
        backend="msprof",
        warmup=0,
        rep=0,
        _n_warmup=5,
        _n_repeat=int(os.environ.get("BENCH_REP", "10")),
        return_mode="median",
        device=torch.device("npu", device),
        cache_size=256,
    )
    return float(ms) * 1000.0


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.detach().cpu(), b.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["single", "double"],
        choices=["single", "double", "patched"],
    )
    args = parser.parse_args()
    device = int(os.environ.get("BENCH_DEVICES", "0"))
    torch.npu.set_device(device)

    x, index, y_parallel = _inputs(device)
    args_parallel, d, stride = _view_args(x, index, y_parallel)
    asc_parallel = gather_asc(d, torch.int8, torch.int32, stride, zero_fill="parallel")
    asc_parallel(*args_parallel)

    y_simd = torch.empty_like(y_parallel)
    args_simd, _, _ = _view_args(x, index, y_simd)
    asc_simd = gather_asc(d, torch.int8, torch.int32, stride, zero_fill="simd")
    asc_simd(*args_simd)
    if not _equal(y_parallel, y_simd):
        raise SystemExit("ASC parallel and explicit simd fills differ")
    print("ASC parallel == simd", flush=True)

    asc_us = _time(lambda: asc_parallel(*args_parallel), device)
    simd_us = _time(lambda: asc_simd(*args_simd), device)
    print(f"ASC parallel={asc_us:.2f}us simd={simd_us:.2f}us ratio={simd_us / asc_us:.3f}", flush=True)

    ptoas_version = os.environ.get("PTOAS_VERSION", "")
    if not ptoas_version and shutil.which("ptoas"):
        try:
            ptoas_version = subprocess.check_output(
                ["ptoas", "--version"], text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            ptoas_version = f"unknown ({exc})"
    print(f"ptoas_version={ptoas_version}", flush=True)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "impl": "asc",
            "style": "parallel",
            "device": device,
            "us": asc_us,
            "correct": True,
            "m": M,
            "n": N,
            "logical_d": LOGICAL_D,
            "kernel_d": d,
        },
        {
            "impl": "asc",
            "style": "simd",
            "device": device,
            "us": simd_us,
            "correct": True,
            "asc_us": asc_us,
            "ratio_vs_parallel": simd_us / asc_us,
            "m": M,
            "n": N,
            "logical_d": LOGICAL_D,
            "kernel_d": d,
        },
    ]
    for style in args.styles:
        y = torch.empty_like(y_parallel)
        launch_args, _, _ = _view_args(x, index, y)
        kernel = gather_pto(d, torch.int8, torch.int32, stride, style=style)
        kernel(*launch_args)
        ok = _equal(y_parallel, y)
        if not ok:
            raise SystemExit(f"PTO style {style} does not match ASC")
        pto_us = _time(lambda kernel=kernel, launch_args=launch_args: kernel(*launch_args), device)
        ratio = asc_us / pto_us
        print(f"PTO {style}: asc={asc_us:.2f}us pto={pto_us:.2f}us ratio={ratio:.3f}", flush=True)
        rows.append(
            {
                "impl": "pto",
                "style": style,
                "device": device,
                "us": pto_us,
                "asc_us": asc_us,
                "ratio": ratio,
                "correct": True,
                "ptoas_version": ptoas_version,
                "m": M,
                "n": N,
                "logical_d": LOGICAL_D,
                "kernel_d": d,
            }
        )

    with RESULTS.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
