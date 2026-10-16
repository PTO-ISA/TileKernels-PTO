"""Official pytest benchmark runner for mhc sinkhorn (vmi-verify skill §7).

side=asc runs the native dispatch (sinkhorn_kernel.py -> sinkhorn_asc);
side=vmi hijacks the lazily-imported _asc module attributes so the official
test suite exercises the VMI kernels end-to-end. Both sides emit JSONL via
--benchmark-output (msprof device-side kernel time with L2 flush).
"""

import os
import sys

os.environ.setdefault("TMPDIR", "/data/sunkexin/tkvmi/tmp")
side = sys.argv[1]  # asc | vmi
sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")
if side == "vmi":
    import importlib

    for mod_name, entries in {
        "tile_kernels.mhc.sinkhorn_asc": ("mhc_sinkhorn_fwd", "mhc_sinkhorn_bwd"),
    }.items():
        asc = importlib.import_module(mod_name)
        vmi = importlib.import_module(mod_name.replace("_asc", "_vmi"))
        for base in entries:
            setattr(asc, f"{base}_asc", getattr(vmi, f"{base}_vmi"))
import pytest

sys.exit(
    pytest.main(
        [
            "-m",
            "benchmark",
            "--run-benchmark",
            "-p",
            "no:cacheprovider",
            f"--benchmark-output=/data/sunkexin/tkvmi/work/sinkhorn_bench_{side}.jsonl",
            "/data/sunkexin/tkvmi/TileKernels-PTO/tests/mhc/test_sinkhorn.py",
        ]
    )
)
