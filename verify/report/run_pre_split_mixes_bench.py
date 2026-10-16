import os, sys
os.environ.setdefault("TMPDIR", "/data/sunkexin/tkvmi/tmp")
side = sys.argv[1]                      # asc | vmi
sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")
if side == "vmi":
    import importlib
    for mod_name, entries in {
        "tile_kernels.mhc.pre_split_mixes_asc": (
            "mhc_pre_split_mixes_fwd_simd_mhc4",
            "mhc_pre_split_mixes_bwd_mhc4",
        ),
    }.items():
        asc = importlib.import_module(mod_name)
        vmi = importlib.import_module(mod_name.replace("_asc", "_vmi"))
        for base in entries:
            setattr(asc, f"{base}_asc", getattr(vmi, f"{base}_vmi"))
import pytest
sys.exit(pytest.main([
    "-m", "benchmark", "--run-benchmark", "-p", "no:cacheprovider",
    f"--benchmark-output=/data/sunkexin/tkvmi/work/pre_split_mixes_bench_{side}.jsonl",
    "/data/sunkexin/tkvmi/TileKernels-PTO/tests/mhc/test_pre_split_mixes.py",
]))
