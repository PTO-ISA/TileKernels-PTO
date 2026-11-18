"""Byte-check and profile the isolated sequential FP8 KV-store variants."""

from __future__ import annotations

import argparse
import importlib.metadata
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
    "--vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 "
    "--enable-vecscope-mem-bar=false",
)
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / ".tilelang_cache_fp8"))

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
from tilelang.profiler import do_bench  # noqa: E402

from deepinfer_ops.fp8_kvstore.common import V41_FORMAT  # noqa: E402
from deepinfer_ops.fp8_kvstore.quant_asc import store_kv_asc  # noqa: E402
from deepinfer_ops.fp8_kvstore.quant_pto import store_kv_pto  # noqa: E402
from deepinfer_ops.fp8_kvstore.store_common import BLOCK_SIZE, PAYLOAD_BYTES  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "bench.jsonl"
TOKENS = 65536
PRODUCTION_VARIANTS = {"native_half", "compact_ub", "direct_brc", "native_half_hoisted"}
ALL_VARIANTS = ["native_half", "compact_ub", "direct_brc", "native_half_hoisted", "scale_only", "pack_only"]


def _toolchain():
    ptoas = shutil.which("ptoas")
    version = ""
    if ptoas:
        version = subprocess.check_output([ptoas, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    return {
        "tilelang": importlib.metadata.version("tilelang"),
        "ptoas": version,
        "torch_npu": importlib.metadata.version("torch-npu"),
        "pto_flags": os.environ["PTO_FLAGS"],
    }


def _time(fn, device, repeat, detail):
    return do_bench(
        fn,
        backend="msprof_detail" if detail else "msprof",
        warmup=0,
        rep=0,
        _n_warmup=5,
        _n_repeat=repeat,
        return_mode="median",
        device=device,
        cache_size=256,
        early_stop_baseline=None if detail else 10000,
    )


def _profile_dict(profile):
    if profile is None:
        return None
    return {
        "dur_ns": profile.dur_ns,
        "aiv_total_cycles": profile.aiv_total_cycles,
        "aiv_vec": profile.aiv_vec,
        "aiv_scalar": profile.aiv_scalar,
        "aiv_mte2": profile.aiv_mte2,
        "aiv_mte3": profile.aiv_mte3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*ALL_VARIANTS, "all"], default="native_half")
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("BENCH_REP", "20")))
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    device_id = int(os.environ.get("BENCH_DEVICES", "1"))
    torch.npu.set_device(device_id)
    device = torch.device("npu", device_id)
    torch.manual_seed(0)
    num_blocks = (TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE
    layout = V41_FORMAT
    block_stride = BLOCK_SIZE * PAYLOAD_BYTES
    kv = torch.randn(TOKENS, layout.num_elements, dtype=torch.bfloat16, device=device)
    slots = torch.arange(TOKENS, dtype=torch.int32, device=device)
    storage = torch.zeros((num_blocks, block_stride), dtype=torch.uint8, device=device)
    print(json.dumps(_toolchain()), flush=True)

    for variant in ALL_VARIANTS if args.variant == "all" else [args.variant]:
        asc_storage = storage.clone()
        pto_storage = storage.clone()
        asc_cache = torch.as_strided(
            asc_storage,
            (num_blocks, BLOCK_SIZE, PAYLOAD_BYTES),
            (block_stride, PAYLOAD_BYTES, 1),
        )
        pto_cache = torch.as_strided(
            pto_storage,
            (num_blocks, BLOCK_SIZE, PAYLOAD_BYTES),
            (block_stride, PAYLOAD_BYTES, 1),
        )
        row = {
            "variant": variant,
            "production": variant in PRODUCTION_VARIANTS,
            "device": device_id,
            "tokens": TOKENS,
            "block_size": BLOCK_SIZE,
            "tokens_per_tile": 16,
            "num_stages": 3,
            "pattern": "sequential",
            "io_bytes": kv.nbytes + asc_cache.nbytes + slots.nbytes,
            **_toolchain(),
        }
        try:
            asc_kernel = store_kv_asc()
            pto_kernel = store_kv_pto(variant)
            asc_kernel(kv, asc_cache, slots)
            pto_kernel(kv, pto_cache, slots)
            torch.npu.synchronize()
            if variant in PRODUCTION_VARIANTS:
                matches = torch.equal(asc_storage.cpu(), pto_storage.cpu())
                row["correct"] = matches
                if not matches:
                    raise RuntimeError(f"{variant}: cache mismatch")
            else:
                row["correct"] = None
            if not args.correctness_only:
                asc_profile = _time(lambda: asc_kernel(kv, asc_cache, slots), device, args.repeat, args.detail)
                pto_profile = _time(lambda: pto_kernel(kv, pto_cache, slots), device, args.repeat, args.detail)
                row["asc_profile"] = _profile_dict(asc_profile)
                row["pto_profile"] = _profile_dict(pto_profile)
                row["asc_us"] = asc_profile.dur_us if args.detail else float(asc_profile) * 1000.0
                row["pto_us"] = pto_profile.dur_us if args.detail else float(pto_profile) * 1000.0
                row["ratio"] = row["asc_us"] / row["pto_us"]
                row["asc_gbs"] = row["io_bytes"] / row["asc_us"] / 1e3
                row["pto_gbs"] = row["io_bytes"] / row["pto_us"] / 1e3
                if args.detail:
                    print(f"ASC {asc_profile}", flush=True)
                    print(f"PTO {pto_profile}", flush=True)
                print(f"{variant}: asc={row['asc_us']:.2f}us pto={row['pto_us']:.2f}us ratio={row['ratio']:.3f}", flush=True)
            else:
                print(f"{variant}: output check passed", flush=True)
        except Exception as exc:  # Record compiler limitations without hiding them.
            row["correct"] = False
            row["production"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{variant}: ERROR {row['error']}", flush=True)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        del asc_storage, pto_storage, asc_cache, pto_cache
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
