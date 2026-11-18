"""Export PTO/VMI and physical lowering evidence for the standalone variants."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "PTO_FLAGS",
    "--pto-backend=vpto --pto-level=level3 "
    "--vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 "
    "--enable-vecscope-mem-bar=false",
)
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / ".tilelang_cache_fp8"))

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
import tilelang  # noqa: E402

from deepinfer_ops.fp8_kvstore.quant_asc import store_kv_asc  # noqa: E402
from deepinfer_ops.fp8_kvstore.quant_pto import store_kv_pto  # noqa: E402
from tilelang.contrib.pto_toolchain import compile_ptodsl_to_pto, parse_pto_kernel_names  # noqa: E402
from tilelang.ascend.target import normalize_pto_target  # noqa: E402


def op_counts(text: str, prefix: str) -> dict[str, int]:
    return dict(sorted(Counter(re.findall(prefix + r"\.([a-zA-Z0-9_]+)", text)).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["asc", "native_half", "compact_ub", "direct_brc", "native_half_hoisted", "all"], default="all")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "lowering")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ.get("BENCH_DEVICES", "1")))
    args.out.mkdir(parents=True, exist_ok=True)
    variants = ["asc", "native_half", "compact_ub", "direct_brc", "native_half_hoisted"] if args.variant == "all" else [args.variant]
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    for variant in variants:
        entry = {}
        try:
            if variant == "asc":
                kernel = store_kv_asc()
                source_path = args.out / "asc_kernel.asc"
                source_path.write_text(kernel.get_kernel_source(kernel_only=True), encoding="utf-8")
                entry["source"] = source_path.name
                summary[variant] = entry
                continue
            prim_func = store_kv_pto.get_tir(variant)
            pass_configs = dict(prim_func.attrs.get("tilelang_pass_configs", {}))
            target = normalize_pto_target("pto")
            with tilelang.transform.PassContext(opt_level=3, config=pass_configs), target:
                source = tilelang.lower(prim_func, target=target, enable_device_compile=False).kernel_source
            dsl_path = args.out / f"{variant}.ptodsl.py"
            pto_path = args.out / f"{variant}.pto"
            vpto_path = args.out / f"{variant}.vpto"
            dsl_path.write_text(source, encoding="utf-8")
            compile_ptodsl_to_pto(source, parse_pto_kernel_names(source), dsl_path, pto_path)
            command = ["ptoas", "--pto-arch=a5", *shlex.split(os.environ["PTO_FLAGS"]), "--emit-vpto", str(pto_path), "-o", str(vpto_path)]
            pto_text = pto_path.read_text(encoding="utf-8")
            entry.update({
                "ptodsl": dsl_path.name,
                "pto": pto_path.name,
                "command": " ".join(command),
                "vmi_op_counts": op_counts(pto_text, r"pto\.vmi"),
            })
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            vpto_text = vpto_path.read_text(encoding="utf-8")
            entry["vpto"] = vpto_path.name
            entry["vpto_op_counts"] = op_counts(vpto_text, r"pto")
            entry["mem_bar_lines"] = [line.strip() for line in vpto_text.splitlines() if "mem_bar" in line]
            entry["evidence_lines"] = [
                line.strip()
                for line in vpto_text.splitlines()
                if any(token in line for token in ("vbrc", "vselr", "vcmax", "vcgmax", "mem_bar", "inv_ub"))
            ][:80]
        except Exception as exc:  # Keep exact compiler rejection text.
            entry["error"] = f"{type(exc).__name__}: {exc}"
        summary[variant] = entry
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
