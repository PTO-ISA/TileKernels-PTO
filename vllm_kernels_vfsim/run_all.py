"""Run every case under msprof and write the VF-cycle table.

Ratio = ASC_vf_cycles / PTO_vf_cycles. Above 1 means PTO used fewer cycles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from harness.compare import check_close, check_exact  # noqa: E402
from harness.sim import run_under_msprof  # noqa: E402


def _variants(case: Path) -> list[str]:
    path = case / "variants.txt"
    if not path.exists():
        return [""]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _ratio(asc: float | None, pto: float | None) -> float | None:
    if not asc or not pto:
        return None
    return asc / pto


def _flag(ratio: float | None, passed: bool) -> str:
    if not passed or ratio is None:
        return "FAIL"
    if ratio < 0.8:
        return "WARN"
    if ratio > 1.1:
        return "WIN"
    return "OK"


def _compare_dumps(case: str, variant: str) -> str:
    dump = ROOT / "perf" / "dumps"
    tag = f"{case}_{variant}" if variant else case
    asc_path = dump / f"{tag}_asc.pt"
    pto_path = dump / f"{tag}_pto.pt"
    if not asc_path.exists() or not pto_path.exists():
        return "no-dump"
    asc = torch.load(asc_path, weights_only=True)
    pto = torch.load(pto_path, weights_only=True)
    try:
        if asc.dtype.is_floating_point or pto.dtype.is_floating_point:
            check_close(pto.float(), asc.float(), tag, rtol=2e-2, atol=2e-2)
        else:
            check_exact(pto, asc, tag)
    except AssertionError as exc:
        return f"mismatch: {exc}"
    return "match"


def main() -> int:
    rows = []
    failed = 0
    for case_dir in sorted(p for p in (ROOT / "cases").iterdir() if (p / "run.py").exists()):
        for variant in _variants(case_dir):
            name = f"{case_dir.name}/{variant}" if variant else case_dir.name
            record = {"case": case_dir.name, "variant": variant, "impl": {}}
            passed = True
            for impl in ("asc", "pto"):
                argv = [str(case_dir / "run.py"), impl]
                if variant:
                    argv.append(variant)
                result = run_under_msprof(argv)
                cycles = (result.get("cycles") or {}).get("cycles")
                record["impl"][impl] = {
                    "passed": result["passed"],
                    "cycles": cycles,
                    "call_count": (result.get("cycles") or {}).get("call_count"),
                    "rows": (result.get("cycles") or {}).get("rows"),
                    "csv": (result.get("cycles") or {}).get("csv"),
                    "seconds": result.get("seconds"),
                    "out_dir": result.get("out_dir"),
                }
                passed = passed and result["passed"]
            asc_c = record["impl"]["asc"]["cycles"]
            pto_c = record["impl"]["pto"]["cycles"]
            ratio = _ratio(asc_c, pto_c) if passed else None
            record["ratio_pto_over_asc"] = ratio
            record["flag"] = _flag(ratio, passed)
            record["asc_vs_pto"] = _compare_dumps(case_dir.name, variant) if passed else "skipped"
            if record["flag"] == "FAIL" or record["asc_vs_pto"].startswith("mismatch"):
                failed += 1
            rows.append(record)
            print(f"[table] {name} flag={record['flag']} ratio={ratio} asc={asc_c} pto={pto_c} cmp={record['asc_vs_pto']}")

    out = ROOT / "perf" / "vf_cycles.json"
    out.write_text(json.dumps(rows, indent=2))
    _write_markdown(rows)
    print(f"wrote {out} failures={failed}")
    return 1 if failed else 0


def _write_markdown(rows: list[dict]) -> None:
    lines = [
        "# PTO:ASC simulated VF cycles",
        "",
        "Ratio = ASC VF cycles / PTO VF cycles. Above 1.0 means the PTO kernel spent fewer simulated vector-function cycles.",
        "",
        "43 kernels cover all 64 `with T.SimdVF():` statements in DeepInfer commit `553741557a57df44a46fb3ab72b6aafeaf945337`. One kernel can hold several statements, and nine kernels are a second arm of a statement already counted. The line-by-line sum is in `README.md` under Coverage.",
        "",
        "Cycles are the sum of the `cycles` column on `VF` / `VF_SIMD` rows in `core0.veccore0_instr_exe.csv`. `call_count` is not a multiplier. A case that fails its numeric check has no ratio.",
        "",
        "- `OK`: ratio in [0.8, 1.1]",
        "- `WIN`: PTO faster than 1.1x ASC",
        "- `WARN`: PTO slower than 0.8x ASC",
        "- `FAIL`: numeric check failed",
        "",
        "| case | ASC cycles | PTO cycles | PTO:ASC | flag | ASC vs PTO |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        name = row["case"] if not row["variant"] else f"{row['case']}/{row['variant']}"
        asc_c = row["impl"]["asc"]["cycles"]
        pto_c = row["impl"]["pto"]["cycles"]
        ratio = row["ratio_pto_over_asc"]
        ratio_s = f"{ratio:.3f}x" if ratio is not None else ""
        lines.append(
            f"| {name} | {asc_c if asc_c is not None else ''} | {pto_c if pto_c is not None else ''} | {ratio_s} | {row['flag']} | {row['asc_vs_pto']} |"
        )
    lines.append("")
    warns = [r for r in rows if r["flag"] == "WARN"]
    wins = [r for r in rows if r["flag"] == "WIN"]
    lines.append(f"Wins: {len(wins)}. Warnings: {len(warns)}. Failures: {sum(r['flag']=='FAIL' for r in rows)}.")
    lines.append("")
    if warns:
        lines.append("Warnings need another look at the VF body. The usual causes are an extra `vsel` that emulates a merging predicate, or a `vload`+`vbrc` pair the assembler did not fuse.")
        lines.append("")
    (ROOT / "PERF.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
