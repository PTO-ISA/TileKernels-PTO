"""Sum simulated VF launch cycles from an msprof op simulator tree.

``cycles`` on a row is already the row total. Do not multiply by ``call_count``.
Key on ``instr`` (``VF`` or ``VF_SIMD``), never on ``pipe``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SIMD_NAMES = {"vf", "vf_simd"}


def _instr_csv(out_dir: Path) -> Path | None:
    hits = sorted(out_dir.rglob("*instr_exe*.csv"))
    for path in hits:
        if "core0.veccore0" in path.name or "core0.veccore0" in str(path.parent):
            return path
    return hits[0] if hits else None


def sum_vf_cycles(out_dir: Path) -> dict:
    res = {
        "csv": None,
        "kind": None,
        "instr": None,
        "rows": 0,
        "call_count": 0,
        "cycles": None,
        "us": None,
    }
    path = _instr_csv(Path(out_dir))
    if path is None:
        return res
    res["csv"] = str(path)
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("instr") or "").strip()
            seen.add(name)
            if name.lower() not in SIMD_NAMES:
                continue
            try:
                bucket = grouped[name]
                bucket[0] += float(row.get("cycles") or 0)
                bucket[1] += float(row.get("running_time(us)") or 0)
                bucket[2] += int(float(row.get("call_count") or 0))
                bucket[3] += 1
            except ValueError:
                continue
    res["seen"] = sorted(seen)
    if not grouped:
        return res
    res.update(
        kind="SIMD",
        instr=",".join(sorted(grouped)),
        cycles=sum(v[0] for v in grouped.values()),
        us=sum(v[1] for v in grouped.values()),
        call_count=sum(v[2] for v in grouped.values()),
        rows=sum(v[3] for v in grouped.values()),
    )
    return res
