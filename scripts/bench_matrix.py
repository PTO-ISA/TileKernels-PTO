"""Correctness-check then msprof-bench ASC vs PTO cases.

One family per process. Example:
  env BENCH_DEVICES=1 BENCH_REP=10 bash scripts/env_npu.sh python scripts/bench_matrix.py --family swa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cases_index import CASES as INDEX_CASES
from cases_numeric import CASES as NUMERIC_CASES
from cases_sample import CASES as SAMPLE_CASES
from harness import run_cases, setup_device

ALL = [*INDEX_CASES, *NUMERIC_CASES, *SAMPLE_CASES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--case", action="append", default=[], help="family/name, repeatable")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for case in ALL:
            print(f"{case.family}/{case.name}")
        return
    selected = ALL
    if args.family:
        wanted = set(args.family)
        selected = [case for case in selected if case.family in wanted]
        missing = wanted - {case.family for case in selected}
        if missing:
            raise SystemExit(f"unknown family {sorted(missing)}")
    if args.case:
        wanted_cases = set(args.case)
        known = {f"{case.family}/{case.name}" for case in ALL}
        missing = wanted_cases - known
        if missing:
            raise SystemExit(f"unknown case {sorted(missing)}")
        selected = [case for case in selected if f"{case.family}/{case.name}" in wanted_cases]
    dev = setup_device()
    import torch

    print(f"device={dev} name={torch.npu.get_device_name(dev)} cases={len(selected)}", flush=True)
    rows = run_cases(selected, dev)
    failed = [row for row in rows if not row.get("correct")]
    print(f"DONE {len(rows) - len(failed)}/{len(rows)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
