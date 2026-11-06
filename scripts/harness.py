"""ASC vs PTO correctness and msprof timing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from tilelang.profiler import do_bench

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class BenchCase:
    family: str
    name: str
    # Returns (asc_launch, pto_launch, compared, nbytes).
    # compared is a list of (label, asc_tensor, pto_tensor) after one launch each.
    prepare: Callable[[], tuple]
    exact: bool = True
    rtol: float = 2e-2
    atol: float = 2e-2
    notes: str = ""


def device_id() -> int:
    return int(os.environ.get("BENCH_DEVICES", "1"))


def setup_device() -> int:
    import torch_npu  # noqa: F401

    dev = device_id()
    torch.npu.set_device(dev)
    return dev


def _close(a: torch.Tensor, b: torch.Tensor, exact: bool, rtol: float, atol: float) -> tuple[bool, str]:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    ac, bc = a.detach().cpu(), b.detach().cpu()
    if exact or not ac.is_floating_point():
        if torch.equal(ac, bc):
            return True, ""
        n = int((ac != bc).sum())
        return False, f"mismatch {n}"
    if torch.allclose(ac, bc, rtol=rtol, atol=atol, equal_nan=True):
        return True, ""
    diff = (ac.float() - bc.float()).abs()
    return False, f"max_abs {float(diff.max())}"


def check_case(case: BenchCase) -> tuple[Callable, Callable, int, str]:
    asc_fn, pto_fn, compared, nbytes = case.prepare()
    notes = []
    for label, a, b in compared:
        ok, msg = _close(a, b, case.exact, case.rtol, case.atol)
        if not ok:
            raise RuntimeError(f"{case.family}/{case.name}/{label}: {msg}")
        notes.append(label)
    return asc_fn, pto_fn, nbytes, ",".join(notes)


def time_launch(fn: Callable, dev: int) -> float:
    ms = do_bench(
        fn,
        backend="msprof",
        warmup=0,
        rep=0,
        _n_warmup=5,
        _n_repeat=int(os.environ.get("BENCH_REP", "20")),
        return_mode="median",
        device=torch.device("npu", dev),
        cache_size=256,
    )
    return float(ms) * 1000.0


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run_cases(cases: list[BenchCase], dev: int) -> list[dict]:
    rows = []
    for case in cases:
        print(f"CASE {case.family}/{case.name}", flush=True)
        row = {
            "family": case.family,
            "name": case.name,
            "device": dev,
            "notes": case.notes,
        }
        try:
            asc_fn, pto_fn, nbytes, compared = check_case(case)
            row["correct"] = True
            row["compared"] = compared
            row["io_bytes"] = nbytes
            row["asc_us"] = time_launch(asc_fn, dev)
            row["pto_us"] = time_launch(pto_fn, dev)
            row["ratio"] = row["asc_us"] / row["pto_us"] if row["pto_us"] else None
            if nbytes and row["asc_us"]:
                row["asc_gbs"] = nbytes / row["asc_us"] / 1e3
                row["pto_gbs"] = nbytes / row["pto_us"] / 1e3
            print(
                f"  OK asc={row['asc_us']:.2f}us pto={row['pto_us']:.2f}us ratio={row['ratio']:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- record and continue to the next kernel
            row["correct"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ERR {row['error']}", flush=True)
        append_jsonl(DATA / "asc" / f"{case.family}.jsonl", row)
        append_jsonl(DATA / "pto" / f"{case.family}.jsonl", row)
        rows.append(row)
    return rows
