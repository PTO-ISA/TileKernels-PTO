"""Shared end-of-run checks and dump paths."""

from __future__ import annotations

from pathlib import Path

import torch

from .compare import check_close, check_exact

ROOT = Path(__file__).resolve().parents[1]


def finish(case: str, impl: str, got: torch.Tensor, ref: torch.Tensor, *, exact: bool = True, rtol: float = 1e-4, atol: float = 1e-4, variant: str = "") -> None:
    name = f"{case}/{variant}/{impl}" if variant else f"{case}/{impl}"
    if exact:
        check_exact(got, ref, name)
    else:
        check_close(got, ref, name, rtol=rtol, atol=atol)
    dump = ROOT / "perf" / "dumps"
    dump.mkdir(parents=True, exist_ok=True)
    tag = f"{case}_{variant}_{impl}" if variant else f"{case}_{impl}"
    torch.save(got.detach().cpu(), dump / f"{tag}.pt")
    print(f"[status] variant={name} result=PASS")
    print("PASS")
