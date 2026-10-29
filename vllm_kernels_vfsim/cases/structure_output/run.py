"""structure_output bitmask VF. ``python run.py asc|pto``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.compare import check_exact  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.structure_output.asc import launch as launch_asc  # noqa: E402
from cases.structure_output.pto import launch as launch_pto  # noqa: E402
from cases.structure_output.ref import reference  # noqa: E402

CASE = "structure_output"


def _inputs():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(128, generator=g)
    bitmask = torch.randint(-(2**31), 2**31 - 1, (4,), generator=g, dtype=torch.int32)
    bitmask[0] = 0x0000000F
    return logits, bitmask


def main() -> None:
    impl = sys.argv[1] if len(sys.argv) > 1 else "pto"
    logits, bitmask = _inputs()
    ref = reference(logits, bitmask)
    got = (launch_asc if impl == "asc" else launch_pto)(logits, bitmask)
    check_exact(got, ref, f"{CASE}/{impl}")
    dump = ROOT / "perf" / "dumps"
    dump.mkdir(parents=True, exist_ok=True)
    torch.save(got.detach().cpu(), dump / f"{CASE}_{impl}.pt")
    print(f"[status] variant={CASE}/{impl} result=PASS shape=128")
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
