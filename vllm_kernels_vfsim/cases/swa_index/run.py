"""swa_index. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.swa_index.asc import launch as launch_asc  # noqa: E402
from cases.swa_index.pto import launch as launch_pto  # noqa: E402
from cases.swa_index.ref import reference  # noqa: E402

CASE = "swa_index"


def main() -> None:
    impl = sys.argv[1]
    pivots = torch.tensor([[3, 40, 100, 50], [0, 0, 7, 200]], dtype=torch.int32)
    ref = reference(pivots)
    got = (launch_asc if impl == "asc" else launch_pto)(pivots)
    finish(CASE, impl, got, ref)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
