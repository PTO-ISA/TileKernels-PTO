"""schedule_mtp_varlen ranking VF."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.schedule_mtp_varlen.asc import launch as launch_asc  # noqa: E402
from cases.schedule_mtp_varlen.pto import launch as launch_pto  # noqa: E402
from cases.schedule_mtp_varlen.ref import reference  # noqa: E402

CASE = "schedule_mtp_varlen"


def main() -> None:
    impl = sys.argv[1]
    rates = torch.full((64,), float("-inf"))
    rates[:8] = torch.tensor([0.9, 0.4, 0.4, 0.1, 0.8, 0.4, 0.2, 0.05])
    extra = 3
    ref = reference(rates, extra)
    got = (launch_asc if impl == "asc" else launch_pto)(rates, extra)
    finish(CASE, impl, got, ref)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
