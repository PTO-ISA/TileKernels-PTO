"""sampler. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.compare import check_close, check_exact  # noqa: E402
from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.sampler.asc import launch as launch_asc  # noqa: E402
from cases.sampler.pto import launch as launch_pto  # noqa: E402
from cases.sampler.ref import reference  # noqa: E402

CASE = "sampler"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(9)
    keys = torch.arange(64, dtype=torch.int32)
    values = torch.randn(64, generator=g)
    src = torch.randn(64, generator=g)
    indices = torch.arange(63, -1, -1, dtype=torch.int32)
    got = (launch_asc if impl == "asc" else launch_pto)(keys, values, 7, src, 1, indices)
    ref = reference(keys, values, 7, src, 1, indices)
    check_close(got[0], ref[0], "selected", rtol=0, atol=0)
    check_exact(got[1], ref[1], "finalized")
    check_close(got[2], ref[2], "gathered", rtol=0, atol=0)
    finish(CASE, impl, got[1], ref[1])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
