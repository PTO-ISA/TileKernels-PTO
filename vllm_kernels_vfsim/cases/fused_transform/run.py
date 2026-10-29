"""fused_transform. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.fused_transform.asc import launch as launch_asc  # noqa: E402
from cases.fused_transform.pto import launch as launch_pto  # noqa: E402
from cases.fused_transform.ref import reference  # noqa: E402

CASE = "fused_transform"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(11)
    x = torch.randn(128, generator=g, dtype=torch.bfloat16)
    weight = (torch.rand(128, generator=g) + 0.25).to(torch.bfloat16)
    got = (launch_asc if impl == "asc" else launch_pto)(x, weight)
    finish(CASE, impl, got.float(), reference(x, weight).float(), exact=False, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
