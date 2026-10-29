"""random_sampling. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.random_sampling.asc import launch as launch_asc  # noqa: E402
from cases.random_sampling.pto import launch as launch_pto  # noqa: E402
from cases.random_sampling.ref import reference  # noqa: E402

CASE = "random_sampling"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(8)
    probs = torch.rand(128, generator=g)
    probs = probs / probs.sum()
    got = (launch_asc if impl == "asc" else launch_pto)(probs)
    finish(CASE, impl, got, reference(probs))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
