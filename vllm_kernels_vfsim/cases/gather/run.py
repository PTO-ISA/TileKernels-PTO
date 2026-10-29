"""gather zero-fill VF plus the scalar indexed copies that give it a result."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.gather.asc import launch as launch_asc  # noqa: E402
from cases.gather.pto import launch as launch_pto  # noqa: E402
from cases.gather.ref import reference  # noqa: E402

CASE = "gather"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(2)
    x = torch.randint(-20, 20, (4, 128), generator=g, dtype=torch.int8)
    index = torch.tensor([2, -1, 0, 9], dtype=torch.int32)
    ref = reference(x, index)
    got = (launch_asc if impl == "asc" else launch_pto)(x, index)
    finish(CASE, impl, got, ref)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
