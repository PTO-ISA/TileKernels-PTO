"""dflash. python run.py asc|pto dot|mean"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.dflash import asc, pto  # noqa: E402
from cases.dflash.ref import reference_dot, reference_mean  # noqa: E402

CASE = "dflash"


def main() -> None:
    impl, variant = sys.argv[1], sys.argv[2]
    g = torch.Generator().manual_seed(5)
    if variant == "dot":
        x = torch.randn(4, 128, generator=g, dtype=torch.bfloat16)
        weight = torch.randn(128, generator=g, dtype=torch.bfloat16)
        ref = reference_dot(x, weight)
        got = (asc if impl == "asc" else pto).launch_dot(x, weight)
        finish(CASE, impl, got, ref, exact=False, rtol=1e-2, atol=2e-2, variant=variant)
    else:
        x = torch.randn(2, 128, generator=g, dtype=torch.bfloat16)
        ref = reference_mean(x)
        got = (asc if impl == "asc" else pto).launch_mean(x)
        finish(CASE, impl, got.float(), ref.float(), exact=False, rtol=1e-2, atol=2e-2, variant=variant)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
