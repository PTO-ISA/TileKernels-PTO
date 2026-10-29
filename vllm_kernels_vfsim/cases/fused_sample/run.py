"""fused_sample. python run.py asc|pto"""

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

from cases.fused_sample.asc import launch as launch_asc  # noqa: E402
from cases.fused_sample.pto import launch as launch_pto  # noqa: E402
from cases.fused_sample.ref import reference  # noqa: E402

CASE = "fused_sample"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(10)
    logits = torch.randn(128, generator=g)
    logits[20] = logits.max() + 0.5
    logits[90] = logits[20]
    got = (launch_asc if impl == "asc" else launch_pto)(logits)
    ref = reference(logits)
    check_close(got[0], ref[0], "lse", rtol=1e-4, atol=1e-4)
    check_exact(got[1], ref[1], "argmax")
    finish(CASE, impl, got[1], ref[1])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
