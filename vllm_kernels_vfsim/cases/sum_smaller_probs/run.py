"""sum_smaller_probs. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.sum_smaller_probs.asc import SAMPLE, launch as launch_asc  # noqa: E402
from cases.sum_smaller_probs.pto import launch as launch_pto  # noqa: E402
from cases.sum_smaller_probs.ref import reference  # noqa: E402

CASE = "sum_smaller_probs"


def main() -> None:
    impl = sys.argv[1]
    g = torch.Generator().manual_seed(3)
    logprobs = torch.randn(128, generator=g)
    logprobs[70] = logprobs[64]
    logprobs[68] = logprobs[70]
    pivot = float(logprobs[SAMPLE])
    ref = reference(logprobs)
    got = (launch_asc if impl == "asc" else launch_pto)(logprobs, pivot)
    finish(CASE, impl, got, ref, exact=False, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
