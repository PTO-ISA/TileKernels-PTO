"""candidate_blocks. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.compare import check_exact  # noqa: E402
from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.candidate_blocks.asc import launch as launch_asc  # noqa: E402
from cases.candidate_blocks.pto import launch as launch_pto  # noqa: E402
from cases.candidate_blocks.ref import reference  # noqa: E402

CASE = "candidate_blocks"


def main() -> None:
    impl = sys.argv[1]
    lengths = torch.tensor([0, 100, 16384, 16385] + [0] * 60, dtype=torch.int32)
    indices = torch.arange(-2, 62, dtype=torch.int32)
    candidates = torch.zeros(64, dtype=torch.int32)
    candidates[:4] = torch.tensor([3, 1, -1, 8], dtype=torch.int32)
    row_len = 40
    got = (launch_asc if impl == "asc" else launch_pto)(lengths, indices, candidates, row_len)
    ref = reference(lengths, indices, candidates, row_len)
    names = ("counts", "prefix", "restored", "total", "compact")
    for name, left, right in zip(names, got, ref):
        check_exact(left, right, f"{CASE}/{impl}/{name}")
    finish(CASE, impl, got[2], ref[2])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
