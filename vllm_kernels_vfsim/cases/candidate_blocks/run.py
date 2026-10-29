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
    variant = sys.argv[2] if len(sys.argv) > 2 else "main"
    if variant == "map":
        mod = __import__(f"cases.candidate_blocks.extra_{impl}", fromlist=["x"])
        ends = torch.zeros(64, dtype=torch.int32)
        ends[:4] = torch.tensor([2, 5, 7, 8])
        lengths = torch.zeros(64, dtype=torch.int32)
        lengths[:4] = torch.tensor([10, 20, 30, 40])
        tasks, _lens = mod.launch_map(ends, lengths)
        prefix = ends[:4].tolist()
        expect_task = []
        expect_len = []
        for pos in range(8):
            row = 0
            for bit in range(6):
                nxt = row + (1 << (5 - bit))
                if nxt < 4 and prefix[nxt - 1] <= pos:
                    row = nxt
            row_start = 0 if row == 0 else prefix[row - 1]
            expect_task.append(row * 8 + (pos - row_start))
            expect_len.append(int(lengths[row]))
        # vintlv packs (task, length) into adjacent lanes of the low register.
        check_exact(tasks[0:16:2], torch.tensor(expect_task, dtype=torch.int32), "tasks")
        check_exact(tasks[1:16:2], torch.tensor(expect_len, dtype=torch.int32), "lengths")
        finish(CASE, impl, tasks[0:16:2], torch.tensor(expect_task, dtype=torch.int32), variant=variant)
        return
    if variant == "max":
        mod = __import__(f"cases.candidate_blocks.extra_{impl}", fromlist=["x"])
        g = torch.Generator().manual_seed(3)
        logits = torch.randn(64, generator=g, dtype=torch.bfloat16)
        got = mod.launch_max(logits)
        expect = logits.float().reshape(8, 8).amax(dim=1)
        check_close = __import__("harness.compare", fromlist=["check_close"]).check_close
        check_close(got, expect, "max", rtol=1e-2, atol=1e-2)
        finish(CASE, impl, got, expect, exact=False, rtol=1e-2, atol=1e-2, variant=variant)
        return
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
