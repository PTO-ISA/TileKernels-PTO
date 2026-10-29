"""kv_index_cat: one SimdVF. Run ``python run.py asc`` or ``python run.py pto``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.compare import check_exact  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.kv_index_cat.asc import launch as launch_asc  # noqa: E402
from cases.kv_index_cat.pto import launch as launch_pto  # noqa: E402
from cases.kv_index_cat.ref import reference  # noqa: E402

CASE = "kv_index_cat"


def _inputs():
    g = torch.Generator().manual_seed(0)
    src = torch.randint(-3, 30, (2, 128), generator=g, dtype=torch.int32)
    src[:, ::17] = -1
    return src, 9


def main() -> None:
    impl = sys.argv[1] if len(sys.argv) > 1 else "pto"
    src, offset = _inputs()
    ref = reference(src, offset)
    got = (launch_asc if impl == "asc" else launch_pto)(src, offset)
    check_exact(got, ref, f"{CASE}/{impl}")
    dump = ROOT / "perf" / "dumps"
    dump.mkdir(parents=True, exist_ok=True)
    torch.save(got.detach().cpu(), dump / f"{CASE}_{impl}.pt")
    print(f"[status] variant={CASE}/{impl} result=PASS shape=2x128")
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
