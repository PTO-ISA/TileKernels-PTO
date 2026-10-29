"""load_paged_kv_cache. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.load_paged_kv_cache.asc import launch as launch_asc  # noqa: E402
from cases.load_paged_kv_cache.pto import launch as launch_pto  # noqa: E402
from cases.load_paged_kv_cache.ref import reference  # noqa: E402

CASE = "load_paged_kv_cache"


def main() -> None:
    impl = sys.argv[1]
    raw = torch.full((64,), 0x38, dtype=torch.uint8)
    packed = 127 | (128 << 8)
    got = (launch_asc if impl == "asc" else launch_pto)(raw, packed)
    ref = reference(raw, packed)
    finish(CASE, impl, got.float(), ref.float(), exact=False, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
