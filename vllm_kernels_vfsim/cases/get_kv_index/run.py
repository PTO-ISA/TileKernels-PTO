"""get_kv_index. python run.py asc|pto blocks|identity|dflash"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.get_kv_index import asc, pto  # noqa: E402
from cases.get_kv_index import ref  # noqa: E402

CASE = "get_kv_index"


def main() -> None:
    impl, variant = sys.argv[1], sys.argv[2]
    mod = asc if impl == "asc" else pto
    if variant == "blocks":
        table = torch.tensor([4, 9, 1, 7], dtype=torch.int32)
        got = mod.launch_blocks(table, 40)
        expect = ref.reference_blocks(table, 40)
    elif variant == "identity":
        got = mod.launch_identity(5, 20)
        expect = ref.reference_identity(5, 20)
    else:
        got = mod.launch_dflash(-3, 100, 40)
        expect = ref.reference_dflash(-3, 100, 40)
    finish(CASE, impl, got, expect, variant=variant)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
