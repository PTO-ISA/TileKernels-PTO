"""store_kv_cache. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.norm.ref import reference_quant  # noqa: E402
from cases.store_kv_cache.asc import launch as launch_asc  # noqa: E402
from cases.store_kv_cache.pto import launch as launch_pto  # noqa: E402

CASE = "store_kv_cache"


def main() -> None:
    impl = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "row"
    g = torch.Generator().manual_seed(12)
    if variant == "rows":
        mod = __import__(f"cases.store_kv_cache.rows_{impl}", fromlist=["x"])
        x = (torch.randn(2, 128, generator=g) * 2).to(torch.bfloat16)
        q, sf = mod.launch(x)
        ref_sf = []
        for row in x.float():
            _, scale = reference_quant(row)
            ref_sf.append(scale)
        expect = torch.stack(ref_sf)
        finish(CASE, impl, sf, expect, exact=False, rtol=2e-2, atol=2e-2, variant=variant)
        return
    x = (torch.randn(128, generator=g) * 2).to(torch.bfloat16)
    q, sf = (launch_asc if impl == "asc" else launch_pto)(x)
    ref_q, ref_sf = reference_quant(x.float())
    finish(CASE, impl, sf, ref_sf, exact=False, rtol=2e-2, atol=2e-2, variant="sf")
    finish(CASE, impl, q.float(), ref_q.float(), exact=False, rtol=2e-1, atol=1e-1, variant="q")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
