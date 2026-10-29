"""fused_transform. python run.py asc|pto"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.fused_transform.asc import launch as launch_asc  # noqa: E402
from cases.fused_transform.pto import launch as launch_pto  # noqa: E402
from cases.fused_transform.ref import reference  # noqa: E402

CASE = "fused_transform"


def main() -> None:
    impl = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "norm"
    g = torch.Generator().manual_seed(11)
    if variant == "rope":
        mod = __import__(f"cases.fused_transform.extra_{impl}", fromlist=["x"])
        even = torch.randn(64, generator=g)
        odd = torch.randn(64, generator=g)
        cos = torch.randn(64, generator=g)
        sin = torch.randn(64, generator=g)
        y0, y1 = mod.launch_rope(even, odd, cos, sin)
        expect0 = even * cos - odd * sin
        expect1 = odd * cos + even * sin
        check = __import__("harness.compare", fromlist=["check_close"]).check_close
        check(y0, expect0, "rope0", rtol=1e-4, atol=1e-4)
        check(y1, expect1, "rope1", rtol=1e-4, atol=1e-4)
        finish(CASE, impl, y0, expect0, exact=False, rtol=1e-4, atol=1e-4, variant=variant)
        return
    if variant == "hadamard":
        mod = __import__(f"cases.fused_transform.extra_{impl}", fromlist=["x"])
        x = torch.randn(64, generator=g)
        got = mod.launch_hadamard(x)
        values = x.clone()
        lane = torch.arange(64)
        for stage in range(6):
            distance = 1 << stage
            partner = values[lane ^ distance]
            low = (lane & distance) == 0
            values = torch.where(low, values + partner, partner - values)
        check = __import__("harness.compare", fromlist=["check_close"]).check_close
        check(got, values, "hadamard", rtol=1e-4, atol=1e-4)
        finish(CASE, impl, got, values, exact=False, rtol=1e-4, atol=1e-4, variant=variant)
        return
    if variant == "quant":
        mod = __import__(f"cases.fused_transform.quant_{impl}", fromlist=["x"])
        from cases.norm.ref import reference_quant

        x = torch.randn(128, generator=g) * 3
        q, sf = mod.launch(x)
        ref_q, ref_sf = reference_quant(x)
        check = __import__("harness.compare", fromlist=["check_close"]).check_close
        check(sf, ref_sf, "sf", rtol=1e-4, atol=1e-4)
        check(q.float(), ref_q.float(), "q", rtol=1e-1, atol=5e-2)
        finish(CASE, impl, sf, ref_sf, exact=False, rtol=1e-4, atol=1e-4, variant=variant)
        return
    if variant == "lut":
        mod = __import__(f"cases.fused_transform.lut_{impl}", fromlist=["x"])
        import struct

        table = [struct.unpack("<I", struct.pack("<f", 8.0 / (8 + m)))[0] for m in range(8)]
        sf = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0] * 64)
        got = mod.launch(sf)

        def apply(value: float) -> float:
            word = struct.unpack("<I", struct.pack("<f", float(value)))[0]
            exponent = (word >> 23) & 255
            if exponent == 255:
                return value
            mantissa = (word >> 20) & 7
            delta = (((127 - exponent) & 0xFFFFFFFF) << 23) & 0xFFFFFFFF
            bits = (table[mantissa] + delta) & 0xFFFFFFFF
            return struct.unpack("<f", struct.pack("<I", bits))[0]

        expect = torch.tensor([apply(v) for v in sf.tolist()])
        check = __import__("harness.compare", fromlist=["check_close"]).check_close
        check(got, expect, "lut", rtol=0, atol=0)
        finish(CASE, impl, got, expect, variant=variant)
        return
    x = torch.randn(128, generator=g, dtype=torch.bfloat16)
    weight = (torch.rand(128, generator=g) + 0.25).to(torch.bfloat16)
    got = (launch_asc if impl == "asc" else launch_pto)(x, weight)
    finish(CASE, impl, got.float(), reference(x, weight).float(), exact=False, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
