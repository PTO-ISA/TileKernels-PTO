"""norm variants: rms, unpack, quant, layout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.norm import asc, pto  # noqa: E402
from cases.norm import quant_asc, quant_pto  # noqa: E402
from cases.norm.ref import reference_layout, reference_quant, reference_rms  # noqa: E402

CASE = "norm"


def main() -> None:
    impl, variant = sys.argv[1], sys.argv[2]
    g = torch.Generator().manual_seed(7)
    if variant == "rms":
        x = torch.randn(128, generator=g)
        weight = torch.rand(128, generator=g) + 0.25
        got = (asc if impl == "asc" else pto).launch_rms(x, weight)
        finish(CASE, impl, got, reference_rms(x, weight), exact=False, rtol=1e-4, atol=1e-4, variant=variant)
    elif variant == "unpack":
        src = torch.randn(128, generator=g, dtype=torch.bfloat16)
        got = (asc if impl == "asc" else pto).launch_unpack(src)
        finish(CASE, impl, got, src.float(), exact=False, rtol=1e-2, atol=1e-2, variant=variant)
    elif variant == "quant":
        x = torch.randn(128, generator=g) * 3
        q, sf = (quant_asc if impl == "asc" else quant_pto).launch_quant(x)
        ref_q, ref_sf = reference_quant(x)
        finish(CASE, impl, sf, ref_sf, exact=False, rtol=1e-4, atol=1e-4, variant=variant + "_sf")
        finish(CASE, impl, q.float(), ref_q.float(), exact=False, rtol=1e-1, atol=5e-2, variant=variant + "_q")
    elif variant == "rstd":
        x = torch.randn(128, generator=g)
        mod_r = __import__(f"cases.norm.rstd_{impl}", fromlist=["x"])
        got = mod_r.launch(x)
        var = (x * x).sum() / 128 + 1e-5
        expect = torch.rsqrt(var).reshape(1)
        finish(CASE, impl, got, expect, exact=False, rtol=1e-3, atol=1e-3, variant=variant)
    elif variant == "bf16":
        x = torch.randn(256, generator=g, dtype=torch.bfloat16)
        mod_b = __import__(f"cases.norm.bf16_{impl}", fromlist=["x"])
        got_amax, got_q = mod_b.launch(x)
        expect = x.float().abs().reshape(8, 32).amax(dim=1)
        finish(CASE, impl, got_amax, expect, exact=False, rtol=2e-2, atol=2e-2, variant=variant)
        scales = torch.clamp(expect, min=1e-4) / 448.0
        ref_q = (x.float().reshape(8, 32) / scales[:, None]).reshape(256).to(torch.float8_e4m3fn)
        finish(CASE, impl, got_q.float(), ref_q.float(), exact=False, rtol=2e-1, atol=1e-1, variant=variant + "_q")
    else:
        table = torch.arange(128, dtype=torch.int32).to(torch.int16).view(torch.uint16)
        idx, got = (quant_asc if impl == "asc" else quant_pto).launch_layout(table)
        ref_idx, ref_got = reference_layout(table)
        got_idx = idx.view(torch.int16).to(torch.int32) & 0xFFFF
        got_val = got.view(torch.int16).to(torch.int32) & 0xFFFF
        finish(CASE, impl, got_idx, ref_idx, variant=variant + "_idx")
        finish(CASE, impl, got_val, ref_got, variant=variant + "_gather")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
