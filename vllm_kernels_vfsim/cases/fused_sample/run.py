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
    variant = sys.argv[2] if len(sys.argv) > 2 else "lse"
    g = torch.Generator().manual_seed(10)
    if variant == "lse":
        logits = torch.randn(128, generator=g)
        logits[20] = logits.max() + 0.5
        logits[90] = logits[20]
        got = (launch_asc if impl == "asc" else launch_pto)(logits)
        ref = reference(logits)
        check_close(got[0], ref[0], "lse", rtol=1e-4, atol=1e-4)
        check_exact(got[1], ref[1], "argmax")
        finish(CASE, impl, got[1], ref[1], variant=variant)
        return
    mod = __import__(f"cases.fused_sample.stages_{impl}", fromlist=["x"])
    if variant == "search":
        probs = torch.rand(128, generator=g)
        quartiles = torch.tensor([0.2, 0.5, 0.8])
        got = mod.launch_search(probs, quartiles)
        sums = []
        counts = []
        for q in quartiles.tolist():
            keep = probs >= q
            sums.append(float(probs[keep].sum()))
            counts.append(int(keep.sum()))
        check_close(got[0], torch.tensor(sums), "sums", rtol=1e-4, atol=1e-4)
        check_exact(got[1], torch.tensor(counts, dtype=torch.int32), "counts")
        finish(CASE, impl, got[1], torch.tensor(counts, dtype=torch.int32), variant=variant)
    elif variant == "filter":
        logits = torch.randn(128, generator=g)
        inv = 0.8
        shifted = logits - logits.max()
        lse0 = float(-torch.log(torch.exp(shifted * inv).sum()))
        lse1 = float(-logits.max())
        weights, total, count = mod.launch_filter(logits, inv, lse0, lse1)
        prob = torch.exp((logits + lse1) * inv + lse0)
        expect = torch.where(prob >= 2.0 ** -32, prob, torch.zeros_like(prob))
        check_close(weights, expect, "weights", rtol=1e-4, atol=1e-4)
        check_close(total, expect.sum().reshape(1), "total", rtol=1e-4, atol=1e-4)
        check_exact(count, torch.tensor([int((expect > 0).sum())], dtype=torch.int32), "count")
        finish(CASE, impl, total, expect.sum().reshape(1), exact=False, rtol=1e-4, atol=1e-4, variant=variant)
    elif variant == "quartile":
        import struct

        def _bits(value: float) -> int:
            return struct.unpack("<i", struct.pack("<f", float(value)))[0]

        def _from_bits(value: int) -> float:
            return struct.unpack("<f", struct.pack("<i", int(value) & 0xFFFFFFFF))[0]

        minimum = 2.0 ** -32
        upper = 1.0 + 1e-3
        # 1.5 ulps of MIN_PROB is exactly representable and stays above the floor used as the divisor.
        target = 1.5 * minimum
        mass = torch.tensor([1.0, 0.0, 1e-9, 1e-10, 0.0, 0.0])
        counts = torch.tensor([20, 0, 40, 10, 4, 1], dtype=torch.int32)
        q, new_lo, new_hi, n_keep = mod.launch_quartile(torch.tensor([minimum, upper]), mass, counts, target, 8)
        left, right = _bits(minimum), _bits(upper)
        width = right - left
        expect_q = torch.tensor(
            [
                _from_bits(left + (width >> 2)),
                _from_bits(left + (width >> 1)),
                _from_bits(right - ((width + 3) >> 2)),
            ]
        )
        # Only sums[2] (1e-9) clears the tiny target, so the window is the first quartile.
        check_close(q, expect_q, "quartiles", rtol=0, atol=0)
        check_close(new_lo, expect_q[0:1], "lo", rtol=0, atol=0)
        check_close(new_hi, expect_q[1:2], "hi", rtol=0, atol=0)
        check_exact(n_keep, torch.tensor([2], dtype=torch.int32), "n_keep")
        finish(CASE, impl, n_keep, torch.tensor([2], dtype=torch.int32), variant=variant)
    elif variant == "mask":
        mask_mod = __import__(f"cases.fused_sample.mask_{impl}", fromlist=["x"])
        probs = torch.rand(128, generator=g)
        logits = torch.randn(128, generator=g)
        boundary = float(probs.median())
        inv, lse0, lse1 = 1.0, 0.0, 0.0
        got = mask_mod.launch_mask(probs, logits, boundary, inv, lse0, lse1)
        expect_mask = (probs >= boundary).to(torch.int32)
        check_exact(got[0], expect_mask, "mask")
        kept = torch.where(probs >= boundary, probs, torch.zeros_like(probs))
        check_close(got[1], torch.log(kept.sum()).reshape(1), "log_mass", rtol=1e-4, atol=1e-4)
        finish(CASE, impl, got[0], expect_mask, variant=variant)
    elif variant == "cdf":
        mask_mod = __import__(f"cases.fused_sample.mask_{impl}", fromlist=["x"])
        probs = torch.rand(128, generator=g) * 0.4 + 0.05
        logprobs = torch.log(probs)
        got = mask_mod.launch_cdf(logprobs)
        scale = float(1 << 31)
        acc = torch.zeros(64, dtype=torch.int32)
        folded = torch.floor(torch.exp(logprobs).float() * scale).to(torch.int32)
        acc = folded[:64] + folded[64:]
        def scan(values):
            out = values.clone()
            for step in (1, 2, 4, 8, 16, 32):
                nxt = out.clone()
                nxt[step:] = out[step:] + out[:-step]
                out = nxt
            return out
        once = scan(acc)
        twice = scan(once)
        # vexp is not libm, so the quantized mass can move by a few ulps of 2^31.
        # The second scan must be exactly the inclusive scan of the first.
        check_exact(got[1], scan(got[0]), "scan-identity")
        delta = (got[0].to(torch.int64) - once.to(torch.int64)).abs().max().item()
        if delta > 4096:
            raise AssertionError(f"scan1 off by {delta}: got {got[0][:4].tolist()} ref {once[:4].tolist()}")
        check_exact(got[1], twice, "scan2") if delta == 0 else None
        finish(CASE, impl, got[1], scan(got[0]), variant=variant)
    else:
        raise SystemExit(f"unknown variant {variant}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
