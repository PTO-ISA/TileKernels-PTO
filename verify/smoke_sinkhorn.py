"""Compile smoke test for the VMI-ported mhc sinkhorn kernels.

Compiles and runs mhc_sinkhorn_fwd_vmi / mhc_sinkhorn_bwd_vmi smallest-first
on NPU 0, checks numerical sanity against the torch reference
(sinkhorn_normalize_ref + autograd), and prints PASS/FAIL per kernel.
Covers the repo test's regimes: random logits, and the large same-sign-logits
cases (offset +-100, eps 1e-2) from test_sinkhorn_large_same_sign_logits.
Failures print the full traceback so the tilelang temp dir hint (kernel.pto /
ptodsl) can be collected for a rule-4 report.

num_tokens must be a multiple of the bwd token_block_size (8) -- the ASC
original has no tail handling (fixed-shape block copies), so only multiples
of 8 are ASC-legal here.
"""

import sys
import traceback

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401  registers the npu backend

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.sinkhorn_vmi import mhc_sinkhorn_bwd_vmi, mhc_sinkhorn_fwd_vmi
from tile_kernels.torch.mhc import sinkhorn_normalize_ref

DEV = "npu"
NVC = get_num_vec_cores()
REPEAT, EPS = 10, 1e-6


def gen_random(n):
    return torch.randn((n, 4, 4), dtype=torch.float32, device=DEV)


def gen_large_logits(n, offset, eps):
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0, 0.0],
            [1.5, 0.5, 3.5, 2.5],
            [2.5, 3.5, 0.5, 1.5],
        ],
        dtype=torch.float32,
        device=DEV,
    )
    return base.repeat(n, 1, 1) + offset


def run_fwd(x, eps):
    out = torch.empty_like(x)
    mhc_sinkhorn_fwd_vmi(x, out, REPEAT, eps, token_block_size=4, num_vec_cores=NVC)
    torch.npu.synchronize()

    ref = sinkhorn_normalize_ref(x, REPEAT, eps)
    torch.testing.assert_close(out, ref)
    return f"n={x.shape[0]} eps={eps} max|diff|={(out - ref).abs().max().item():.3e} bitwise={torch.equal(out, ref)}"


def run_bwd(o_grad, x, eps):
    grad_input = torch.empty_like(x)
    mhc_sinkhorn_bwd_vmi(o_grad, x, grad_input, REPEAT, eps, token_block_size=8, num_vec_cores=NVC)
    torch.npu.synchronize()

    x_ref = x.clone().requires_grad_()
    out_ref = sinkhorn_normalize_ref(x_ref, REPEAT, eps)
    torch.autograd.backward([out_ref], [o_grad])
    torch.testing.assert_close(grad_input, x_ref.grad)
    assert torch.isfinite(grad_input).all()
    return f"n={x.shape[0]} eps={eps} max|diff|={(grad_input - x_ref.grad).abs().max().item():.3e}"


CASES = [
    ("fwd n=8 random", run_fwd, (gen_random(8), EPS)),
    ("fwd n=64 random", run_fwd, (gen_random(64), EPS)),
    ("fwd n=1024 random", run_fwd, (gen_random(1024), EPS)),
    ("fwd n=8 logits+100", run_fwd, (gen_large_logits(8, 100.0, EPS), EPS)),
    ("fwd n=8 logits-100 eps=1e-2", run_fwd, (gen_large_logits(8, -100.0, 1e-2), 1e-2)),
    ("bwd n=8 random", run_bwd, (gen_random(8), gen_random(8), EPS)),
    ("bwd n=64 random", run_bwd, (gen_random(64), gen_random(64), EPS)),
    ("bwd n=1024 random", run_bwd, (gen_random(1024), gen_random(1024), EPS)),
    ("bwd n=8 logits+100", run_bwd, (gen_random(8), gen_large_logits(8, 100.0, EPS), EPS)),
    ("bwd n=8 logits-100 eps=1e-2", run_bwd, (gen_random(8), gen_large_logits(8, -100.0, 1e-2), 1e-2)),
]

results = []
for name, fn, args in CASES:
    try:
        detail = fn(*args)
        results.append((name, True, detail))
        print(f"PASS {name}: {detail}", flush=True)
    except Exception:
        results.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}:\n{traceback.format_exc()}", flush=True)

failed = [name for name, ok, _ in results if not ok]
print(f"\n=== {len(results) - len(failed)}/{len(results)} passed ===")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
