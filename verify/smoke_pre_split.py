"""Compile smoke test for the VMI-ported mhc pre_split_mixes kernels.

Compiles and runs mhc_pre_split_mixes_fwd_simd_mhc4_vmi /
mhc_pre_split_mixes_bwd_mhc4_vmi on NPU 0 with the host parameters the
dispatch layer uses (token_block_size=64, num_stages=2), checks numerical
sanity against the torch reference (mhc_pre_split_mixes_ref + autograd, same
criteria as tests/mhc/test_pre_split_mixes.py), and prints PASS/FAIL per
case.  The grid includes n1=1001 so the ragged tail block (41 valid tokens)
exercises the bwd per-token lane masks.  Failures print the full traceback so
the tilelang temp dir hint (kernel.pto / ptodsl) can be collected for a
rule-4 report.

All tensors are fp32 end to end (no bf16 rounding pitfalls).
"""

import sys
import traceback

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401  registers the npu backend

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_split_mixes_vmi import mhc_pre_split_mixes_bwd_mhc4_vmi, mhc_pre_split_mixes_fwd_simd_mhc4_vmi
from tile_kernels.torch.mhc import mhc_pre_split_mixes_ref

DEV = "npu"
NVC = get_num_vec_cores()
MHC_MULT = 4
POST_MULT = 2.0
PRE_EPS = 1e-2


def gen(n0, n1, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mhc_mult3 = MHC_MULT * 2 + MHC_MULT * MHC_MULT
    return {
        "input_mixes": torch.randn((n0, n1, mhc_mult3), generator=g, dtype=torch.float32).to(DEV),
        "mhc_scale": torch.randn((3,), generator=g, dtype=torch.float32).to(DEV),
        "mhc_base": torch.randn((mhc_mult3,), generator=g, dtype=torch.float32).to(DEV),
        "pre_grad": torch.randn((n0, n1, MHC_MULT, 1), generator=g, dtype=torch.float32).to(DEV),
        "post_grad": torch.randn((n0, n1, MHC_MULT, 1), generator=g, dtype=torch.float32).to(DEV),
        "comb_grad": torch.randn((n0, n1, MHC_MULT, MHC_MULT), generator=g, dtype=torch.float32).to(DEV),
    }


def run_case(n0, n1):
    d = gen(n0, n1, seed=0)
    mhc_mult2 = MHC_MULT * MHC_MULT
    mhc_mult3 = MHC_MULT * 2 + mhc_mult2
    n = n0 * n1

    input_mixes_2d = d["input_mixes"].reshape(n, mhc_mult3)
    pre_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    post_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    comb_out = torch.empty((n, mhc_mult2), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_fwd_simd_mhc4_vmi(
        input_mixes_2d,
        d["mhc_scale"],
        d["mhc_base"],
        pre_out,
        post_out,
        comb_out,
        POST_MULT,
        PRE_EPS,
        token_block_size=64,
        num_stages=2,
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    input_mixes_grad = torch.empty((n, mhc_mult3), dtype=torch.float32, device=DEV)
    scale_grad_partial = torch.empty((NVC, 3), dtype=torch.float32, device=DEV)
    base_grad_partial = torch.empty((NVC, mhc_mult3), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_bwd_mhc4_vmi(
        d["pre_grad"].reshape(n, MHC_MULT),
        d["post_grad"].reshape(n, MHC_MULT),
        d["comb_grad"].reshape(n, mhc_mult2),
        input_mixes_2d,
        post_out,
        d["mhc_scale"],
        d["mhc_base"],
        input_mixes_grad,
        scale_grad_partial,
        base_grad_partial,
        POST_MULT,
        token_block_size=64,
        num_stages=2,
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    pre_ref, post_ref, comb_ref = mhc_pre_split_mixes_ref(d["input_mixes"], d["mhc_scale"], d["mhc_base"], MHC_MULT, POST_MULT, PRE_EPS)
    input_mixes_ref = d["input_mixes"].clone().requires_grad_()
    mhc_scale_ref = d["mhc_scale"].clone().requires_grad_()
    mhc_base_ref = d["mhc_base"].clone().requires_grad_()
    pre_r, post_r, comb_r = mhc_pre_split_mixes_ref(input_mixes_ref, mhc_scale_ref, mhc_base_ref, MHC_MULT, POST_MULT, PRE_EPS)
    torch.autograd.backward([pre_r, post_r, comb_r], [d["pre_grad"], d["post_grad"], d["comb_grad"]])

    torch.testing.assert_close(pre_out.view(n0, n1, MHC_MULT, 1), pre_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(post_out.view(n0, n1, MHC_MULT, 1), post_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(comb_out.view(n0, n1, MHC_MULT, MHC_MULT), comb_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(input_mixes_grad.view(n0, n1, mhc_mult3), input_mixes_ref.grad, rtol=1e-5, atol=2e-5)
    scale_grad = scale_grad_partial.sum(0)
    base_grad = base_grad_partial.sum(0)
    torch.testing.assert_close(scale_grad, mhc_scale_ref.grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(base_grad, mhc_base_ref.grad, rtol=1e-5, atol=2e-5)
    return (
        f"n={n} max|diff| fwd pre/post/comb="
        f"{(pre_out.view_as(pre_ref) - pre_ref).abs().max().item():.2e}/"
        f"{(post_out.view_as(post_ref) - post_ref).abs().max().item():.2e}/"
        f"{(comb_out.view_as(comb_ref) - comb_ref).abs().max().item():.2e} "
        f"bwd in_grad={(input_mixes_grad.view_as(input_mixes_ref.grad) - input_mixes_ref.grad).abs().max().item():.2e}"
    )


CASES = [
    (1, 1001),  # ragged tail: 15*64 + 41 tokens
    (1, 1024),
    (1, 4096),
    (2, 1001),
    (2, 4096),
]

results = []
for n0, n1 in CASES:
    name = f"n0={n0} n1={n1}"
    try:
        detail = run_case(n0, n1)
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
