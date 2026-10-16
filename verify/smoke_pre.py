"""Compile smoke test for the VMI-ported mhc pre_apply_mix kernels.

Compiles and runs mhc_pre_apply_mix_fwd_vmi / mhc_pre_apply_mix_bwd_vmi
smallest-first on NPU 0, checks numerical sanity against the torch reference
(mhc_pre_apply_mix_ref + autograd), and prints PASS/FAIL per kernel.  The bwd
x_grad accumulator is pre-filled with random values so the
read-modify-write accumulation path is exercised (the repo test only uses
zeros).  Failures print the full traceback so the tilelang temp dir hint
(kernel.pto / ptodsl) can be collected for a rule-4 report.
"""

import sys
import traceback

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401  registers the npu backend

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_apply_mix_vmi import mhc_pre_apply_mix_bwd_vmi, mhc_pre_apply_mix_fwd_vmi
from tile_kernels.torch.mhc import mhc_pre_apply_mix_ref
from tile_kernels.utils import align, ceil_div

DEV = "npu"
NVC = get_num_vec_cores()


def host_h_blk_fwd(hidden: int) -> int:
    # mirrors tile_kernels.mhc.pre_apply_mix_kernel.mhc_pre_apply_mix_fwd
    h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
    return align(h_blk, 64)


def host_h_blk_bwd(hidden: int) -> int:
    # mirrors tile_kernels.mhc.pre_apply_mix_kernel.mhc_pre_apply_mix_bwd
    return hidden if hidden <= 4096 else align(min(ceil_div(hidden, 4), 3072), 64)


def host_num_stages_bwd(hidden: int) -> int:
    return 3 if hidden <= 4096 else 4


def gen(n0, n1, h, mhc):
    x = torch.randn((n0, n1, mhc, h), dtype=torch.float32, device=DEV).to(torch.bfloat16)
    mix = torch.randn((n0, n1, mhc, 1), dtype=torch.float32, device=DEV)
    o_grad = torch.randn((n0, n1, h), dtype=torch.float32, device=DEV).to(torch.bfloat16)
    return x, mix, o_grad


def run_fwd(n0, n1, h, mhc):
    n = n0 * n1
    x, mix, o_grad = gen(n0, n1, h, mhc)

    o = torch.empty((n0, n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_apply_mix_fwd_vmi(
        x.view(n, mhc, h),
        mix.view(n, mhc),
        o.view(n, h),
        h_blk=host_h_blk_fwd(h),
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    o_ref = mhc_pre_apply_mix_ref(x, mix)
    torch.testing.assert_close(o, o_ref)
    return f"n={n} h={h} max|diff|={(o.float() - o_ref.float()).abs().max().item():.3e} bitwise={torch.equal(o, o_ref)}"


def run_bwd(n0, n1, h, mhc):
    n = n0 * n1
    x, mix, o_grad = gen(n0, n1, h, mhc)
    x_init = torch.randn((n0, n1, mhc, h), dtype=torch.float32, device=DEV).to(torch.bfloat16)

    x_ref = x.clone().requires_grad_()
    mix_ref = mix.clone().requires_grad_()
    o_ref = mhc_pre_apply_mix_ref(x_ref, mix_ref)
    torch.autograd.backward([o_ref], [o_grad])

    x_grad = x_init.clone()
    mix_grad = mhc_pre_apply_mix_bwd_vmi(
        o_grad.view(n, h),
        x.view(n, mhc, h),
        mix.view(n, mhc),
        x_grad.view(n, mhc, h),
        h_blk=host_h_blk_bwd(h),
        num_stages=host_num_stages_bwd(h),
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    x_grad = x_grad.view(n0, n1, mhc, h)
    mix_grad = mix_grad.view(n0, n1, mhc, 1)
    # Kernel math per element: bf16(f32(x_init) + mix_f32 * f32(o_grad)) -- the
    # product stays fp32.  Autograd's x_ref.grad would pre-round mix*o_grad to
    # bf16 (o_ref is bf16), so build the reference from the raw operands
    # instead; that is exactly what the ASC kernel computes too.
    x_grad_expect = (x_init.float() + mix.float() * o_grad.view(n0, n1, 1, h).float()).to(torch.bfloat16)

    torch.testing.assert_close(x_grad, x_grad_expect, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(mix_grad, mix_ref.grad, atol=1e-4, rtol=1e-4)
    return (
        f"n={n} h={h} h_blk={host_h_blk_bwd(h)} stages={host_num_stages_bwd(h)} "
        f"x_grad_max={(x_grad.float() - x_grad_expect.float()).abs().max().item():.2e} "
        f"mix_grad_max={(mix_grad - mix_ref.grad).abs().max().item():.2e}"
    )


CASES = [
    ("fwd n=8 h=1280", run_fwd, (1, 8, 1280, 4)),
    ("fwd n=8 h=4096", run_fwd, (1, 8, 4096, 4)),
    ("fwd n=8 h=5120", run_fwd, (1, 8, 5120, 4)),
    ("fwd n=64 h=7168", run_fwd, (1, 64, 7168, 4)),
    ("bwd n=8 h=1280", run_bwd, (1, 8, 1280, 4)),
    ("bwd n=8 h=4096", run_bwd, (1, 8, 4096, 4)),
    ("bwd n=8 h=5120", run_bwd, (1, 8, 5120, 4)),
    ("bwd n=64 h=7168", run_bwd, (1, 64, 7168, 4)),
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
