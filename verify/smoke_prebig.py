"""Compile smoke test for the VMI-ported mhc pre_big_fuse kernel.

Compiles and runs mhc_pre_big_fuse_fwd_vmi smallest-first on NPU 0 and checks
numerical sanity against a pure-torch chain reference (reduce/rmsnorm +
split mixes + sinkhorn + apply mix, using the tile_kernels torch refs).
Bitwise fidelity to the ASC kernel is established by the A/B scripts; this
smoke only guards against gross errors and collects compile failures with
their full traceback (for a rule-4 report).
"""

import math
import sys
import traceback

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401  registers the npu backend

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_big_fuse_vmi import mhc_pre_big_fuse_fwd_vmi
from tile_kernels.torch.mhc import mhc_pre_apply_mix_ref, mhc_pre_split_mixes_ref, sinkhorn_normalize_ref
from tile_kernels.utils import align

DEV = "npu"
NVC = get_num_vec_cores()
MHC = 4


def host_h_blk(hidden: int) -> int:
    # mirrors tile_kernels.mhc.pre_big_fuse_kernel.mhc_pre_big_fuse_fwd
    return min(math.gcd(512, hidden), 256)


def gen(n1, h, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mhc3 = MHC * 2 + MHC * MHC

    def rnd(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32)

    residual = (rnd((1, n1, MHC, h)) * (1 + torch.arange(MHC).mul(0.01).view(1, 1, -1, 1))).bfloat16()
    fn = (rnd((mhc3, MHC, h)) * 1e-4 * (1 + torch.arange(MHC).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    mhc_scale = rnd((3,)) * 0.1
    mhc_base = rnd((mhc3,)) * 0.1

    # GEMM partials as the test builds them (Ascend path: n_splits=1).
    x_flat = residual.float().flatten(2, 3).reshape(n1, MHC * h)
    out_mul = torch.einsum('mbk,nbk->mbn', x_flat.view(n1, 1, MHC * h), fn.view(mhc3, 1, MHC * h)).reshape(n1, mhc3)
    sqrsum = x_flat.square().sum(-1).reshape(n1)
    return residual, out_mul, sqrsum, mhc_scale, mhc_base


def torch_reference(residual, out_mul, sqrsum, mhc_scale, mhc_base, h, eps, pre_eps, sk_eps, post_mult, repeat):
    n1 = residual.shape[1]
    K = MHC * h
    # reduce + rmsnorm on the single partial, mirroring the ASC chain's math.
    rms = (sqrsum / K + eps).rsqrt()
    mixes = (out_mul * rms.view(n1, 1)).view(1, n1, -1)
    pre, post, comb = mhc_pre_split_mixes_ref(mixes, mhc_scale, mhc_base, MHC, post_mult, pre_eps)
    comb_norm = sinkhorn_normalize_ref(comb, repeat, sk_eps)
    layer_input = mhc_pre_apply_mix_ref(residual, pre)
    return post.view(n1, MHC), comb_norm.reshape(n1, MHC * MHC), layer_input.view(n1, h)


def run(n1, h):
    residual, out_mul, sqrsum, mhc_scale, mhc_base = gen(n1, h, seed=0)
    eps, pre_eps, sk_eps, post_mult, repeat = 1e-6, 1e-6, 1e-6, 1.0, 10

    post_mix = torch.empty((n1, MHC), dtype=torch.float32, device=DEV)
    comb_mix = torch.empty((n1, MHC * MHC), dtype=torch.float32, device=DEV)
    layer_input = torch.empty((n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_big_fuse_fwd_vmi(
        out_mul.to(DEV).view(1, n1, -1),
        sqrsum.to(DEV).view(1, n1),
        mhc_scale.to(DEV),
        mhc_base.to(DEV),
        residual.view(n1, MHC, h).to(DEV),
        post_mix,
        comb_mix,
        layer_input,
        eps,
        pre_eps,
        sk_eps,
        post_mult,
        repeat,
        h_blk=host_h_blk(h),
        n_splits=1,
        token_block=16,
        num_stages=4,
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    post_ref, comb_ref, layer_ref = torch_reference(
        residual, out_mul, sqrsum, mhc_scale, mhc_base, h, eps, pre_eps, sk_eps, post_mult, repeat
    )
    post_ref, comb_ref, layer_ref = post_ref.to(DEV), comb_ref.to(DEV), layer_ref.to(DEV)
    torch.testing.assert_close(post_mix, post_ref, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(comb_mix, comb_ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(layer_input.float(), layer_ref.float(), atol=2e-2, rtol=2e-2)
    return (
        f"n1={n1} h={h} h_blk={host_h_blk(h)} "
        f"maxd post={(post_mix - post_ref).abs().max().item():.2e} "
        f"comb={(comb_mix - comb_ref).abs().max().item():.2e} "
        f"layer={(layer_input.float() - layer_ref.float()).abs().max().item():.2e}"
    )


CASES = [
    ("n1=16 h=1280", 16, 1280),
    ("n1=32 h=1280", 32, 1280),
    ("n1=32 h=4096", 32, 4096),
    ("n1=512 h=4096", 512, 4096),
    ("n1=1021 h=4096 (tail)", 1021, 4096),
]

results = []
for name, n1, h in CASES:
    try:
        detail = run(n1, h)
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
