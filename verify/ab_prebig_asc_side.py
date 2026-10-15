"""ASC side of the pre_big_fuse bitwise A/B (run on NPU 0).

Generates the shared inputs with a fixed CPU seed, runs the ASC fused kernel,
and saves inputs + outputs to a .pt artifact for the VMI side to compare
against.
"""

import math
import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_big_fuse_asc import mhc_pre_big_fuse_fwd_asc

DEV = "npu"
NVC = get_num_vec_cores()
MHC = 4
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"
os.makedirs(ART_DIR, exist_ok=True)


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

    x_flat = residual.float().flatten(2, 3).reshape(n1, MHC * h)
    out_mul = torch.einsum('mbk,nbk->mbn', x_flat.view(n1, 1, MHC * h), fn.view(mhc3, 1, MHC * h)).reshape(n1, mhc3)
    sqrsum = x_flat.square().sum(-1).reshape(n1)
    return residual, out_mul, sqrsum, mhc_scale, mhc_base


CASES = [
    (16, 1280),
    (128, 1280),
    (128, 2560),
    (128, 4096),
    (128, 4096 + 64),
    (128, 5120),
    (128, 7168),
    (512, 7168),
    (1021, 4096),
]

for n1, h in CASES:
    tag = f"n1{n1}_h{h}_seed0"
    art_path = os.path.join(ART_DIR, f"prebig_asc_{tag}.pt")
    residual, out_mul, sqrsum, mhc_scale, mhc_base = gen(n1, h, seed=0)

    post_mix = torch.empty((n1, MHC), dtype=torch.float32, device=DEV)
    comb_mix = torch.empty((n1, MHC * MHC), dtype=torch.float32, device=DEV)
    layer_input = torch.empty((n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_big_fuse_fwd_asc(
        out_mul.to(DEV).view(1, n1, -1),
        sqrsum.to(DEV).view(1, n1),
        mhc_scale.to(DEV),
        mhc_base.to(DEV),
        residual.view(n1, MHC, h).to(DEV),
        post_mix,
        comb_mix,
        layer_input,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        10,
        h_blk=host_h_blk(h),
        n_splits=1,
        token_block=16,
        num_stages=4,
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    torch.save(
        {
            "inputs": {
                "out_mul": out_mul.cpu(),
                "sqrsum": sqrsum.cpu(),
                "mhc_scale": mhc_scale.cpu(),
                "mhc_base": mhc_base.cpu(),
                "residual": residual.cpu(),
            },
            "out": {"post_mix": post_mix.cpu(), "comb_mix": comb_mix.cpu(), "layer_input": layer_input.cpu()},
        },
        art_path,
    )
    print(f"ASC done {tag} -> {art_path}", flush=True)

print("ASC side complete")
