"""ASC side of the pre_apply_mix bitwise A/B (run with the tl-run conda env).

Generates the shared inputs with a fixed CPU seed, runs the ASC kernels, and
saves inputs + outputs to a .pt artifact for the VMI side to compare against.
The bwd x_grad accumulator is pre-filled with random values (not zeros) so the
read-modify-write accumulation path is exercised on both sides.
"""

import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_bwd_asc, mhc_pre_apply_mix_fwd_asc
from tile_kernels.utils import align, ceil_div

DEV = "npu"
NVC = get_num_vec_cores()
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"
os.makedirs(ART_DIR, exist_ok=True)


def host_h_blk_fwd(hidden: int) -> int:
    h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
    return align(h_blk, 64)


def host_h_blk_bwd(hidden: int) -> int:
    return hidden if hidden <= 4096 else align(min(ceil_div(hidden, 4), 3072), 64)


def host_num_stages_bwd(hidden: int) -> int:
    return 3 if hidden <= 4096 else 4


def gen(n0, n1, h, mhc, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(shape, dtype):
        return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).to(DEV)

    x = rnd((n0, n1, mhc, h), torch.bfloat16)
    mix = rnd((n0, n1, mhc, 1), torch.float32)
    o_grad = rnd((n0, n1, h), torch.bfloat16)
    x_grad_init = rnd((n0, n1, mhc, h), torch.bfloat16)
    return x, mix, o_grad, x_grad_init


CASES = [
    (1, 128, 1280, 4),
    (1, 128, 2560, 4),
    (1, 128, 4096, 4),
    (1, 128, 5120, 4),
    (1, 128, 7168, 4),
    (2, 512, 7168, 4),
]

for n0, n1, h, mhc in CASES:
    tag = f"n0{n0}_n1{n1}_h{h}_seed0"
    art_path = os.path.join(ART_DIR, f"pre_asc_{tag}.pt")
    x, mix, o_grad, x_grad_init = gen(n0, n1, h, mhc, seed=0)
    n = n0 * n1

    o = torch.empty((n0, n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_apply_mix_fwd_asc(
        x.view(n, mhc, h),
        mix.view(n, mhc),
        o.view(n, h),
        h_blk=host_h_blk_fwd(h),
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    x_grad = x_grad_init.clone()
    mix_grad = mhc_pre_apply_mix_bwd_asc(
        o_grad.view(n, h),
        x.view(n, mhc, h),
        mix.view(n, mhc),
        x_grad.view(n, mhc, h),
        h_blk=host_h_blk_bwd(h),
        num_stages=host_num_stages_bwd(h),
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()

    torch.save(
        {
            "inputs": {
                "x": x.cpu(),
                "mix": mix.cpu(),
                "o_grad": o_grad.cpu(),
                "x_grad_init": x_grad_init.cpu(),
            },
            "fwd_out": o.cpu(),
            "bwd": {"x_grad": x_grad.cpu(), "mix_grad": mix_grad.cpu()},
        },
        art_path,
    )
    print(f"ASC done {tag} -> {art_path}", flush=True)

print("ASC side complete")
