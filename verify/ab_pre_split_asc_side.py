"""ASC side of the pre_split_mixes bitwise A/B (run with the tl-run conda env).

Generates the shared inputs with a fixed CPU seed, runs the ASC kernels with
the dispatch layer's host parameters (token_block_size=64, num_stages=2), and
saves inputs + outputs to a .pt artifact for the VMI side to compare against.
The bwd post_layer_mix input is produced by the torch reference so both sides
consume bit-identical bwd inputs.  The grid includes n1=1001 for the ragged
tail block (bwd per-token lane masks).
"""

import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_split_mixes_asc import mhc_pre_split_mixes_bwd_mhc4_asc, mhc_pre_split_mixes_fwd_simd_mhc4_asc
from tile_kernels.torch.mhc import mhc_pre_split_mixes_ref

DEV = "npu"
NVC = get_num_vec_cores()
MHC_MULT = 4
MHC_MULT2 = 16
MHC_MULT3 = 24
POST_MULT = 2.0
PRE_EPS = 1e-2
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"
os.makedirs(ART_DIR, exist_ok=True)


def gen(n0, n1, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(shape):
        return torch.randn(shape, generator=g, dtype=torch.float32).to(DEV)

    return {
        "input_mixes": rnd((n0, n1, MHC_MULT3)),
        "mhc_scale": rnd((3,)),
        "mhc_base": rnd((MHC_MULT3,)),
        "pre_grad": rnd((n0, n1, MHC_MULT)),
        "post_grad": rnd((n0, n1, MHC_MULT)),
        "comb_grad": rnd((n0, n1, MHC_MULT2)),
    }


CASES = [
    (1, 1001),
    (1, 1024),
    (1, 4096),
    (2, 1001),
    (2, 4096),
]

for n0, n1 in CASES:
    tag = f"n0{n0}_n1{n1}_seed0"
    art_path = os.path.join(ART_DIR, f"pre_split_asc_{tag}.pt")
    d = gen(n0, n1, seed=0)
    n = n0 * n1

    input_mixes_2d = d["input_mixes"].reshape(n, MHC_MULT3)
    pre_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    post_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    comb_out = torch.empty((n, MHC_MULT2), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_fwd_simd_mhc4_asc(
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

    # Bit-identical bwd input for both sides: torch-ref post_layer_mix
    # (= sigmoid(x*scale1+base) * post_mult, same math the fwd computes).
    _, post_ref_2d, _ = mhc_pre_split_mixes_ref(d["input_mixes"], d["mhc_scale"], d["mhc_base"], MHC_MULT, POST_MULT, PRE_EPS)
    post_for_bwd = post_ref_2d.reshape(n, MHC_MULT).contiguous()

    input_mixes_grad = torch.empty((n, MHC_MULT3), dtype=torch.float32, device=DEV)
    scale_grad_partial = torch.empty((NVC, 3), dtype=torch.float32, device=DEV)
    base_grad_partial = torch.empty((NVC, MHC_MULT3), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_bwd_mhc4_asc(
        d["pre_grad"].reshape(n, MHC_MULT),
        d["post_grad"].reshape(n, MHC_MULT),
        d["comb_grad"].reshape(n, MHC_MULT2),
        input_mixes_2d,
        post_for_bwd,
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

    torch.save(
        {
            "inputs": {
                "input_mixes": d["input_mixes"].cpu(),
                "mhc_scale": d["mhc_scale"].cpu(),
                "mhc_base": d["mhc_base"].cpu(),
                "pre_grad": d["pre_grad"].cpu(),
                "post_grad": d["post_grad"].cpu(),
                "comb_grad": d["comb_grad"].cpu(),
                "post_for_bwd": post_for_bwd.cpu(),
            },
            "fwd_out": {"pre": pre_out.cpu(), "post": post_out.cpu(), "comb": comb_out.cpu()},
            "bwd": {
                "input_mixes_grad": input_mixes_grad.cpu(),
                "scale_grad_partial": scale_grad_partial.cpu(),
                "base_grad_partial": base_grad_partial.cpu(),
            },
        },
        art_path,
    )
    print(f"ASC done {tag} -> {art_path}", flush=True)

print("ASC side complete")
