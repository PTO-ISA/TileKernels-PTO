"""ASC side of the sinkhorn bitwise A/B (run with the tl-run conda env).

Generates the shared inputs with a fixed CPU seed, runs the ASC kernels, and
saves inputs + outputs to a .pt artifact for the VMI side to compare against.
Covers the repo test grid (random logits) plus the large same-sign-logits
regime (offset +-100, eps 1e-2).
"""

import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.sinkhorn_asc import mhc_sinkhorn_bwd_asc, mhc_sinkhorn_fwd_asc

DEV = "npu"
NVC = get_num_vec_cores()
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"
os.makedirs(ART_DIR, exist_ok=True)
REPEAT = 10


def gen_random(n, g):
    return torch.randn((n, 4, 4), generator=g, dtype=torch.float32).to(DEV)


def gen_large_logits(n, offset, g):
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0, 0.0],
            [1.5, 0.5, 3.5, 2.5],
            [2.5, 3.5, 0.5, 1.5],
        ],
        dtype=torch.float32,
    )
    return (base + offset).repeat(n, 1, 1).to(DEV)


# (tag, x, o_grad, eps); num_tokens is always a multiple of the bwd block (8).
g = torch.Generator(device="cpu").manual_seed(0)
CASES = [
    ("random_n8", gen_random(8, g), gen_random(8, g), 1e-6),
    ("random_n128", gen_random(128, g), gen_random(128, g), 1e-6),
    ("random_n1024", gen_random(1024, g), gen_random(1024, g), 1e-6),
    ("random_n4096", gen_random(4096, g), gen_random(4096, g), 1e-6),
    ("logits_p100_n8", gen_large_logits(8, 100.0, g), gen_random(8, g), 1e-6),
    ("logits_m100_n8", gen_large_logits(8, -100.0, g), gen_random(8, g), 1e-2),
]

for tag, x, o_grad, eps in CASES:
    art_path = os.path.join(ART_DIR, f"sinkhorn_asc_{tag}.pt")

    out = torch.empty_like(x)
    mhc_sinkhorn_fwd_asc(x, out, REPEAT, eps, token_block_size=4, num_vec_cores=NVC)
    torch.npu.synchronize()

    grad_input = torch.empty_like(x)
    mhc_sinkhorn_bwd_asc(o_grad, x, grad_input, REPEAT, eps, token_block_size=8, num_vec_cores=NVC)
    torch.npu.synchronize()

    torch.save(
        {
            "inputs": {"x": x.cpu(), "o_grad": o_grad.cpu(), "eps": eps, "repeat": REPEAT},
            "fwd_out": out.cpu(),
            "bwd": {"grad_input": grad_input.cpu()},
        },
        art_path,
    )
    print(f"ASC done {tag} -> {art_path}", flush=True)

print("ASC side complete")
