"""VMI side of the pre_apply_mix bitwise A/B (run with the tkvmi conda env).

Loads the ASC artifacts (pre_asc_*.pt), reruns the same inputs through the
VMI kernels, and compares every output tensor bitwise (torch.equal) with the
mismatch count and max integer bit-distance reported per tensor.
"""

import glob
import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_apply_mix_vmi import mhc_pre_apply_mix_bwd_vmi, mhc_pre_apply_mix_fwd_vmi
from tile_kernels.utils import align, ceil_div

DEV = "npu"
NVC = get_num_vec_cores()
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"


def host_h_blk_fwd(hidden: int) -> int:
    h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
    return align(h_blk, 64)


def host_h_blk_bwd(hidden: int) -> int:
    return hidden if hidden <= 4096 else align(min(ceil_div(hidden, 4), 3072), 64)


def host_num_stages_bwd(hidden: int) -> int:
    return 3 if hidden <= 4096 else 4


def bitdist(a, b):
    """(mismatch_count, max_unsigned_int_bit_distance) over two same-dtype tensors."""
    if torch.equal(a, b):
        return 0, 0
    neq = a != b
    if a.dtype == torch.bfloat16:
        ai = a.view(torch.int16).to(torch.int32) & 0xFFFF
        bi = b.view(torch.int16).to(torch.int32) & 0xFFFF
    else:
        ai = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        bi = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    d = (ai - bi).abs()
    return int(neq.sum().item()), int(d.max().item())


all_ok = True
for art_path in sorted(glob.glob(os.path.join(ART_DIR, "pre_asc_*.pt"))):
    art = torch.load(art_path, weights_only=True)
    inp = art["inputs"]
    x = inp["x"].to(DEV)
    mix = inp["mix"].to(DEV)
    o_grad = inp["o_grad"].to(DEV)
    x_grad_init = inp["x_grad_init"].to(DEV)
    n0, n1, mhc, h = x.shape
    n = n0 * n1

    print(f"--- {os.path.basename(art_path)} ---", flush=True)

    o = torch.empty((n0, n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_apply_mix_fwd_vmi(
        x.view(n, mhc, h),
        mix.view(n, mhc),
        o.view(n, h),
        h_blk=host_h_blk_fwd(h),
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()
    cnt, dist = bitdist(o.cpu(), art["fwd_out"])
    print(f"   fwd o: eq={cnt == 0} mismatch={cnt}/{o.numel()} maxbitdist={dist}", flush=True)
    all_ok &= cnt == 0

    x_grad = x_grad_init.clone()
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
    for name, v, ref in (("x_grad", x_grad, art["bwd"]["x_grad"]), ("mix_grad", mix_grad, art["bwd"]["mix_grad"])):
        cnt, dist = bitdist(v.cpu(), ref)
        print(f"   bwd {name}: eq={cnt == 0} mismatch={cnt}/{v.numel()} maxbitdist={dist}", flush=True)
        all_ok &= cnt == 0

print("\nALL BITWISE EQUAL" if all_ok else "\nMISMATCHES PRESENT (see above)")
sys.exit(0 if all_ok else 1)
