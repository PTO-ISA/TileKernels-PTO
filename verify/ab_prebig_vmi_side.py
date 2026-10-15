"""VMI side of the pre_big_fuse bitwise A/B (run on NPU 0 with the tkvmi env).

Loads the ASC artifacts (prebig_asc_*.pt), reruns the same inputs through the
VMI kernel, and compares every output tensor bitwise (torch.equal) with the
mismatch count and max integer bit-distance reported per tensor.
"""

import glob
import math
import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_big_fuse_vmi import mhc_pre_big_fuse_fwd_vmi

DEV = "npu"
NVC = get_num_vec_cores()
MHC = 4
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"


def host_h_blk(hidden: int) -> int:
    return min(math.gcd(512, hidden), 256)


def bitdist(a, b):
    """(mismatch_count, max_unsigned_int_bit_distance) over two same-dtype tensors."""
    if torch.equal(a, b):
        return 0, 0
    neq = a != b
    if a.dtype == torch.bfloat16:
        ai = a.view(torch.int16).to(torch.int32) & 0xFFFF
        bi = b.view(torch.int16).to(torch.int32) & 0xFFFF
    elif a.dtype == torch.float32:
        ai = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        bi = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    else:
        ai = a.to(torch.int64)
        bi = b.to(torch.int64)
    d = (ai - bi).abs()
    return int(neq.sum().item()), int(d.max().item())


all_ok = True
for art_path in sorted(glob.glob(os.path.join(ART_DIR, "prebig_asc_*.pt"))):
    art = torch.load(art_path, weights_only=True)
    inp = art["inputs"]
    out_mul = inp["out_mul"].to(DEV)
    sqrsum = inp["sqrsum"].to(DEV)
    mhc_scale = inp["mhc_scale"].to(DEV)
    mhc_base = inp["mhc_base"].to(DEV)
    residual = inp["residual"].to(DEV)
    n1, h = residual.shape[1], residual.shape[3]

    print(f"--- {os.path.basename(art_path)} ---", flush=True)

    post_mix = torch.empty((n1, MHC), dtype=torch.float32, device=DEV)
    comb_mix = torch.empty((n1, MHC * MHC), dtype=torch.float32, device=DEV)
    layer_input = torch.empty((n1, h), dtype=torch.bfloat16, device=DEV)
    mhc_pre_big_fuse_fwd_vmi(
        out_mul.view(1, n1, -1),
        sqrsum.view(1, n1),
        mhc_scale,
        mhc_base,
        residual.view(n1, MHC, h),
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

    for name, v, ref in (
        ("post_mix", post_mix, art["out"]["post_mix"]),
        ("comb_mix", comb_mix, art["out"]["comb_mix"]),
        ("layer_input", layer_input, art["out"]["layer_input"]),
    ):
        cnt, dist = bitdist(v.cpu(), ref)
        print(f"   {name}: eq={cnt == 0} mismatch={cnt}/{v.numel()} maxbitdist={dist}", flush=True)
        all_ok &= cnt == 0

print("\nALL BITWISE EQUAL" if all_ok else "\nMISMATCHES PRESENT (see above)")
sys.exit(0 if all_ok else 1)
