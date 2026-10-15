"""VMI side of the pre_split_mixes bitwise A/B (run with the tkvmi conda env).

Loads the ASC artifacts (pre_split_asc_*.pt), reruns the same inputs through
the VMI kernels, and compares every output tensor bitwise (torch.equal) with
the mismatch count and max integer bit-distance reported per tensor.

Expected outcome: the comb outputs and the bwd comb-column gradients are pure
mul/add chains -> bitwise identical.  The pre/post outputs and the sigmoid-
derived bwd quantities go through a hardware divide (sigmoid = 1/(1+exp(-x)))
on both sides; ASC and VMI may differ by a few ULP there (documented as
rounding difference, discussion #1616 acceptance note 2).
"""

import glob
import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.pre_split_mixes_vmi import mhc_pre_split_mixes_bwd_mhc4_vmi, mhc_pre_split_mixes_fwd_simd_mhc4_vmi

DEV = "npu"
NVC = get_num_vec_cores()
MHC_MULT = 4
MHC_MULT2 = 16
MHC_MULT3 = 24
POST_MULT = 2.0
PRE_EPS = 1e-2
ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"


def bitdist(a, b):
    """(mismatch_count, max_unsigned_int_bit_distance) over two same-dtype tensors."""
    if torch.equal(a, b):
        return 0, 0
    neq = a != b
    ai = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bi = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    d = (ai - bi).abs()
    return int(neq.sum().item()), int(d.max().item())


def ulp(a, b):
    """Max relative ULP distance (integer bit distance on the fp32 pattern)."""
    ai = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bi = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return int((ai - bi).abs().max().item())


all_ok = True
for art_path in sorted(glob.glob(os.path.join(ART_DIR, "pre_split_asc_*.pt"))):
    art = torch.load(art_path, weights_only=True)
    inp = art["inputs"]
    n0, n1, _ = inp["input_mixes"].shape
    n = n0 * n1
    input_mixes_2d = inp["input_mixes"].to(DEV).reshape(n, MHC_MULT3)
    mhc_scale = inp["mhc_scale"].to(DEV)
    mhc_base = inp["mhc_base"].to(DEV)

    print(f"--- {os.path.basename(art_path)} ---", flush=True)

    pre_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    post_out = torch.empty((n, MHC_MULT), dtype=torch.float32, device=DEV)
    comb_out = torch.empty((n, MHC_MULT2), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_fwd_simd_mhc4_vmi(
        input_mixes_2d,
        mhc_scale,
        mhc_base,
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
    for name, v, ref in (
        ("fwd pre", pre_out.cpu(), art["fwd_out"]["pre"]),
        ("fwd post", post_out.cpu(), art["fwd_out"]["post"]),
        ("fwd comb", comb_out.cpu(), art["fwd_out"]["comb"]),
    ):
        cnt, dist = bitdist(v, ref)
        print(f"   {name}: eq={cnt == 0} mismatch={cnt}/{v.numel()} maxbitdist={dist}", flush=True)
        all_ok &= cnt == 0

    input_mixes_grad = torch.empty((n, MHC_MULT3), dtype=torch.float32, device=DEV)
    scale_grad_partial = torch.empty((NVC, 3), dtype=torch.float32, device=DEV)
    base_grad_partial = torch.empty((NVC, MHC_MULT3), dtype=torch.float32, device=DEV)
    mhc_pre_split_mixes_bwd_mhc4_vmi(
        inp["pre_grad"].to(DEV).reshape(n, MHC_MULT),
        inp["post_grad"].to(DEV).reshape(n, MHC_MULT),
        inp["comb_grad"].to(DEV).reshape(n, MHC_MULT2),
        input_mixes_2d,
        inp["post_for_bwd"].to(DEV),
        mhc_scale,
        mhc_base,
        input_mixes_grad,
        scale_grad_partial,
        base_grad_partial,
        POST_MULT,
        token_block_size=64,
        num_stages=2,
        num_vec_cores=NVC,
    )
    torch.npu.synchronize()
    for name, v, ref in (
        ("bwd input_mixes_grad", input_mixes_grad.cpu(), art["bwd"]["input_mixes_grad"]),
        ("bwd scale_grad_partial", scale_grad_partial.cpu(), art["bwd"]["scale_grad_partial"]),
        ("bwd base_grad_partial", base_grad_partial.cpu(), art["bwd"]["base_grad_partial"]),
    ):
        cnt, dist = bitdist(v, ref)
        print(f"   {name}: eq={cnt == 0} mismatch={cnt}/{v.numel()} maxbitdist={dist}", flush=True)
        all_ok &= cnt == 0

    # Column-split summary for input_mixes_grad: comb columns (8..23) are the
    # pure mul/add path and must stay bitwise; pre/post columns carry the
    # sigmoid divide.
    g_v = input_mixes_grad.cpu().view(n, MHC_MULT3)
    g_r = art["bwd"]["input_mixes_grad"].view(n, MHC_MULT3)
    c_pre = ulp(g_v[:, :4], g_r[:, :4])
    c_post = ulp(g_v[:, 4:8], g_r[:, 4:8])
    c_comb = ulp(g_v[:, 8:], g_r[:, 8:])
    print(f"   grad columns maxbitdist: pre={c_pre} post={c_post} comb={c_comb}", flush=True)

print("\nALL BITWISE EQUAL" if all_ok else "\nMISMATCHES PRESENT (see above)")
sys.exit(0 if all_ok else 1)
