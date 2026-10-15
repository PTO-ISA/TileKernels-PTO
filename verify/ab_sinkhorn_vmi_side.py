"""VMI side of the sinkhorn bitwise A/B (run with the tkvmi conda env).

Loads the ASC artifacts (sinkhorn_asc_*.pt), reruns the same inputs through
the VMI kernels, and compares every output tensor bitwise (torch.equal) with
the mismatch count and max integer bit-distance reported per tensor.
"""

import glob
import os
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

from tile_kernels.config import get_num_vec_cores
from tile_kernels.mhc.sinkhorn_vmi import mhc_sinkhorn_bwd_vmi, mhc_sinkhorn_fwd_vmi

DEV = "npu"
NVC = get_num_vec_cores()
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


all_ok = True
for art_path in sorted(glob.glob(os.path.join(ART_DIR, "sinkhorn_asc_*.pt"))):
    art = torch.load(art_path, weights_only=True)
    if "inputs" not in art:  # skip non-A/B probe artifacts (e.g. stage dumps)
        continue
    inp = art["inputs"]
    x = inp["x"].to(DEV)
    o_grad = inp["o_grad"].to(DEV)
    eps, repeat = inp["eps"], inp["repeat"]

    print(f"--- {os.path.basename(art_path)} ---", flush=True)

    out = torch.empty_like(x)
    mhc_sinkhorn_fwd_vmi(x, out, repeat, eps, token_block_size=4, num_vec_cores=NVC)
    torch.npu.synchronize()
    cnt, dist = bitdist(out.cpu(), art["fwd_out"])
    print(f"   fwd out: eq={cnt == 0} mismatch={cnt}/{out.numel()} maxbitdist={dist}", flush=True)
    all_ok &= cnt == 0

    grad_input = torch.empty_like(x)
    mhc_sinkhorn_bwd_vmi(o_grad, x, grad_input, repeat, eps, token_block_size=8, num_vec_cores=NVC)
    torch.npu.synchronize()
    cnt, dist = bitdist(grad_input.cpu(), art["bwd"]["grad_input"])
    print(f"   bwd grad_input: eq={cnt == 0} mismatch={cnt}/{grad_input.numel()} maxbitdist={dist}", flush=True)
    all_ok &= cnt == 0

print("\nALL BITWISE EQUAL" if all_ok else "\nMISMATCHES PRESENT (see above)")
sys.exit(0 if all_ok else 1)
