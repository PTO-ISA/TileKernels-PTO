"""Primitive-level probe for the sinkhorn VMI port: dumps each vector
primitive's output for a known ramp input and compares against the expected
math on host.  Isolates which of vci / vcadd(group) / vbrc(group) / vselr /
vcmps / vcvt(int32->f32) diverges from the assumed semantics.
"""

import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

torch.npu.set_device(0)
DEV = "npu"

PROBES = ['iota', 'masklo', 'cgadd_lo', 'selr', 'bidx', 'tr']


@tilelang.jit(target='pto')
def probe_vmi(x, o_iota, o_masklo, o_cgadd_lo, o_selr, o_bidx, o_tr):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_iota: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_masklo: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_cgadd_lo: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_selr: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_bidx: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_tr: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        iota_ub = T.alloc_shared((4, 4, 4), T.float32)
        masklo_ub = T.alloc_shared((4, 4, 4), T.float32)
        cgadd_ub = T.alloc_shared((4, 4, 4), T.float32)
        selr_ub = T.alloc_shared((4, 4, 4), T.float32)
        bidx_ub = T.alloc_shared((4, 4, 4), T.float32)
        tr_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)

            idx = V.vci(T.int32(0), size=64)
            V.vstore(V.vcvt(idx, 'float32'), iota_ub[0, 0, 0])

            seven = V.vbrc(T.int32(7), size=64)
            mod8 = V.vand(idx, seven)
            mask_lo = V.vcmps(mod8, T.int32(4), mask_full, 'lt')
            ones = V.vbrc(T.float32(1.0), size=64)
            zeros = V.vbrc(T.float32(0.0), size=64)
            V.vstore(V.vsel(mask_lo, ones, zeros), masklo_ub[0, 0, 0])

            idx_f = V.vcvt(idx, 'float32')
            cg_lo = V.vbrc(V.vcadd(idx_f, mask_lo, group=8, reassoc=True), size=64, group=8)
            V.vstore(cg_lo, cgadd_ub[0, 0, 0])

            four = T.int32(4)
            fifteen = V.vbrc(T.int32(15), size=64)
            token2 = V.vshls(V.vshrs(idx, four, mask_full), T.int32(1), mask_full)
            local = V.vand(idx, fifteen)
            local_div8 = V.vshrs(local, T.int32(3), mask_full)
            bcast_idx2 = V.vadd(token2, local_div8)
            V.vstore(V.vcvt(bcast_idx2, 'float32'), bidx_ub[0, 0, 0])
            V.vstore(V.vselr(cg_lo, bcast_idx2), selr_ub[0, 0, 0])

            token_base_tidx = V.vshls(V.vshrs(idx, four, mask_full), four, mask_full)
            col = V.vand(idx, V.vbrc(T.int32(3), size=64))
            col4 = V.vshls(col, T.int32(2), mask_full)
            t_idx = V.vadd(V.vadd(token_base_tidx, col4), V.vshrs(local, T.int32(2), mask_full))
            mat = V.vload(x_ub[0, 0, 0], size=64)
            V.vstore(V.vselr(mat, t_idx), tr_ub[0, 0, 0])

        T.copy(iota_ub, o_iota[0, 0, 0])
        T.copy(masklo_ub, o_masklo[0, 0, 0])
        T.copy(cgadd_ub, o_cgadd_lo[0, 0, 0])
        T.copy(selr_ub, o_selr[0, 0, 0])
        T.copy(bidx_ub, o_bidx[0, 0, 0])
        T.copy(tr_ub, o_tr[0, 0, 0])


import numpy as np

n = 4
x = torch.arange(64, dtype=torch.float32, device=DEV).reshape(n, 4, 4)
outs = {k: torch.full((n, 4, 4), float('nan'), dtype=torch.float32, device=DEV) for k in PROBES}

probe_vmi(x, *[outs[k] for k in PROBES])
torch.npu.synchronize()


def check(name, got, exp):
    got = got.flatten().cpu()
    exp = torch.tensor(np.asarray(exp, dtype=np.float32))
    ok = torch.equal(got, exp)
    if ok:
        print(f"{name}: OK")
        return
    close = torch.isclose(got, exp)
    bad = (~close).sum().item()
    idxs = (~close).nonzero().flatten()[:8].tolist()
    print(f"{name}: MISMATCH ({bad}/64) first_bad_lanes={idxs} "
          f"got={[got[i].item() for i in idxs]} exp={[exp[i].item() for i in idxs]}")


lane = np.arange(64)
check('iota', outs['iota'], lane)

exp_masklo = np.where(lane % 8 < 4, 1.0, 0.0)
check('mask_lo', outs['masklo'], exp_masklo)

# vcadd(idx_f, mask_lo, group=8): lane g of result = sum lanes [8g,8g+4)
# vbrc(..., group=8): replicated over each 8-lane group
rsum = np.array([lane[8 * g:8 * g + 4].sum() for g in range(8)])
exp_cg = np.repeat(rsum, 8)
check('cgadd_lo+vbrc', outs['cgadd_lo'], exp_cg)

bidx = (lane // 16) * 2 + (lane % 16) // 8
check('bcast_idx2', outs['bidx'], bidx)
check('selr(cg, bidx)', outs['selr'], exp_cg[bidx])

# 4x4 transpose within each 16-lane token
tidx = (lane // 16) * 16 + (lane % 4) * 4 + (lane % 16) // 4
check('transpose(t_idx)', outs['tr'], lane[tidx])
