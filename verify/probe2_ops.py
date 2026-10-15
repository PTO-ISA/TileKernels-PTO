"""Probe #2: vcmax / vexpdif / vdiv and the full sinkhorn Step-1 pipeline.

Probe #1 established vci/vcadd(group)/vbrc(group)/vselr/vcmps/vsel are all
correct.  This probe tests the remaining ops (vcmax with partial mask on
negative data -- ASC vcgmax uses MODE_ZEROING --, vexpdif, vdiv) and a full
row-softmax pipeline against exact torch math.
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

PROBES = ['cmax_pos', 'cmax_neg', 'expdif', 'vdiv', 'step1']


@tilelang.jit(target='pto')
def probe_vmi2(x, o_cmax_pos, o_cmax_neg, o_expdif, o_vdiv, o_step1):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_cmax_pos: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_cmax_neg: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_expdif: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_vdiv: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_step1: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        cmax_pos_ub = T.alloc_shared((4, 4, 4), T.float32)
        cmax_neg_ub = T.alloc_shared((4, 4, 4), T.float32)
        expdif_ub = T.alloc_shared((4, 4, 4), T.float32)
        vdiv_ub = T.alloc_shared((4, 4, 4), T.float32)
        step1_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)

            idx = V.vci(T.int32(0), size=64)
            idx_f = V.vcvt(idx, 'float32')
            seven = V.vbrc(T.int32(7), size=64)
            mod8 = V.vand(idx, seven)
            mask_lo = V.vcmps(mod8, T.int32(4), mask_full, 'lt')
            mask_hi = V.vcmps(mod8, T.int32(4), mask_full, 'ge')
            four = T.int32(4)
            fifteen = V.vbrc(T.int32(15), size=64)
            token2 = V.vshls(V.vshrs(idx, four, mask_full), T.int32(1), mask_full)
            local = V.vand(idx, fifteen)
            local_div8 = V.vshrs(local, T.int32(3), mask_full)
            bcast_idx2 = V.vadd(token2, local_div8)

            mat = V.vload(x_ub[0, 0, 0], size=64)

            # vcmax on positive data (ramp 0..63)
            cmax_pos = V.vsel(
                mask_lo,
                V.vselr(V.vbrc(V.vcmax(idx_f, mask_lo, group=8), size=64, group=8), bcast_idx2),
                V.vselr(V.vbrc(V.vcmax(idx_f, mask_hi, group=8), size=64, group=8), bcast_idx2),
            )
            V.vstore(cmax_pos, cmax_pos_ub[0, 0, 0])

            # vcmax on negative data (ramp -63..0)
            neg = V.vsub(V.vbrc(T.float32(0.0), size=64), idx_f)
            cmax_neg = V.vsel(
                mask_lo,
                V.vselr(V.vbrc(V.vcmax(neg, mask_lo, group=8), size=64, group=8), bcast_idx2),
                V.vselr(V.vbrc(V.vcmax(neg, mask_hi, group=8), size=64, group=8), bcast_idx2),
            )
            V.vstore(cmax_neg, cmax_neg_ub[0, 0, 0])

            # vexpdif(idx, 32) = exp(idx - 32)
            V.vstore(V.vexpdif(idx_f, V.vbrc(T.float32(32.0), size=64), mask_full), expdif_ub[0, 0, 0])

            # vdiv(idx, 64)
            V.vstore(V.vdiv(idx_f, V.vbrc(T.float32(64.0), size=64)), vdiv_ub[0, 0, 0])

            # Full Step-1: row softmax + eps on mat
            rmax_b = V.vsel(
                mask_lo,
                V.vselr(V.vbrc(V.vcmax(mat, mask_lo, group=8), size=64, group=8), bcast_idx2),
                V.vselr(V.vbrc(V.vcmax(mat, mask_hi, group=8), size=64, group=8), bcast_idx2),
            )
            e = V.vexpdif(mat, rmax_b, mask_full)
            rsum_b = V.vsel(
                mask_lo,
                V.vselr(V.vbrc(V.vcadd(e, mask_lo, group=8, reassoc=True), size=64, group=8), bcast_idx2),
                V.vselr(V.vbrc(V.vcadd(e, mask_hi, group=8, reassoc=True), size=64, group=8), bcast_idx2),
            )
            s = V.vdiv(e, rsum_b)
            V.vstore(V.vadd(s, V.vbrc(T.float32(1e-6), size=64)), step1_ub[0, 0, 0])

        T.copy(cmax_pos_ub, o_cmax_pos[0, 0, 0])
        T.copy(cmax_neg_ub, o_cmax_neg[0, 0, 0])
        T.copy(expdif_ub, o_expdif[0, 0, 0])
        T.copy(vdiv_ub, o_vdiv[0, 0, 0])
        T.copy(step1_ub, o_step1[0, 0, 0])


import numpy as np

n = 4
x = torch.arange(64, dtype=torch.float32, device=DEV).reshape(n, 4, 4)
outs = {k: torch.full((n, 4, 4), float('nan'), dtype=torch.float32, device=DEV) for k in PROBES}

probe_vmi2(x, *[outs[k] for k in PROBES])
torch.npu.synchronize()


def check(name, got, exp, bitwise=True):
    got = got.flatten().cpu()
    exp = torch.tensor(np.asarray(exp, dtype=np.float32))
    if torch.equal(got, exp):
        print(f"{name}: OK (bitwise)")
        return
    close = torch.isclose(got, exp, rtol=1e-5, atol=1e-6)
    bad = (~close).sum().item()
    idxs = (~close).nonzero().flatten()[:6].tolist()
    tag = "OK (~1e-5)" if bad == 0 else f"MISMATCH ({bad}/64)"
    print(f"{name}: {tag}" + ("" if bad == 0 else f" lanes={idxs} got={[got[i].item() for i in idxs]} exp={[exp[i].item() for i in idxs]}"))


lane = np.arange(64).astype(np.float32)
# row of lane ℓ = (ℓ % 16) // 4; rowmax over that row's 4 lanes
row = (lane.astype(int) % 16) // 4
rowmax = np.array([lane[16 * (ℓ // 16) + 4 * r: 16 * (ℓ // 16) + 4 * r + 4].max() for ℓ, r in zip(lane.astype(int), row)])
check('cmax_pos (row max of ramp)', outs['cmax_pos'], rowmax)

lane_neg = -lane[::-1]  # lane ℓ = ℓ - 63  (i.e. -63..0 ascending)
rowmax_neg = np.array([lane_neg[16 * (ℓ // 16) + 4 * r: 16 * (ℓ // 16) + 4 * r + 4].max() for ℓ, r in zip(lane.astype(int), row)])
check('cmax_neg (row max of neg ramp)', outs['cmax_neg'], rowmax_neg)

check('expdif', outs['expdif'], np.exp(lane - 32.0), bitwise=False)
check('vdiv', outs['vdiv'], lane / 64.0, bitwise=False)

sm = np.exp(lane - rowmax)
rsum = np.array([sm[16 * (ℓ // 16) + 4 * r: 16 * (ℓ // 16) + 4 * r + 4].sum() for ℓ, r in zip(lane.astype(int), row)])
step1 = sm / rsum + 1e-6
check('step1 (softmax+eps)', outs['step1'], step1, bitwise=False)
