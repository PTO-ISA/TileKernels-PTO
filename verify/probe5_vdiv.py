"""Probe #5: same-kernel dump of rsum_b and its vdiv consumption.

probe4 showed the row-sum broadcast stores correctly while the softmax
(vdiv of the same values) is wrong for lanes 8+.  This probe dumps, in ONE
kernel: rsum_b, sm = vdiv(e, rsum_b), sm_masked = vdiv(e, rsum_b, mask_full),
and sm_reloaded = vdiv(e, rsum_roundtrip) where rsum_roundtrip goes through a
UB store+load -- to pinpoint whether the div or the operand production is
miscompiled when consumed.
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

STAGES = ['rsum', 'sm', 'sm_masked', 'sm_reloaded']


@tilelang.jit(target='pto')
def probe_vmi6(x, o_rsum, o_sm, o_sm_masked, o_sm_reloaded):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rsum: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm_masked: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm_reloaded: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        rsum_ub = T.alloc_shared((4, 4, 4), T.float32)
        sm_ub = T.alloc_shared((4, 4, 4), T.float32)
        smm_ub = T.alloc_shared((4, 4, 4), T.float32)
        smr_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)
            idx = V.vci(T.int32(0), size=64)
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
            one = V.vbrc(T.int32(1), size=64)
            pair_swap_idx = V.vxor(idx, one)
            row_base = V.vand(idx, V.vbrc(T.int32(-4), size=64))
            two = V.vbrc(T.int32(2), size=64)
            cross_pair_idx = V.vadd(V.vsub(V.vadd(row_base, two), V.vand(idx, two)), V.vand(idx, one))

            mat_0 = V.vload(x_ub[0, 0, 0], size=64)
            rmax_b_0 = V.vmax(mat_0, V.vselr(mat_0, pair_swap_idx))
            rmax_b_0 = V.vmax(rmax_b_0, V.vselr(rmax_b_0, cross_pair_idx))
            mat_1 = V.vexpdif(mat_0, rmax_b_0, mask_full)
            # vcadd(group=8) + grouped vbrc + vselr/vsel chain (miscompile trigger)
            lo64 = V.vbrc(V.vcadd(mat_1, mask_lo, group=8, reassoc=True), size=64, group=8)
            hi64 = V.vbrc(V.vcadd(mat_1, mask_hi, group=8, reassoc=True), size=64, group=8)
            rsum_b_0 = V.vsel(
                mask_lo,
                V.vselr(lo64, bcast_idx2),
                V.vselr(hi64, bcast_idx2),
            )

            V.vstore(rsum_b_0, rsum_ub[0, 0, 0])

            sm = V.vdiv(mat_1, rsum_b_0)
            sm_masked = V.vdiv(mat_1, rsum_b_0, mask_full)

            rsum_rt = V.vload(rsum_ub[0, 0, 0], size=64)
            sm_reloaded = V.vdiv(mat_1, rsum_rt)

            V.vstore(sm, sm_ub[0, 0, 0])
            V.vstore(sm_masked, smm_ub[0, 0, 0])
            V.vstore(sm_reloaded, smr_ub[0, 0, 0])

        T.copy(rsum_ub, o_rsum[0, 0, 0])
        T.copy(sm_ub, o_sm[0, 0, 0])
        T.copy(smm_ub, o_sm_masked[0, 0, 0])
        T.copy(smr_ub, o_sm_reloaded[0, 0, 0])


import numpy as np

n = 4
g = torch.Generator(device="cpu").manual_seed(42)
x = torch.randn((n, 4, 4), generator=g, dtype=torch.float32).to(DEV)
outs = {k: torch.full((n, 4, 4), float('nan'), dtype=torch.float32, device=DEV) for k in STAGES}
probe_vmi6(x, *[outs[k] for k in STAGES])
torch.npu.synchronize()

xh = x.flatten().cpu().numpy()


def report(name, got, exp, tol=2e-5):
    got = got.flatten().cpu().numpy()
    exp = np.asarray(exp, dtype=np.float32)
    d = np.abs(got.astype(np.float64) - exp.astype(np.float64))
    bad = d > tol
    print(f"{name}: {'OK' if not bad.any() else f'MISMATCH {bad.sum()}/64 maxdiff={d.max():.3e}'}")
    if bad.any():
        idxs = np.nonzero(bad)[0][:6]
        print(f"   lanes={idxs.tolist()} got={got[idxs].tolist()} exp={exp[idxs].tolist()}")


rmax = np.repeat(xh.reshape(4, 4, 4).max(axis=2), 4, axis=1).reshape(-1)
e = np.exp(xh - rmax)
rsum = np.repeat(e.reshape(4, 4, 4).sum(axis=2), 4, axis=1).reshape(-1)
report('rsum', outs['rsum'], rsum, tol=1e-5)
report('sm = vdiv(e, rsum)', outs['sm'], e / rsum)
report('sm_masked = vdiv(e, rsum, mask)', outs['sm_masked'], e / rsum)
report('sm_reloaded = vdiv(e, rsum_rt)', outs['sm_reloaded'], e / rsum)
