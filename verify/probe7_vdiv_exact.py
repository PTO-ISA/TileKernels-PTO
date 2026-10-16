"""Probe #7b: pairwise sums (bitwise-confirmed ASC order) + vdiv default vs
precision='exact', compared against the ASC stage dump."""
import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

torch.npu.set_device(0)
DEV = "npu"

OUTS = ['rsum', 'sm_hw', 'sm_exact']


@tilelang.jit(target='pto')
def probe_vmi7b(x, o_rsum, o_sm_hw, o_sm_exact):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rsum: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm_hw: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm_exact: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        rsum_ub = T.alloc_shared((4, 4, 4), T.float32)
        smh_ub = T.alloc_shared((4, 4, 4), T.float32)
        sme_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)
            idx = V.vci(T.int32(0), size=64)
            one = V.vbrc(T.int32(1), size=64)
            two = V.vbrc(T.int32(2), size=64)
            pair_swap_idx = V.vxor(idx, one)
            row_base = V.vand(idx, V.vbrc(T.int32(-4), size=64))
            cross_pair_idx = V.vadd(V.vsub(V.vadd(row_base, two), V.vand(idx, two)), V.vand(idx, one))
            four = T.int32(4)
            fifteen = V.vbrc(T.int32(15), size=64)
            local = V.vand(idx, fifteen)
            token_base_tidx = V.vshls(V.vshrs(idx, four, mask_full), four, mask_full)
            col = V.vand(idx, V.vbrc(T.int32(3), size=64))
            col4 = V.vshls(col, T.int32(2), mask_full)
            t_idx = V.vadd(V.vadd(token_base_tidx, col4), V.vshrs(local, T.int32(2), mask_full))

            # row max: 2-round vselr/vmax tree (verified bitwise vs ASC)
            mat_0 = V.vload(x_ub[0, 0, 0], size=64)
            rmax_b_0 = V.vmax(mat_0, V.vselr(mat_0, pair_swap_idx))
            rmax_b_0 = V.vmax(rmax_b_0, V.vselr(rmax_b_0, cross_pair_idx))
            mat_1 = V.vexpdif(mat_0, rmax_b_0, mask_full)

            # row sum: pairwise tree (a+b)+(c+d), the confirmed ASC vcgadd order
            pair = V.vadd(mat_1, V.vselr(mat_1, pair_swap_idx))
            rsum_b_0 = V.vadd(pair, V.vselr(pair, cross_pair_idx))

            sm_hw = V.vdiv(mat_1, rsum_b_0)
            sm_exact = V.vdiv(mat_1, rsum_b_0, mask_full, precision='exact')

            V.vstore(rsum_b_0, rsum_ub[0, 0, 0])
            V.vstore(sm_hw, smh_ub[0, 0, 0])
            V.vstore(sm_exact, sme_ub[0, 0, 0])

        T.copy(rsum_ub, o_rsum[0, 0, 0])
        T.copy(smh_ub, o_sm_hw[0, 0, 0])
        T.copy(sme_ub, o_sm_exact[0, 0, 0])


art = torch.load("/data/sunkexin/tkvmi/work/ab_artifacts/sinkhorn_asc_stages.pt", weights_only=True)
x = art["x"].to(DEV)
outs = {k: torch.full_like(x, float('nan')) for k in OUTS}
probe_vmi7b(x, *[outs[k] for k in OUTS])
torch.npu.synchronize()


def ulpreport(name, got, ref):
    got = got.flatten().cpu()
    ref = ref.flatten().cpu()
    same = torch.equal(got, ref)
    n = int((got != ref).sum())
    print(name + (": BITWISE EQUAL" if same else (": diff " + str(n) + "/64")))


ulpreport('rsum (pairwise tree) vs ASC rsum', outs['rsum'], art['rsum'])
ulpreport('sm (vdiv hw) vs ASC sm', outs['sm_hw'], art['sm'])
ulpreport('sm (vdiv exact) vs ASC sm', outs['sm_exact'], art['sm'])
