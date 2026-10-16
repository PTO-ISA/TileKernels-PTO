"""Probe #6 (ASC side, run with the tl-run env): stage dump of the ASC fwd
kernel copy (S.* ops) for cross-env comparison with the VMI stage dump.
Saves rmax/e/sm/tr0/mat4 for one 4-token block to a .pt artifact."""

import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import tilelang
import torch
import torch_npu  # noqa: F401
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

torch.npu.set_device(0)
DEV = "npu"

STAGES = ['rmax', 'e', 'rsum', 'sm', 'tr0', 'mat4']


@tilelang.jit
def probe_asc_stages(x, o_rmax, o_e, o_rsum, o_sm, o_tr0, o_mat4):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rmax: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_e: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rsum: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_sm: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_tr0: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_mat4: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        rmax_ub = T.alloc_shared((4, 4, 4), T.float32)
        e_ub = T.alloc_shared((4, 4, 4), T.float32)
        rsum_ub = T.alloc_shared((4, 4, 4), T.float32)
        sm_ub = T.alloc_shared((4, 4, 4), T.float32)
        tr0_ub = T.alloc_shared((4, 4, 4), T.float32)
        mat4_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = S.pset(32)
            idx = S.vci(T.int32(0), T.int32)
            const_7 = S.vdup(T.int32(7), T.int32, mask_full)
            mod8 = S.vand(idx, const_7, mask_full)
            mask_lo = S.vcmps(mod8, T.int32(4), mask_full, 'lt')
            mask_hi = S.vcmps(mod8, T.int32(4), mask_full, 'ge')
            two = T.int32(2)
            three = T.int32(3)
            four = T.int32(4)
            const_15 = S.vdup(T.int32(15), T.int32, mask_full)
            token2 = S.vshls(S.vshrs(idx, four, mask_full), T.int32(1), mask_full)
            local = S.vand(idx, const_15, mask_full)
            local_div8 = S.vshrs(local, three, mask_full)
            bcast_idx2 = S.vadd(token2, local_div8, mask_full)
            token_base_tidx = S.vshls(S.vshrs(idx, four, mask_full), four, mask_full)
            col = S.vand(idx, S.vdup(T.int32(3), T.int32, mask_full), mask_full)
            col4 = S.vshls(col, two, mask_full)
            t_idx = S.vadd(S.vadd(token_base_tidx, col4, mask_full), S.vshrs(local, two, mask_full), mask_full)

            mat_0 = S.vld(x_ub[0, 0, 0])

            rmax_b_0 = S.vsel(
                S.vselr(S.vcgmax(mat_0, mask_lo), bcast_idx2),
                S.vselr(S.vcgmax(mat_0, mask_hi), bcast_idx2),
                mask_lo,
            )
            mat_1 = S.vexpdif(mat_0, rmax_b_0, mask_full)
            rsum_b_0 = S.vsel(
                S.vselr(S.vcgadd(mat_1, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(mat_1, mask_hi), bcast_idx2),
                mask_lo,
            )
            mat_2 = S.vdiv(mat_1, rsum_b_0, mask_full)
            mat_3 = S.vadds(mat_2, T.float32(1e-6), mask_full)

            tr_0 = S.vselr(mat_3, t_idx)
            csum_b_0 = S.vsel(
                S.vselr(S.vcgadd(tr_0, mask_lo), bcast_idx2),
                S.vselr(S.vcgadd(tr_0, mask_hi), bcast_idx2),
                mask_lo,
            )
            tr_1 = S.vdiv(tr_0, S.vadds(csum_b_0, T.float32(1e-6), mask_full), mask_full)
            mat_4 = S.vselr(tr_1, t_idx)

            S.vsts(rmax_ub[0, 0, 0], rmax_b_0, mask_full)
            S.vsts(e_ub[0, 0, 0], mat_1, mask_full)
            S.vsts(rsum_ub[0, 0, 0], rsum_b_0, mask_full)
            S.vsts(sm_ub[0, 0, 0], mat_2, mask_full)
            S.vsts(tr0_ub[0, 0, 0], tr_0, mask_full)
            S.vsts(mat4_ub[0, 0, 0], mat_4, mask_full)

        T.copy(rmax_ub, o_rmax[0, 0, 0])
        T.copy(e_ub, o_e[0, 0, 0])
        T.copy(rsum_ub, o_rsum[0, 0, 0])
        T.copy(sm_ub, o_sm[0, 0, 0])
        T.copy(tr0_ub, o_tr0[0, 0, 0])
        T.copy(mat4_ub, o_mat4[0, 0, 0])


n = 4
g = torch.Generator(device="cpu").manual_seed(42)
x = torch.randn((n, 4, 4), generator=g, dtype=torch.float32).to(DEV)
outs = {k: torch.full((n, 4, 4), float('nan'), dtype=torch.float32, device=DEV) for k in STAGES}
probe_asc_stages(x, *[outs[k] for k in STAGES])
torch.npu.synchronize()

torch.save({"x": x.cpu(), **{k: outs[k].cpu() for k in STAGES}}, "/data/sunkexin/tkvmi/work/ab_artifacts/sinkhorn_asc_stages.pt")
for k in STAGES:
    print(k, "saved", outs[k].flatten()[:4].tolist())
print("ASC stage dump complete")
