"""Probe #3: the vselr/vmax row-max tree and its integer index vectors.

Probe #2 showed vcmax(mask, group=8) is numerically wrong; the kernel now uses
a two-round vselr/vmax tree.  This probe verifies that tree in isolation on
ramp / negative ramp / random data, plus the pair_swap / cross_pair index
vectors (vxor, vand with the negative constant -4, int vsub).
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

PROBES = ['i_swap', 'i_cross', 'rmax_ramp', 'rmax_neg', 'rmax_rand', 'rmax_x']


@tilelang.jit(target='pto')
def probe_vmi3(x, o_i_swap, o_i_cross, o_rmax_ramp, o_rmax_neg, o_rmax_rand, o_rmax_x):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_i_swap: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_i_cross: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rmax_ramp: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rmax_neg: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rmax_rand: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    o_rmax_x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(1):
        x_ub = T.alloc_shared((4, 4, 4), T.float32)
        i_swap_ub = T.alloc_shared((4, 4, 4), T.float32)
        i_cross_ub = T.alloc_shared((4, 4, 4), T.float32)
        rmax_ramp_ub = T.alloc_shared((4, 4, 4), T.float32)
        rmax_neg_ub = T.alloc_shared((4, 4, 4), T.float32)
        rmax_rand_ub = T.alloc_shared((4, 4, 4), T.float32)
        rmax_x_ub = T.alloc_shared((4, 4, 4), T.float32)
        T.copy(x[0, 0, 0], x_ub)

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)
            idx = V.vci(T.int32(0), size=64)
            idx_f = V.vcvt(idx, 'float32')

            one = V.vbrc(T.int32(1), size=64)
            pair_swap_idx = V.vxor(idx, one)
            row_base = V.vand(idx, V.vbrc(T.int32(-4), size=64))
            two = V.vbrc(T.int32(2), size=64)
            cross_pair_idx = V.vadd(
                V.vsub(V.vadd(row_base, two), V.vand(idx, two)),
                V.vand(idx, one),
            )
            V.vstore(V.vcvt(pair_swap_idx, 'float32'), i_swap_ub[0, 0, 0])
            V.vstore(V.vcvt(cross_pair_idx, 'float32'), i_cross_ub[0, 0, 0])

            neg = V.vsub(V.vbrc(T.float32(0.0), size=64), idx_f)
            mat = V.vload(x_ub[0, 0, 0], size=64)

            for dst, v in ((rmax_ramp_ub, idx_f), (rmax_neg_ub, neg), (rmax_rand_ub, mat)):
                pair = V.vmax(v, V.vselr(v, pair_swap_idx))
                V.vstore(V.vmax(pair, V.vselr(pair, cross_pair_idx)), dst[0, 0, 0])

        T.copy(i_swap_ub, o_i_swap[0, 0, 0])
        T.copy(i_cross_ub, o_i_cross[0, 0, 0])
        T.copy(rmax_ramp_ub, o_rmax_ramp[0, 0, 0])
        T.copy(rmax_neg_ub, o_rmax_neg[0, 0, 0])
        T.copy(rmax_rand_ub, o_rmax_rand[0, 0, 0])
        T.copy(rmax_x_ub, o_rmax_x[0, 0, 0])


import numpy as np

n = 4
g = torch.Generator(device="cpu").manual_seed(42)
x = torch.randn((n, 4, 4), generator=g, dtype=torch.float32).to(DEV)
outs = {k: torch.full((n, 4, 4), float('nan'), dtype=torch.float32, device=DEV) for k in PROBES}

probe_vmi3(x, *[outs[k] for k in PROBES])
torch.npu.synchronize()


def check(name, got, exp):
    got = got.flatten().cpu()
    exp = torch.tensor(np.asarray(exp, dtype=np.float32))
    ok = torch.equal(got, exp)
    if ok:
        print(f"{name}: OK (bitwise)")
        return
    close = torch.isclose(got, exp, rtol=1e-6, atol=1e-6)
    bad = (~close).sum().item()
    idxs = (~close).nonzero().flatten()[:6].tolist()
    tag = "OK (~1e-6)" if bad == 0 else f"MISMATCH ({bad}/64)"
    print(f"{name}: {tag}" + ("" if bad == 0 else f" lanes={idxs} got={[got[i].item() for i in idxs]} exp={[exp[i].item() for i in idxs]}"))


lane = np.arange(64)
check('pair_swap_idx (idx^1)', outs['i_swap'], lane ^ 1)
check('cross_pair_idx', outs['i_cross'], (lane & ~3) + 2 - (lane & 2) + (lane & 1))

rowmax_of = lambda arr: np.array([arr[16 * (l // 16) + 4 * ((l % 16) // 4): 16 * (l // 16) + 4 * ((l % 16) // 4) + 4].max() for l in lane])
check('rmax_ramp', outs['rmax_ramp'], rowmax_of(lane.astype(np.float32)))
check('rmax_neg', outs['rmax_neg'], rowmax_of((lane.astype(np.float32) - 63.0)))
xh = x.flatten().cpu().numpy()
check('rmax_rand', outs['rmax_rand'], rowmax_of(xh))
