"""PTO FP8 quant-only head. One grouped vcmax and one vcvt."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.norm.quant_asc import CLAMP, GROUPS, HIDDEN, QMAX


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.float32),
        q: T.Tensor((HIDDEN,), T.float8_e4m3fn),
        sf: T.Tensor((GROUPS,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HIDDEN,), T.float32)
            q_ub = T.alloc_shared((HIDDEN,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((GROUPS,), T.float32)
            inv_ub = T.alloc_shared((GROUPS,), T.float32)
            T.copy(x, x_ub)
            with T.SimdVF():
                mask = V.create_mask(HIDDEN, size=HIDDEN)
                mask4 = V.create_mask(GROUPS, size=GROUPS)
                values = V.vload(x_ub[0], size=HIDDEN)
                amax = V.vcmax(V.vabs(values, mask), mask, group=GROUPS)
                clamped = V.vmax(amax, V.vbrc(T.float32(CLAMP), size=GROUPS), mask4)
                qmax = V.vbrc(T.float32(QMAX), size=GROUPS)
                V.vstore(V.vdiv(clamped, qmax, mask4), amax_ub[0])
                V.vstore(V.vdiv(qmax, clamped, mask4), inv_ub[0])
                T.simd.mem_bar("VST_VLD")
                slots = V.vload(inv_ub[0], size=GROUPS, stride=1, group=GROUPS)
                inv = V.vbrc(slots, size=HIDDEN, group=GROUPS)
                quant = V.vmul(values, inv, mask)
                V.vstore(V.vcvt(quant, "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[0])
            T.copy(q_ub, q)
            T.copy(amax_ub, sf)

    return kernel


_KERNEL = None


def launch(x):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = x.npu()
    q = torch.empty(HIDDEN, dtype=torch.float8_e4m3fn, device=src.device)
    sf = torch.empty(GROUPS, device=src.device)
    _KERNEL(src, q, sf)
    return q.cpu(), sf.cpu()
