"""PTO bf16 quant: one grouped vcmax, then one vcvt to FP8."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.norm.bf16_asc import GROUPS, HIDDEN

QMAX = 448.0
CLAMP = 1e-4


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.bfloat16),
        amax: T.Tensor((GROUPS,), T.float32),
        q: T.Tensor((HIDDEN,), T.float8_e4m3fn),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            amax_ub = T.alloc_shared((GROUPS,), T.float32)
            inv_ub = T.alloc_shared((GROUPS,), T.float32)
            q_ub = T.alloc_shared((HIDDEN,), T.float8_e4m3fn)
            T.copy(x, ub)
            with T.SimdVF():
                mask = V.create_mask(HIDDEN, size=HIDDEN)
                mask_g = V.create_mask(GROUPS, size=GROUPS)
                values = V.vcvt(V.vload(ub[0], size=HIDDEN), "float32")
                reduced = V.vcmax(V.vabs(values, mask), mask, group=GROUPS)
                qmax = V.vbrc(T.float32(QMAX), size=GROUPS)
                clamped = V.vmax(reduced, V.vbrc(T.float32(CLAMP), size=GROUPS), mask_g)
                V.vstore(reduced, amax_ub[0])
                V.vstore(V.vdiv(qmax, clamped, mask_g), inv_ub[0])
                T.simd.mem_bar("VST_VLD")
                slots = V.vload(inv_ub[0], size=GROUPS, stride=1, group=GROUPS)
                inv = V.vbrc(slots, size=HIDDEN, group=GROUPS)
                quant = V.vmul(values, inv, mask)
                V.vstore(V.vcvt(quant, "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[0])
            T.copy(amax_ub, amax)
            T.copy(q_ub, q)

    return kernel


_KERNEL = None


def launch(x):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = x.npu()
    amax = torch.empty(GROUPS, device=src.device)
    q = torch.empty(HIDDEN, dtype=torch.float8_e4m3fn, device=src.device)
    _KERNEL(src, amax, q)
    return amax.cpu(), q.cpu()
