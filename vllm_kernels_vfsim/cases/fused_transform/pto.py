"""PTO fused norm. One 128-lane convert replaces the paired UNPK loads."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_transform.asc import EPS, HIDDEN


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.bfloat16),
        weight: T.Tensor((HIDDEN,), T.bfloat16),
        y: T.Tensor((HIDDEN,), T.bfloat16),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            w_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            y_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            T.copy(x, x_ub)
            T.copy(weight, w_ub)
            with T.SimdVF():
                mask = V.create_mask(HIDDEN, size=HIDDEN)
                values = V.vcvt(V.vload(x_ub[0], size=HIDDEN), "float32")
                weight_v = V.vcvt(V.vload(w_ub[0], size=HIDDEN), "float32")
                square = V.vbrc(V.vcadd(V.vmul(values, values, mask), mask, reassoc=True), size=HIDDEN)
                inv = V.vbrc(T.float32(1.0 / HIDDEN), size=HIDDEN)
                eps = V.vbrc(T.float32(EPS), size=HIDDEN)
                variance = V.vadd(V.vmul(square, inv, mask), eps, mask)
                rstd = V.vdiv(V.vbrc(T.float32(1.0), size=HIDDEN), V.vsqrt(variance, mask), mask)
                out = V.vmul(V.vmul(values, rstd, mask), weight_v, mask)
                V.vstore(V.vcvt(out, "bfloat16", rounding="R"), y_ub[0])
            T.copy(y_ub, y)

    return kernel


_KERNEL = None


def launch(x, weight):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    x_n = x.npu()
    y = torch.empty_like(x_n)
    _KERNEL(x_n, weight.npu(), y)
    return y.cpu()
