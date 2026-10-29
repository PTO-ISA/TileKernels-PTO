"""PTO compensated rstd. Same FMA chain as the ASC helper, logical vectors."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.norm.rstd_asc import EPS_HI, EPS_LO, HIDDEN, INV_HI, INV_LO


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(x: T.Tensor((HIDDEN,), T.float32), y: T.Tensor((64,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((HIDDEN,), T.float32)
            out = T.alloc_shared((64,), T.float32)
            T.copy(x, ub)
            with T.SimdVF():
                mask = V.create_mask(64, size=64)
                a = V.vload(ub[0], size=64)
                b = V.vload(ub[64], size=64)
                acc = V.vadd(V.vmul(a, a, mask), V.vmul(b, b, mask), mask)
                square = V.vbrc(V.vcadd(acc, mask, reassoc=True), size=64)
                product = V.vmuls(square, T.float32(INV_HI), mask)
                value = V.vadds(product, T.float32(EPS_HI), mask)
                carried = V.vsub(value, product, mask)
                product_sum_error = V.vsub(product, V.vsub(value, carried, mask), mask)
                c1_sum_error = V.vadds(V.vneg(carried, mask), T.float32(EPS_HI), mask)
                sum_error = V.vadd(product_sum_error, c1_sum_error, mask)
                product_error = V.vmula(V.vneg(product, mask), square, V.vbrc(T.float32(INV_HI), size=64), mask)
                value_error = V.vadd(product_error, sum_error, mask)
                value_error = V.vmula(value_error, square, V.vbrc(T.float32(INV_LO), size=64), mask)
                value_error = V.vadds(value_error, T.float32(EPS_LO), mask)
                estimate = V.vdiv(V.vbrc(T.float32(1.0), size=64), V.vsqrt(value, mask), mask)
                est2 = V.vmul(estimate, estimate, mask)
                square_error = V.vmula(V.vneg(est2, mask), estimate, estimate, mask)
                residual = V.vbrc(T.float32(1.0), size=64)
                neg_value = V.vneg(value, mask)
                neg_error = V.vneg(value_error, mask)
                residual = V.vmula(residual, neg_value, est2, mask)
                residual = V.vmula(residual, neg_value, square_error, mask)
                residual = V.vmula(residual, neg_error, est2, mask)
                residual = V.vmula(residual, neg_error, square_error, mask)
                refined = V.vmula(estimate, V.vmuls(estimate, T.float32(0.5), mask), residual, mask)
                V.vstore(refined, out[0], mask)
            T.copy(out, y)

    return kernel


_KERNEL = None


def launch(x):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    src = x.npu()
    import torch

    y = torch.empty(64, device=src.device)
    _KERNEL(src, y)
    return y.cpu()[:1]
