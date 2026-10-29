"""Compensated rstd inlined into the norm VF. Not a separate SimdVF in DeepInfer.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/rstd_1ulp.py
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L154
"""

import struct

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 128
EPS = 1e-5


def _split(value: float):
    high = struct.unpack("=f", struct.pack("=f", value))[0]
    low = struct.unpack("=f", struct.pack("=f", value - high))[0]
    return high, low


INV_HI, INV_LO = _split(1.0 / HIDDEN)
EPS_HI, EPS_LO = _split(EPS)


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(x: T.Tensor((HIDDEN,), T.float32), y: T.Tensor((1,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((HIDDEN,), T.float32)
            out = T.alloc_shared((8,), T.float32)
            T.copy(x, ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                acc = S.alloc_var(T.float32)
                acc = S.vdup(0.0, T.float32)
                acc = S.vadd(acc, S.vmul(S.vld(ub[0]), S.vld(ub[0]), full), full)
                acc = S.vadd(acc, S.vmul(S.vld(ub[64]), S.vld(ub[64]), full), full)
                square = S.vdupv(S.vcadd(acc, full), full, "POS_LOWEST")
                recip_hi = S.vdup(INV_HI, T.float32)
                product = S.vmuls(square, INV_HI, full)
                value = S.vadds(product, EPS_HI, full)
                carried = S.vsub(value, product, full)
                product_sum_error = S.vsub(product, S.vsub(value, carried, full), full)
                c1_sum_error = S.vadds(S.vneg(carried, full), EPS_HI, full)
                sum_error = S.vadd(product_sum_error, c1_sum_error, full)
                product_error = S.alloc_var(T.float32)
                product_error = S.vneg(product, full)
                S.vmula(product_error, square, recip_hi, full)
                value_error = S.alloc_var(T.float32)
                value_error = S.vadd(product_error, sum_error, full)
                S.vmula(value_error, square, S.vdup(INV_LO, T.float32), full)
                value_error = S.vadds(value_error, EPS_LO, full)
                estimate = S.vdiv(S.vdup(1.0, T.float32), S.vsqrt(value), full)
                est2 = S.vmul(estimate, estimate, full)
                square_error = S.alloc_var(T.float32)
                square_error = S.vneg(est2, full)
                S.vmula(square_error, estimate, estimate, full)
                residual = S.alloc_var(T.float32)
                residual = S.vdup(1.0, T.float32)
                neg_value = S.vneg(value, full)
                neg_error = S.vneg(value_error, full)
                S.vmula(residual, neg_value, est2, full)
                S.vmula(residual, neg_value, square_error, full)
                S.vmula(residual, neg_error, est2, full)
                S.vmula(residual, neg_error, square_error, full)
                refined = S.alloc_var(T.float32)
                refined = estimate
                S.vmula(refined, S.vmuls(estimate, 0.5, full), residual, full)
                S.vsts(out[0], refined, one, "ONEPT_B32")
            T.copy(out[0:1], y)

    return kernel


_KERNEL = None


def launch(x):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    src = x.npu()
    import torch

    y = torch.empty(1, device=src.device)
    _KERNEL(src, y)
    return y.cpu()
