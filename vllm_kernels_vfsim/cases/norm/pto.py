"""PTO RMSNorm and bf16 weight unpack.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L154
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L374

The two half-register converts are one ``vcvt`` of 128 bfloat16 lanes.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.norm.asc import EPS, HIDDEN

_INV_H = 1.0 / HIDDEN


@tilelang.jit(target="pto")
def compile_rms():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.float32),
        weight: T.Tensor((HIDDEN,), T.float32),
        y: T.Tensor((HIDDEN,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HIDDEN,), T.float32)
            w_ub = T.alloc_shared((HIDDEN,), T.float32)
            y_ub = T.alloc_shared((HIDDEN,), T.float32)
            T.copy(x, x_ub)
            T.copy(weight, w_ub)
            with T.SimdVF():
                mask = V.create_mask(64, size=64)
                v0 = V.vload(x_ub[0], size=64)
                v1 = V.vload(x_ub[64], size=64)
                acc = V.vadd(V.vmul(v0, v0, mask), V.vmul(v1, v1, mask), mask)
                square = V.vbrc(V.vcadd(acc, mask, reassoc=True), size=64)
                inv = V.vbrc(T.float32(_INV_H), size=64)
                eps = V.vbrc(T.float32(EPS), size=64)
                variance = V.vadd(V.vmul(square, inv, mask), eps, mask)
                rstd = V.vdiv(V.vbrc(T.float32(1.0), size=64), V.vsqrt(variance, mask), mask)
                for half in T.serial(2):
                    col = half * 64
                    values = V.vload(x_ub[col], size=64)
                    weight_v = V.vload(w_ub[col], size=64)
                    V.vstore(V.vmul(V.vmul(values, rstd, mask), weight_v, mask), y_ub[col], mask)
            T.copy(y_ub, y)

    return kernel


@tilelang.jit(target="pto")
def compile_unpack():
    @T.prim_func
    def kernel(src: T.Tensor((HIDDEN,), T.bfloat16), dst: T.Tensor((HIDDEN,), T.float32)):
        with T.Kernel(1):
            src_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            dst_ub = T.alloc_shared((HIDDEN,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                V.vstore(V.vcvt(V.vload(src_ub[0], size=HIDDEN), "float32"), dst_ub[0])
            T.copy(dst_ub, dst)

    return kernel


_RMS = None
_UNPACK = None


def launch_rms(x, weight):
    global _RMS
    if _RMS is None:
        _RMS = compile_rms()
    x_n = x.npu()
    y = x_n.new_empty(x_n.shape)
    _RMS(x_n, weight.npu(), y)
    return y.cpu()


def launch_unpack(src):
    global _UNPACK
    if _UNPACK is None:
        _UNPACK = compile_unpack()
    import torch

    src_n = src.npu()
    dst = torch.empty(HIDDEN, dtype=torch.float32, device=src_n.device)
    _UNPACK(src_n, dst)
    return dst.cpu()
