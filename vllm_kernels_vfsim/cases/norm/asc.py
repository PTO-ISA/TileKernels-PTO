"""RMSNorm and the rstd helper used inside the norm SimdVF.

Sources:
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L154
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/rstd_1ulp.py
"""

import struct

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 128
EPS = 1e-6


def _split(value: float) -> tuple[float, float]:
    high = struct.unpack("=f", struct.pack("=f", value))[0]
    low = struct.unpack("=f", struct.pack("=f", value - high))[0]
    return high, low


_INV_H_HI, _INV_H_LO = _split(1.0 / HIDDEN)
_EPS_HI, _EPS_LO = _split(EPS)


@tilelang.jit(target="ascend")
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
            sq_ub = T.alloc_shared((8,), T.float32)
            T.copy(x, x_ub)
            T.copy(weight, w_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                acc0 = S.alloc_var(T.float32)
                acc1 = S.alloc_var(T.float32)
                acc0 = S.vdup(0.0, T.float32)
                acc1 = S.vdup(0.0, T.float32)
                v0 = S.vld(x_ub[0])
                v1 = S.vld(x_ub[64])
                S.vmula(acc0, v0, v0, full)
                S.vmula(acc1, v1, v1, full)
                total = S.vcadd(S.vadd(acc0, acc1, full), full)
                S.vsts(sq_ub[0], total, one, "ONEPT_B32")
                S.mem_bar("VST_VLD")
                square = S.vld(sq_ub[0], dist="BRC_B32")
                inv_h = S.vdup(_INV_H_HI, T.float32)
                variance = S.vadds(S.vmuls(square, _INV_H_HI), _EPS_HI)
                rstd = S.vdiv(S.vdup(1.0, T.float32), S.vsqrt(variance))
                for half in T.serial(2):
                    col = half * 64
                    values = S.vld(x_ub[col])
                    scaled = S.vmul(S.vmul(values, rstd, full), S.vld(w_ub[col]), full)
                    S.vsts(y_ub[col], scaled, full)
            T.copy(y_ub, y)

    return kernel


@tilelang.jit(target="ascend")
def compile_unpack():
    @T.prim_func
    def kernel(src: T.Tensor((HIDDEN,), T.bfloat16), dst: T.Tensor((HIDDEN,), T.float32)):
        with T.Kernel(1):
            src_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            dst_ub = T.alloc_shared((HIDDEN,), T.float32)
            T.copy(src, src_ub)
            with T.SimdVF():
                low = S.vcvt(S.vld(src_ub[0], dist="UNPK_B16"), T.float32, part=0)
                high = S.vcvt(S.vld(src_ub[64], dist="UNPK_B16"), T.float32, part=0)
                S.vsts(dst_ub[0], low)
                S.vsts(dst_ub[64], high)
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
