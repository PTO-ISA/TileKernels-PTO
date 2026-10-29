"""ASC dflash VFs: confidence dot, and mean over heads.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L57
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L137

The sigmoid after the dot is T.SimtVF and is not part of this driver.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

ROWS = 4
DIM = 128
HEADS = 2
VEC = 64


@tilelang.jit(target="ascend")
def compile_dot():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, DIM), T.bfloat16),
        weight: T.Tensor((DIM,), T.bfloat16),
        out: T.Tensor((ROWS,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((ROWS, DIM), T.bfloat16)
            w_ub = T.alloc_shared((DIM,), T.bfloat16)
            sum_ub = T.alloc_shared((ROWS,), T.float32)
            T.copy(x, x_ub)
            T.copy(weight, w_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                zero_v = S.vdup(0.0, T.float32)
                acc = S.alloc_local((ROWS,), T.float32)
                for row in T.unroll(ROWS, explicit=True):
                    acc[row] = zero_v
                for i in T.serial(DIM // VEC):
                    off = i * VEC
                    w_v = S.vcvt(S.vld(w_ub[off], dist="UNPK_B16"), T.float32, part=0)
                    for row in T.unroll(ROWS, explicit=True):
                        x_v = S.vcvt(S.vld(x_ub[row, off], dist="UNPK_B16"), T.float32, part=0)
                        S.vmula(acc[row], w_v, x_v, full)
                for row in T.unroll(ROWS, explicit=True):
                    S.vsts(sum_ub[row], S.vcadd(acc[row], full), one, "ONEPT_B32")
            T.copy(sum_ub, out)

    return kernel


@tilelang.jit(target="ascend")
def compile_mean():
    @T.prim_func
    def kernel(
        x: T.Tensor((HEADS, DIM), T.bfloat16),
        out: T.Tensor((DIM,), T.bfloat16),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HEADS, DIM), T.bfloat16)
            T.copy(x, x_ub)
            with T.SimdVF():
                for c in T.serial(DIM // VEC):
                    off = c * VEC
                    acc = S.vcvt(S.vld(x_ub[0, off], dist="UNPK_B16"), T.float32, part=0)
                    val = S.vcvt(S.vld(x_ub[1, off], dist="UNPK_B16"), T.float32, part=0)
                    acc = S.vadd(acc, val)
                    acc = S.vmuls(acc, T.float32(1.0 / HEADS))
                    S.vsts(x_ub[0, off], S.vcvt(acc, T.bfloat16), dist="PK_B32")
            T.copy(x_ub[0, :], out)

    return kernel


_DOT = None
_MEAN = None


def launch_dot(x, weight):
    global _DOT
    if _DOT is None:
        _DOT = compile_dot()
    x_n = x.npu()
    import torch

    out = torch.empty(ROWS, dtype=torch.float32, device=x_n.device)
    _DOT(x_n, weight.npu(), out)
    return out.cpu()


def launch_mean(x):
    global _MEAN
    if _MEAN is None:
        _MEAN = compile_mean()
    x_n = x.npu()
    import torch

    out = torch.empty(DIM, dtype=torch.bfloat16, device=x_n.device)
    _MEAN(x_n, out)
    return out.cpu()
