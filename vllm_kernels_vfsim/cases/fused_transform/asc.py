"""ASC fused norm+store VF, norm enabled, RoPE/Hadamard/quant off.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477

Those other flags reuse the quant path in `norm` and the same fp32 arithmetic.
This shape is one head of 128 bfloat16 values.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 128
EPS = 1e-6


@tilelang.jit(target="ascend")
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
                v0 = S.vcvt(S.vld(x_ub[0], dist="UNPK_B16"), T.float32, part=0)
                v1 = S.vcvt(S.vld(x_ub[64], dist="UNPK_B16"), T.float32, part=0)
                S.vmula(acc0, v0, v0, full)
                S.vmula(acc1, v1, v1, full)
                total = S.vcadd(S.vadd(acc0, acc1, full), full)
                S.vsts(sq_ub[0], total, one, "ONEPT_B32")
                S.mem_bar("VST_VLD")
                square = S.vld(sq_ub[0], dist="BRC_B32")
                variance = S.vadds(S.vmuls(square, 1.0 / HIDDEN), EPS)
                rstd = S.vdiv(S.vdup(1.0, T.float32), S.vsqrt(variance))
                w0 = S.vcvt(S.vld(w_ub[0], dist="UNPK_B16"), T.float32, part=0)
                w1 = S.vcvt(S.vld(w_ub[64], dist="UNPK_B16"), T.float32, part=0)
                y0 = S.vmul(S.vmul(v0, rstd, full), w0, full)
                y1 = S.vmul(S.vmul(v1, rstd, full), w1, full)
                S.vsts(y_ub[0], S.vcvt(y0, T.bfloat16), dist="PK_B32")
                S.vsts(y_ub[64], S.vcvt(y1, T.bfloat16), dist="PK_B32")
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
