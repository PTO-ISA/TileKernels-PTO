"""Two-row KV quant, the batched SimdVF. PTO unrolls the rows.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py#L184
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from cases.norm.quant_asc import CLAMP, GROUPS, HIDDEN, QMAX

ROWS = 2


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((ROWS, HIDDEN), T.bfloat16),
        q: T.Tensor((ROWS, HIDDEN), T.float8_e4m3fn),
        sf: T.Tensor((ROWS, GROUPS), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((ROWS, HIDDEN), T.bfloat16)
            q_ub = T.alloc_shared((ROWS, HIDDEN), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((ROWS, 64), T.float32)
            inv_ub = T.alloc_shared((ROWS, 64), T.float32)
            T.copy(x, x_ub)
            with T.SimdVF():
                low = S.pset(32, "PAT_VL32")
                all_lanes = S.pset(32, "PAT_ALL")
                high = S.pnot(low, all_lanes)
                qmax = S.vdup(QMAX, T.float32)
                zero = S.vdup(0.0, T.float32)
                for row in T.unroll(ROWS, explicit=True):
                    S.vsts(amax_ub[row, 0], zero)
                    v0 = S.vabs(S.vcvt(S.vld(x_ub[row, 0], dist="UNPK_B16"), T.float32, part=0))
                    v1 = S.vabs(S.vcvt(S.vld(x_ub[row, 64], dist="UNPK_B16"), T.float32, part=0))
                    S.vsts(amax_ub[row, 0], S.vcmax(v0, low), dist="ONEPT_B32")
                    S.vsts(amax_ub[row, 1], S.vcmax(v0, high), dist="ONEPT_B32")
                    S.vsts(amax_ub[row, 2], S.vcmax(v1, low), dist="ONEPT_B32")
                    S.vsts(amax_ub[row, 3], S.vcmax(v1, high), dist="ONEPT_B32")
                S.mem_bar("VST_VLD")
                for row in T.unroll(ROWS, explicit=True):
                    clamped = S.vmaxs(S.vld(amax_ub[row, 0]), CLAMP)
                    S.vsts(inv_ub[row, 0], S.vdiv(qmax, clamped))
                    S.vsts(amax_ub[row, 0], S.vdiv(clamped, qmax))
                S.mem_bar("VST_VLD")
                for row in T.unroll(ROWS, explicit=True):
                    i0 = S.vld(inv_ub[row, 0], dist="BRC_B32")
                    i1 = S.vld(inv_ub[row, 1], dist="BRC_B32")
                    i2 = S.vld(inv_ub[row, 2], dist="BRC_B32")
                    i3 = S.vld(inv_ub[row, 3], dist="BRC_B32")
                    x0 = S.vcvt(S.vld(x_ub[row, 0], dist="UNPK_B16"), T.float32, part=0)
                    x1 = S.vcvt(S.vld(x_ub[row, 64], dist="UNPK_B16"), T.float32, part=0)
                    S.vsts(q_ub[row, 0], S.vcvt(S.vmul(x0, S.vsel(i0, i1, low)), T.float8_e4m3fn), dist="PK4_B32")
                    S.vsts(q_ub[row, 64], S.vcvt(S.vmul(x1, S.vsel(i2, i3, low)), T.float8_e4m3fn), dist="PK4_B32")
            T.copy(q_ub, q)
            T.copy(amax_ub[0:ROWS, 0:GROUPS], sf)

    return kernel


_KERNEL = None


def launch(x):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = x.npu()
    q = torch.empty((ROWS, HIDDEN), dtype=torch.float8_e4m3fn, device=src.device)
    sf = torch.empty((ROWS, GROUPS), device=src.device)
    _KERNEL(src, q, sf)
    return q.cpu(), sf.cpu()
