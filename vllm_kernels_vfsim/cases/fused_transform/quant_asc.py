"""FP8 quant-only head from fused_transform, head_dim 128.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from cases.norm.quant_asc import CLAMP, GROUPS, HIDDEN, QMAX


@tilelang.jit(target="ascend")
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
            amax_ub = T.alloc_shared((64,), T.float32)
            inv_ub = T.alloc_shared((64,), T.float32)
            T.copy(x, x_ub)
            with T.SimdVF():
                low = S.pset(32, "PAT_VL32")
                all_lanes = S.pset(32, "PAT_ALL")
                high = S.pnot(low, all_lanes)
                v0 = S.vabs(S.vld(x_ub[0]))
                v1 = S.vabs(S.vld(x_ub[64]))
                S.vsts(amax_ub[0], S.vcmax(v0, low), dist="ONEPT_B32")
                S.vsts(amax_ub[1], S.vcmax(v0, high), dist="ONEPT_B32")
                S.vsts(amax_ub[2], S.vcmax(v1, low), dist="ONEPT_B32")
                S.vsts(amax_ub[3], S.vcmax(v1, high), dist="ONEPT_B32")
                S.mem_bar("VST_VLD")
                clamped = S.vmaxs(S.vld(amax_ub[0]), CLAMP)
                qmax = S.vdup(QMAX, T.float32)
                S.vsts(inv_ub[0], S.vdiv(qmax, clamped))
                S.vsts(amax_ub[0], S.vdiv(clamped, qmax))
                S.mem_bar("VST_VLD")
                i0 = S.vld(inv_ub[0], dist="BRC_B32")
                i1 = S.vld(inv_ub[1], dist="BRC_B32")
                i2 = S.vld(inv_ub[2], dist="BRC_B32")
                i3 = S.vld(inv_ub[3], dist="BRC_B32")
                q0 = S.vmul(S.vld(x_ub[0]), S.vsel(i0, i1, low))
                q1 = S.vmul(S.vld(x_ub[64]), S.vsel(i2, i3, low))
                S.vsts(q_ub[0], S.vcvt(q0, T.float8_e4m3fn), dist="PK4_B32")
                S.vsts(q_ub[64], S.vcvt(q1, T.float8_e4m3fn), dist="PK4_B32")
            T.copy(q_ub, q)
            T.copy(amax_ub[0:GROUPS], sf)

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
