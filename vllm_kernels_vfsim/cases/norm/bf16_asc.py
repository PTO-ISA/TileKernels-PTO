"""bf16 group-amax, the vld2 / vcgmax path.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L235
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 256
GROUPS = 8


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.bfloat16),
        amax: T.Tensor((GROUPS,), T.float32),
        q: T.Tensor((HIDDEN,), T.float8_e4m3fn),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            out = T.alloc_shared((64,), T.float32)
            inv_ub = T.alloc_shared((64,), T.float32)
            q_ub = T.alloc_shared((HIDDEN,), T.float8_e4m3fn)
            T.copy(x, ub)
            with T.SimdVF():
                mask_vl8 = S.pset(32, "PAT_VL8")
                abs_mask = S.vdup(0x7FFF, T.uint16)
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                x0, x1 = S.vld2(ub[0], dist="DINTLV_B16")
                a0 = S.vand(T.reinterpret(x0, "uint16x128"), abs_mask)
                a1 = S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)
                maxima = S.vcgmax(S.vmax(a0, a1))
                dense, _ = S.vintlv(zero_bf16, T.reinterpret(maxima, "bfloat16x128"))
                S.vsts(out[0], T.reinterpret(dense, "float32x64"), mask_vl8, dist="NORM_B32", extent=GROUPS)
                S.mem_bar("VST_VLD")
                low = S.pset(32, "PAT_VL32")
                one = S.pset(32, "PAT_VL1")
                qmax = S.vdup(448.0, T.float32)
                for g in T.unroll(GROUPS, explicit=True):
                    scale = S.vmaxs(S.vld(out[g], dist="BRC_B32"), 1e-4)
                    S.vsts(inv_ub[g], S.vdiv(qmax, scale), one, "ONEPT_B32")
                S.mem_bar("VST_VLD")
                for pair in T.unroll(HIDDEN // 64, explicit=True):
                    value = S.vcvt(S.vld(ub[pair * 64], dist="UNPK_B16"), T.float32)
                    inv0 = S.vld(inv_ub[pair * 2], dist="BRC_B32")
                    inv1 = S.vld(inv_ub[pair * 2 + 1], dist="BRC_B32")
                    mixed = S.vsel(inv0, inv1, low)
                    S.vsts(q_ub[pair * 64], S.vcvt(S.vmul(value, mixed), T.float8_e4m3fn), dist="PK4_B32")
            T.copy(out[0:GROUPS], amax)
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
