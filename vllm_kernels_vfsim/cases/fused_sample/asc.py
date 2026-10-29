"""ASC fused-sample pieces that are not already the sampler CDF or the bitmask.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py

This driver runs the log-sum-exp merge (init, chunk max/sum, lse) and the
chunk argmax. The quartile search, keep-mask, and fixed-point CDF are the same
vector patterns as `sum_smaller_probs`, `structure_output`, and `random_sampling`.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 128
VL = 64


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        logits: T.Tensor((N,), T.float32),
        lse: T.Tensor((1,), T.float32),
        argmax: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            lse_ub = T.alloc_shared((8,), T.float32)
            idx_ub = T.alloc_shared((8,), T.int32)
            T.copy(logits, ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                neg = S.vdup(float("-inf"), T.float32, full)
                big = S.vdup(T.int32(1 << 30), T.int32, full)
                c0 = S.vld(ub[0])
                c1 = S.vld(ub[VL])
                m0 = S.vdupv(S.vcmax(c0, full), full, "POS_LOWEST")
                m1 = S.vdupv(S.vcmax(c1, full), full, "POS_LOWEST")
                m = S.vmax(m0, m1, full)
                s0 = S.vcadd(S.vexp(S.vsub(c0, m, full), full), full)
                s1 = S.vcadd(S.vexp(S.vsub(c1, m, full), full), full)
                total = S.vadd(s0, s1, full)
                lse_v = S.vadd(m, S.vln(total, full), full)
                S.vsts(lse_ub[0], lse_v, one, "ONEPT_B32")
                i0 = S.vci(0, "int32")
                i1 = S.vci(VL, "int32")
                match0 = S.vcmp(c0, m0, full, "eq")
                match1 = S.vcmp(c1, m1, full, "eq")
                best0 = S.vcmin(S.vsel(i0, big, match0), full)
                best1 = S.vcmin(S.vsel(i1, big, match1), full)
                take0 = S.vcmp(m0, m, full, "gt")
                tie = S.pand(S.vcmp(m0, m, full, "eq"), S.vcmp(best0, best1, full, "le"), full)
                pick0 = S.por(take0, tie, full)
                chosen = S.vsel(best0, best1, pick0)
                S.vsts(idx_ub[0], chosen, one, "ONEPT_B32")
            T.copy(lse_ub[0:1], lse)
            T.copy(idx_ub[0:1], argmax)

    return kernel


_KERNEL = None


def launch(logits):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    src = logits.npu()
    import torch

    lse = torch.empty(1, device=src.device)
    argmax = torch.empty(1, dtype=torch.int32, device=src.device)
    _KERNEL(src, lse, argmax)
    return lse.cpu(), argmax.cpu()
