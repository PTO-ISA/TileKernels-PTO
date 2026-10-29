"""PTO log-sum-exp and argmax. Reductions stay 1-lane and are broadcast for the compare."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_sample.asc import N, VL


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        logits: T.Tensor((N,), T.float32),
        lse: T.Tensor((1,), T.float32),
        argmax: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            lse_ub = T.alloc_shared((1,), T.float32)
            idx_ub = T.alloc_shared((1,), T.int32)
            T.copy(logits, ub)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                big = V.vbrc(T.int32(1 << 30), size=VL)
                c0 = V.vload(ub[0], size=VL)
                c1 = V.vload(ub[VL], size=VL)
                m0 = V.vcmax(c0, mask)
                m1 = V.vcmax(c1, mask)
                m = V.vmax(m0, m1, V.create_mask(1, size=1))
                m_b = V.vbrc(m, size=VL)
                s0 = V.vcadd(V.vexp(V.vsub(c0, m_b, mask), mask), mask, reassoc=True)
                s1 = V.vcadd(V.vexp(V.vsub(c1, m_b, mask), mask), mask, reassoc=True)
                total = V.vadd(s0, s1, V.create_mask(1, size=1))
                lse_v = V.vadd(m, V.vln(total, V.create_mask(1, size=1)), V.create_mask(1, size=1))
                V.vstore(lse_v, lse_ub[0])
                i0 = V.vci(T.int32(0), size=VL)
                i1 = V.vci(T.int32(VL), size=VL)
                m0b = V.vbrc(m0, size=VL)
                m1b = V.vbrc(m1, size=VL)
                best0 = V.vcmin(V.vsel(V.vcmp(c0, m0b, mask, "eq"), i0, big), mask)
                best1 = V.vcmin(V.vsel(V.vcmp(c1, m1b, mask, "eq"), i1, big), mask)
                take0 = V.vcmp(m0, m, V.create_mask(1, size=1), "gt")
                tie = V.vand(
                    V.vcmp(m0, m, V.create_mask(1, size=1), "eq"),
                    V.vcmp(best0, best1, V.create_mask(1, size=1), "le"),
                )
                pick0 = V.vor(take0, tie)
                chosen = V.vsel(pick0, best0, best1)
                V.vstore(chosen, idx_ub[0])
            T.copy(lse_ub, lse)
            T.copy(idx_ub, argmax)

    return kernel


_KERNEL = None


def launch(logits):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = logits.npu()
    lse = torch.empty(1, device=src.device)
    argmax = torch.empty(1, dtype=torch.int32, device=src.device)
    _KERNEL(src, lse, argmax)
    return lse.cpu(), argmax.cpu()
