"""PTO FP4 dequant: vcvt from packed float4, then multiply by the scale."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.load_paged_kv_cache.fp4_asc import N


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        raw: T.Tensor((N // 2,), T.uint8),
        scale: T.Tensor((N,), T.bfloat16),
        out: T.Tensor((N,), T.bfloat16),
    ):
        with T.Kernel(1):
            raw_ub = T.alloc_shared((N // 2,), T.uint8)
            scale_ub = T.alloc_shared((N,), T.bfloat16)
            out_ub = T.alloc_shared((N,), T.bfloat16)
            T.copy(raw, raw_ub)
            T.copy(scale, scale_ub)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                packed = V.vinterpret_cast(V.vload(raw_ub[0], size=N // 2), "float4_e2m1fn")
                values = V.vcvt(packed, "bfloat16")
                scales = V.vload(scale_ub[0], size=N)
                V.vstore(V.vmul(values, scales, mask), out_ub[0], mask)
            T.copy(out_ub, out)

    return kernel


_KERNEL = None


def launch(raw, scale):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = raw.npu()
    out = torch.empty(N, dtype=torch.bfloat16, device=src.device)
    _KERNEL(src, scale.npu(), out)
    return out.cpu()
