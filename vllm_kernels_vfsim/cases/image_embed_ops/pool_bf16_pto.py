"""PTO bf16 window mean: sum in fp32, one convert back. No even/odd pack."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.image_embed_ops.pool_bf16_asc import D2, DIM


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(window: T.Tensor((D2, DIM), T.bfloat16), out: T.Tensor((DIM,), T.bfloat16), inv: T.float32):
        with T.Kernel(1):
            win = T.alloc_shared((D2, DIM), T.bfloat16)
            dst = T.alloc_shared((DIM,), T.bfloat16)
            T.copy(window, win)
            with T.SimdVF():
                mask = V.create_mask(DIM, size=DIM)
                acc = V.alloc_var(T.float32, size=DIM)
                acc = V.vbrc(T.float32(0.0), size=DIM)
                for i in T.serial(D2):
                    acc = V.vadd(acc, V.vcvt(V.vload(win[i, 0], size=DIM), "float32"), mask)
                mean = V.vmuls(acc, inv, mask)
                V.vstore(V.vcvt(mean, "bfloat16", rounding="R"), dst[0])
            T.copy(dst, out)

    return kernel


_KERNEL = None


def launch(window, inv):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = window.npu()
    out = torch.empty(DIM, dtype=torch.bfloat16, device=src.device)
    _KERNEL(src, out, float(inv))
    return out.cpu()
