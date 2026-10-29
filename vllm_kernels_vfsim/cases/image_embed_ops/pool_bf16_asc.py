"""bf16 mean of a D*D window. ASC packs even/odd halves; PTO converts once.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L342
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

D2 = 4
DIM = 128


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(window: T.Tensor((D2, DIM), T.bfloat16), out: T.Tensor((DIM,), T.bfloat16), inv: T.float32):
        with T.Kernel(1):
            win = T.alloc_shared((D2, DIM), T.bfloat16)
            dst = T.alloc_shared((DIM,), T.bfloat16)
            acc0 = T.alloc_shared((64,), T.float32)
            acc1 = T.alloc_shared((64,), T.float32)
            T.copy(window, win)
            with T.SimdVF():
                full = S.pset(32)
                a0 = S.alloc_var(T.float32)
                a1 = S.alloc_var(T.float32)
                a0 = S.vdup(0.0, T.float32)
                a1 = S.vdup(0.0, T.float32)
                for i in T.serial(D2):
                    packed = S.vld(win[i, 0])
                    a0 = S.vadd(a0, S.vcvt(packed, T.float32, part=0), full)
                    a1 = S.vadd(a1, S.vcvt(packed, T.float32, part=1), full)
                a0 = S.vmuls(a0, inv, full)
                a1 = S.vmuls(a1, inv, full)
                packed = S.vor(
                    T.reinterpret(S.vcvt(a0, T.bfloat16, part=0), "uint16x128"),
                    T.reinterpret(S.vcvt(a1, T.bfloat16, part=1), "uint16x128"),
                )
                S.vsts(dst[0], T.reinterpret(packed, "bfloat16x128"), dist="NORM_B16")
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
