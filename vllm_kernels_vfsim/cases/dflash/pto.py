"""PTO dflash VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L57
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L137

``UNPK_B16`` plus ``vcvt(..., part=0)`` is one ``vcvt`` of a 64-lane bfloat16 load.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

ROWS = 4
DIM = 128
HEADS = 2
VEC = 64


@tilelang.jit(target="pto")
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
                mask = V.create_mask(VEC, size=VEC)
                zero = V.vbrc(T.float32(0.0), size=VEC)
                acc0 = V.alloc_var(T.float32, size=VEC)
                acc1 = V.alloc_var(T.float32, size=VEC)
                acc2 = V.alloc_var(T.float32, size=VEC)
                acc3 = V.alloc_var(T.float32, size=VEC)
                acc0 = zero
                acc1 = zero
                acc2 = zero
                acc3 = zero
                for i in T.serial(DIM // VEC):
                    off = i * VEC
                    w_v = V.vcvt(V.vload(w_ub[off], size=VEC), "float32")
                    acc0 = V.vmula(acc0, w_v, V.vcvt(V.vload(x_ub[0, off], size=VEC), "float32"), mask)
                    acc1 = V.vmula(acc1, w_v, V.vcvt(V.vload(x_ub[1, off], size=VEC), "float32"), mask)
                    acc2 = V.vmula(acc2, w_v, V.vcvt(V.vload(x_ub[2, off], size=VEC), "float32"), mask)
                    acc3 = V.vmula(acc3, w_v, V.vcvt(V.vload(x_ub[3, off], size=VEC), "float32"), mask)
                V.vstore(V.vcadd(acc0, mask, reassoc=True), sum_ub[0])
                V.vstore(V.vcadd(acc1, mask, reassoc=True), sum_ub[1])
                V.vstore(V.vcadd(acc2, mask, reassoc=True), sum_ub[2])
                V.vstore(V.vcadd(acc3, mask, reassoc=True), sum_ub[3])
            T.copy(sum_ub, out)

    return kernel


@tilelang.jit(target="pto")
def compile_mean():
    @T.prim_func
    def kernel(
        x: T.Tensor((HEADS, DIM), T.bfloat16),
        out: T.Tensor((DIM,), T.bfloat16),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HEADS, DIM), T.bfloat16)
            y_ub = T.alloc_shared((DIM,), T.bfloat16)
            T.copy(x, x_ub)
            with T.SimdVF():
                mask = V.create_mask(DIM, size=DIM)
                a = V.vcvt(V.vload(x_ub[0, 0], size=DIM), "float32")
                b = V.vcvt(V.vload(x_ub[1, 0], size=DIM), "float32")
                mean = V.vmuls(V.vadd(a, b, mask), T.float32(1.0 / HEADS), mask)
                V.vstore(V.vcvt(mean, "bfloat16", rounding="R"), y_ub[0])
            T.copy(y_ub, out)

    return kernel


_DOT = None
_MEAN = None


def launch_dot(x, weight):
    global _DOT
    if _DOT is None:
        _DOT = compile_dot()
    import torch

    x_n = x.npu()
    out = torch.empty(ROWS, dtype=torch.float32, device=x_n.device)
    _DOT(x_n, weight.npu(), out)
    return out.cpu()


def launch_mean(x):
    global _MEAN
    if _MEAN is None:
        _MEAN = compile_mean()
    import torch

    x_n = x.npu()
    out = torch.empty(DIM, dtype=torch.bfloat16, device=x_n.device)
    _MEAN(x_n, out)
    return out.cpu()
