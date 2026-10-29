"""PTO gather zero-fill. One broadcast store per row replaces the scalar parallel fill.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/gather.py#L56
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

N = 4
M = 4
D = 128
PAD = 256


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        x: T.Tensor((N, D), T.int8),
        index: T.Tensor((M,), T.int32),
        y: T.Tensor((M, D), T.int8),
    ):
        with T.Kernel(1):
            stage = T.alloc_shared((M, PAD), T.int8)
            with T.SimdVF():
                zero = V.vbrc(T.int8(0), size=PAD)
                for row in T.serial(M):
                    V.vstore(zero, stage[row, 0])
            for b in T.serial(M):
                idx = T.alloc_var(T.int32, index[b])
                if idx >= 0 and idx < N:
                    T.copy(x[idx, 0:D], stage[b, 0:D])
            T.copy(stage[0:M, 0:D], y)

    return kernel


_KERNEL = None


def launch(x, index):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    x_n = x.npu()
    y_n = x_n.new_empty((M, D))
    _KERNEL(x_n, index.npu(), y_n)
    return y_n.cpu()
