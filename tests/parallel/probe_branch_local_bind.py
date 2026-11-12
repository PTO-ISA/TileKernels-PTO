"""CPU-only reproducer for scalar temporaries in a predicated Parallel body."""

from tilelang.ascend import language as T
from tilelang.engine.lower import lower


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1):
            a = T.alloc_shared((64,), T.float32)
            b = T.alloc_shared((64,), T.float32)
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(64):
                    if j < 32:
                        value = T.exp(-a[j])
                        b[j] = 1.0 / (1.0 + value)
            T.copy(b[:32], B[:32])

    return main
