"""CPU-only checks of the precision contract carried by scalar Parallel '/'."""

import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def division(n):
    padded = (n + 63) // 64 * 64

    @T.prim_func
    def main(A: T.Tensor((n,), 'float32'), B: T.Tensor((n,), 'float32'), C: T.Tensor((n,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((padded,), 'float32')
            b = T.alloc_shared((padded,), 'float32')
            c = T.alloc_shared((padded,), 'float32')
            T.copy(A, a[:n])
            T.copy(B, b[:n])
            with T.SimdVF(lanes=64):
                for j in T.Parallel(n):
                    c[j] = a[j] / b[j]
            T.copy(c[:n], C)

    return main
