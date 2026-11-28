"""Compile-only feature probes: no NPU allocation or runtime calls."""

import tilelang.ascend.language as T
from tilelang.engine.lower import lower


def reduction():
    @T.prim_func
    def main(A: T.Tensor((4, 32), 'float32'), B: T.Tensor((4,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((4, 32), 'float32')
            b = T.alloc_shared((4,), 'float32')
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                T.reduce_sum(a, b, dim=1)
                for i in T.Parallel(4):
                    b[i] += 1e-20
            T.copy(b, B)

    return main


def grouped_broadcast():
    @T.prim_func
    def main(A: T.Tensor((128,), 'float32'), S: T.Tensor((4,), 'float32'), B: T.Tensor((128,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((128,), 'float32')
            s = T.alloc_shared((64,), 'float32')
            b = T.alloc_shared((128,), 'float32')
            T.copy(A, a)
            T.copy(S, s[:4])
            with T.SimdVF(lanes=64):
                for i in T.Parallel(128):
                    b[i] = a[i] * s[i // 32]
            T.copy(b, B)

    return main


def packed_scale():
    @T.prim_func
    def main(A: T.Tensor((64,), 'float32'), B: T.Tensor((64,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((64,), 'float32')
            b = T.alloc_shared((64,), 'float32')
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(64):
                    bits = T.reinterpret(a[i], T.uint32)
                    exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
                    b[i] = T.reinterpret(exponent << 23, T.float32)
            T.copy(b, B)

    return main


def fp4_cast():
    @T.prim_func
    def main(A: T.Tensor((128,), 'float32'), B: T.Tensor((128,), 'float4_e2m1fn')):
        with T.Kernel(1):
            a = T.alloc_shared((128,), 'float32')
            b = T.alloc_shared((128,), 'float4_e2m1fn')
            T.copy(A, a)
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    b[i] = T.cast(a[i], T.float4_e2m1fn)
            T.copy(b, B)

    return main


def transpose():
    @T.prim_func
    def main(A: T.Tensor((32, 32), 'float32'), B: T.Tensor((32, 32), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((32, 32), 'float32')
            b = T.alloc_shared((32, 32), 'float32')
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for i, j in T.Parallel(32, 32):
                    b[j, i] = a[i, j]
            T.copy(b, B)

    return main


def serial_loops():
    @T.prim_func
    def main(A: T.Tensor((4, 128), 'float32'), B: T.Tensor((4, 128), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((4, 128), 'float32')
            b = T.alloc_shared((4, 128), 'float32')
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(128):
                    total = T.alloc_var(T.float32, init=0.0)
                    for row in T.serial(4):
                        total += a[row, j]
                    for row in T.serial(4):
                        b[row, j] = a[row, j] * total
            T.copy(b, B)

    return main


def single_lane():
    @T.prim_func
    def main(A: T.Tensor((64,), 'float32'), B: T.Tensor((64,), 'float32')):
        with T.Kernel(1):
            a = T.alloc_shared((64,), 'float32')
            b = T.alloc_shared((64,), 'float32')
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                for i in T.Parallel(1):
                    b[i] = a[i]
            T.copy(b[:1], B[:1])

    return main
