"""PTO reciprocal LUT. The 8-entry table is one unrolled select, applied to 512 lanes."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_transform.lut_asc import N, VL, _BITS


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(sf: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            src = T.alloc_shared((N,), T.float32)
            dst = T.alloc_shared((N,), T.float32)
            T.copy(sf, src)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                lane = V.vinterpret_cast(V.vci(T.int32(0), size=VL), "uint32")
                index = V.vand(lane, V.vbrc(T.uint32(7), size=VL))
                table = V.alloc_var(T.uint32, size=VL)
                table = V.vbrc(T.uint32(_BITS[0]), size=VL)
                table = V.vsel(V.vcmps(index, T.uint32(1), mask, "eq"), V.vbrc(T.uint32(_BITS[1]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(2), mask, "eq"), V.vbrc(T.uint32(_BITS[2]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(3), mask, "eq"), V.vbrc(T.uint32(_BITS[3]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(4), mask, "eq"), V.vbrc(T.uint32(_BITS[4]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(5), mask, "eq"), V.vbrc(T.uint32(_BITS[5]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(6), mask, "eq"), V.vbrc(T.uint32(_BITS[6]), size=VL), table)
                table = V.vsel(V.vcmps(index, T.uint32(7), mask, "eq"), V.vbrc(T.uint32(_BITS[7]), size=VL), table)
                for chunk in T.serial(N // VL):
                    bits = V.vinterpret_cast(V.vload(src[chunk * VL], size=VL), "uint32")
                    mantissa = V.vand(V.vshrs(bits, 20, mask), V.vbrc(T.uint32(7), size=VL))
                    normalized = V.vselr(table, mantissa)
                    exponent = V.vand(V.vshrs(bits, 23, mask), V.vbrc(T.uint32(255), size=VL))
                    delta = V.vshls(V.vsub(V.vbrc(T.uint32(127), size=VL), exponent, mask), 23, mask)
                    inverse_bits = V.vadd(normalized, delta, mask)
                    keep_bits = V.vcmps(exponent, T.uint32(255), mask, "eq")
                    inverse = V.vinterpret_cast(V.vsel(keep_bits, bits, inverse_bits), "float32")
                    V.vstore(inverse, dst[chunk * VL], mask)
            T.copy(dst, out)

    return kernel


_KERNEL = None


def launch(sf):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = sf.npu()
    out = torch.empty_like(src)
    _KERNEL(src, out)
    return out.cpu()
