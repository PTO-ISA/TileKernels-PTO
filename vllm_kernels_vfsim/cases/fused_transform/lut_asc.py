"""FP4 reciprocal LUT over one row of 512 scales.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L144
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L290
The source asserts head_dim == 512 before building this table. This driver runs that table on 512 lanes.
"""

import struct

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 512
VL = 64
_BITS = tuple(struct.unpack("<I", struct.pack("<f", 8.0 / (8 + m)))[0] for m in range(8))


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(sf: T.Tensor((N,), T.float32), out: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            src = T.alloc_shared((N,), T.float32)
            dst = T.alloc_shared((N,), T.float32)
            T.copy(sf, src)
            with T.SimdVF():
                full = S.pset(32)
                lane = T.reinterpret(S.vci(0, T.int32), "uint32x64")
                index = S.vand(lane, S.vdup(7, T.uint32, full), full)
                table = S.vdup(_BITS[0], T.uint32, full)
                table = S.vsel(S.vdup(_BITS[1], T.uint32, full), table, S.vcmps(index, 1, full, "eq"))
                table = S.vsel(S.vdup(_BITS[2], T.uint32, full), table, S.vcmps(index, 2, full, "eq"))
                table = S.vsel(S.vdup(_BITS[3], T.uint32, full), table, S.vcmps(index, 3, full, "eq"))
                table = S.vsel(S.vdup(_BITS[4], T.uint32, full), table, S.vcmps(index, 4, full, "eq"))
                table = S.vsel(S.vdup(_BITS[5], T.uint32, full), table, S.vcmps(index, 5, full, "eq"))
                table = S.vsel(S.vdup(_BITS[6], T.uint32, full), table, S.vcmps(index, 6, full, "eq"))
                table = S.vsel(S.vdup(_BITS[7], T.uint32, full), table, S.vcmps(index, 7, full, "eq"))
                for chunk in T.serial(N // VL):
                    bits = T.reinterpret(S.vld(src[chunk * VL]), "uint32x64")
                    mantissa = S.vand(S.vshrs(bits, 20, full), S.vdup(7, T.uint32, full), full)
                    normalized = S.vselr(table, mantissa)
                    exponent = S.vand(S.vshrs(bits, 23, full), S.vdup(255, T.uint32, full), full)
                    inverse_bits = S.vadd(normalized, S.vshls(S.vsub(S.vdup(127, T.uint32, full), exponent, full), 23, full), full)
                    keep_bits = S.vcmps(exponent, 255, full, "eq")
                    inverse = T.reinterpret(S.vsel(bits, inverse_bits, keep_bits), "float32x64")
                    S.vsts(dst[chunk * VL], inverse, full)
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
