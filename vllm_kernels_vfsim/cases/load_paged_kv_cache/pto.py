"""PTO paged-KV dequant. The two scale shifts stay a vector ``vshl``."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.load_paged_kv_cache.asc import N


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        raw: T.Tensor((N,), T.uint8),
        packed_scale: T.int32,
        out: T.Tensor((2, N), T.bfloat16),
    ):
        with T.Kernel(1):
            raw_ub = T.alloc_shared((N,), T.uint8)
            out_ub = T.alloc_shared((2, N), T.bfloat16)
            T.copy(raw, raw_ub)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                low = V.create_mask(32, size=N)
                shift = V.vsel(
                    low,
                    V.vbrc(T.int32(23), size=N),
                    V.vbrc(T.int32(15), size=N),
                )
                shift_u = V.vinterpret_cast(shift, "uint32")
                word = V.vinterpret_cast(V.vbrc(packed_scale, size=N), "uint32")
                bits = V.vand(V.vshl(word, shift_u, mask), V.vbrc(T.uint32(255 << 23), size=N), mask)
                scale = V.vinterpret_cast(bits, "float32")
                fp8 = V.vinterpret_cast(V.vload(raw_ub[0], size=N), "float8_e4m3fn")
                values = V.vcvt(fp8, "float32")
                V.vstore(V.vcvt(V.vmul(values, scale, mask), "bfloat16", rounding="R"), out_ub[0, 0])
            with T.SimdVF():
                zero = V.vcvt(V.vbrc(T.float32(0.0), size=N), "bfloat16", rounding="R")
                V.vstore(zero, out_ub[1, 0])
            T.copy(out_ub, out)

    return kernel


_KERNEL = None


def launch(raw, packed_scale):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    raw_n = raw.npu()
    out = torch.empty((2, N), dtype=torch.bfloat16, device=raw_n.device)
    _KERNEL(raw_n, int(packed_scale), out)
    return out.cpu()
