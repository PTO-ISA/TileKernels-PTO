"""PTO remainder via float divide. pto.vmi.vdiv does not accept integers."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.get_kv_index.mod_asc import BLOCK, VL


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(out: T.Tensor((VL,), T.int32), offset: T.int32, length: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((VL,), T.int32)
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                pos = V.vadds(V.vci(T.int32(0), size=VL), offset, full)
                pos_f = V.vcvt(pos, "float32")
                quot_f = V.vdiv(pos_f, V.vbrc(T.float32(BLOCK), size=VL), full)
                quot = V.vcvt(quot_f, "int32", rounding="Z")
                rem = V.vsub(pos, V.vmul(quot, V.vbrc(T.int32(BLOCK), size=VL), full), full)
                inside = V.vcmps(V.vci(T.int32(0), size=VL), length, full, "lt")
                V.vstore(V.vsel(inside, rem, V.vbrc(T.int32(-1), size=VL)), ub[0], full)
            T.copy(ub, out)

    return kernel


_KERNEL = None


def launch(offset, length):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    dummy = torch.empty(1, dtype=torch.int32).npu()
    out = torch.empty(VL, dtype=torch.int32, device=dummy.device)
    _KERNEL(out, int(offset), int(length))
    return out.cpu()
