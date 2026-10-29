"""PTO single-core driver for kv_index_cat.

Source VF: https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/kv_index_cat.py#L60

The ASC broadcast-add / compare / select is the same three operations. Masks
and ``size`` are explicit, which is the VMI calling convention.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

ROWS = 2
COLS = 128
VL = 64


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        src: T.Tensor((ROWS, COLS), T.int32),
        dst: T.Tensor((ROWS, COLS), T.int32),
        offset: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((ROWS, COLS), T.int32)
            T.copy(src, ub)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                for r in T.serial(ROWS):
                    for v in T.serial(COLS // VL):
                        val = V.vload(ub[r, v * VL], size=VL)
                        added = V.vadds(val, offset, mask)
                        keep = V.vcmps(val, T.int32(-1), mask, "eq")
                        V.vstore(V.vsel(keep, val, added), ub[r, v * VL], mask)
            T.copy(ub, dst)

    return kernel


_KERNEL = None


def launch(src, offset):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    src_n = src.npu()
    dst_n = src_n.new_empty(src_n.shape)
    _KERNEL(src_n, dst_n, int(offset))
    return dst_n.cpu()
