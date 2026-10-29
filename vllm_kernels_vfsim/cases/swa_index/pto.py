"""PTO expand_swa_index VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/swa_index.py#L41

Each ``BRC_B32`` load is ``vload`` of one lane plus ``vbrc``.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

ROWS = 2
WIDTH = 128
VL = 64


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        pivots: T.Tensor((ROWS, 4), T.int32),
        output: T.Tensor((ROWS, WIDTH), T.int32),
    ):
        with T.Kernel(1):
            pivots_ub = T.alloc_shared((ROWS, 4), T.int32)
            output_ub = T.alloc_shared((ROWS, WIDTH), T.int32)
            T.copy(pivots, pivots_ub)
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                neg1 = V.vbrc(T.int32(-1), size=VL)
                for row in T.serial(ROWS):
                    offset1 = V.vbrc(V.vload(pivots_ub[row, 0], size=1), size=VL)
                    len1 = V.vbrc(V.vload(pivots_ub[row, 1], size=1), size=VL)
                    offset2 = V.vbrc(V.vload(pivots_ub[row, 2], size=1), size=VL)
                    len2 = V.vbrc(V.vload(pivots_ub[row, 3], size=1), size=VL)
                    end2 = V.vadd(len1, len2, full)
                    base2 = V.vsub(offset2, len1, full)
                    for chunk in T.serial(WIDTH // VL):
                        col = V.vci(chunk * VL, size=VL)
                        first = V.vcmp(col, len1, full, "lt")
                        inside = V.vcmp(col, end2, full, "lt")
                        value1 = V.vadd(offset1, col, full)
                        value2 = V.vadd(base2, col, full)
                        value = V.vsel(first, value1, value2)
                        V.vstore(V.vsel(inside, value, neg1), output_ub[row, chunk * VL], full)
            T.copy(output_ub, output)

    return kernel


_KERNEL = None


def launch(pivots):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    pivots_n = pivots.npu()
    out_n = pivots_n.new_empty((ROWS, WIDTH))
    _KERNEL(pivots_n, out_n)
    return out_n.cpu()
