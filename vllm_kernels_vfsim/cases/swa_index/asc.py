"""ASC expand_swa_index VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/swa_index.py#L41
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

ROWS = 2
WIDTH = 128
VL = 64


@tilelang.jit(target="ascend")
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
                full = S.pset(32)
                neg1 = S.vdup(-1, T.int32, full)
                for row in T.serial(ROWS):
                    offset1 = S.vld(pivots_ub[row, 0], dist="BRC_B32")
                    len1 = S.vld(pivots_ub[row, 1], dist="BRC_B32")
                    offset2 = S.vld(pivots_ub[row, 2], dist="BRC_B32")
                    len2 = S.vld(pivots_ub[row, 3], dist="BRC_B32")
                    end2 = S.vadd(len1, len2, full)
                    base2 = S.vsub(offset2, len1, full)
                    for chunk in T.serial(WIDTH // VL):
                        col = S.vci(chunk * VL, T.int32)
                        first = S.vcmp(col, len1, full, "lt")
                        inside = S.vcmp(col, end2, full, "lt")
                        value1 = S.vadd(offset1, col, full)
                        value2 = S.vadd(base2, col, full)
                        value = S.vsel(value1, value2, first)
                        value = S.vsel(value, neg1, inside)
                        S.vsts(output_ub[row, chunk * VL], value, full)
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
