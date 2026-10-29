"""Non-power-of-two remainder. ASC uses integer vdiv. PTO must use float vdiv.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L52
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

VL = 64
BLOCK = 6


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(out: T.Tensor((VL,), T.int32), offset: T.int32, length: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((VL,), T.int32)
            with T.SimdVF():
                full = S.pset(32)
                pos = S.vadds(S.vci(0, T.int32), offset, full)
                bs = S.vdup(BLOCK, T.int32, full)
                quot = S.vdiv(pos, bs, full)
                rem = S.vsub(pos, S.vmul(quot, bs, full), full)
                inside = S.vcmps(S.vci(0, T.int32), length, full, "lt")
                S.vsts(ub[0], S.vsel(rem, S.vdup(-1, T.int32, full), inside), full)
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
