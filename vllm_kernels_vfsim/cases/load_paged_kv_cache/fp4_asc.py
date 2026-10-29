"""FP4 dequant arm of load_paged_kv_cache. Kept even if the compiler rejects it.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py#L258
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 128


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        raw: T.Tensor((N // 2,), T.uint8),
        scale: T.Tensor((N,), T.bfloat16),
        out: T.Tensor((N,), T.bfloat16),
    ):
        with T.Kernel(1):
            raw_ub = T.alloc_shared((N // 2,), T.uint8)
            scale_ub = T.alloc_shared((N,), T.bfloat16)
            out_ub = T.alloc_shared((N,), T.bfloat16)
            T.copy(raw, raw_ub)
            T.copy(scale, scale_ub)
            with T.SimdVF():
                mask = S.pset(16)
                packed = S.vld(raw_ub[0], dist="UNPK4_B8")
                values = S.vcvt(T.reinterpret(packed, "float4_e2m1fnx512"), T.bfloat16, part=0)
                scales = S.vld(scale_ub[0])
                S.vsts(out_ub[0], S.vmul(values, scales, mask), mask, dist="NORM_B16")
            T.copy(out_ub, out)

    return kernel


_KERNEL = None


def launch(raw, scale):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    src = raw.npu()
    out = torch.empty(N, dtype=torch.bfloat16, device=src.device)
    _KERNEL(src, scale.npu(), out)
    return out.cpu()
