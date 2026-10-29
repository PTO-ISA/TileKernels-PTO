"""ASC paged-KV dequant VF and the missing-page zero VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py#L258
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py#L326

One vector of 64 FP8 values. Two UE8M0 scale bytes, low 32 lanes and high 32 lanes.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 64


@tilelang.jit(target="ascend")
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
                mask = S.pset(32)
                low = S.pset(32, "PAT_VL32")
                lane = S.vci(0, "int32")
                low_half = S.vcmps(lane, 32, mask, "lt")
                shift = S.vsel(S.vdup(23, T.int32, mask), S.vdup(15, T.int32, mask), low_half)
                word = S.vdup(packed_scale, T.int32, mask)
                shifted = S.vshl(word, shift, mask)
                bits = S.vand(shifted, S.vdup(255 << 23, T.int32, mask), mask)
                scale = T.reinterpret("float32x64", bits)
                fp8_u8 = S.vld(raw_ub[0], dist="UNPK4_B8")
                values = S.vcvt(T.reinterpret("float8_e4m3fnx256", fp8_u8), T.float32, part=0)
                result = S.vmul(values, scale, mask)
                S.vsts(out_ub[0, 0], S.vcvt(result, T.bfloat16), mask, "PK_B32")
            with T.SimdVF():
                mask = S.pset(32)
                zero = S.vcvt(S.vdup(0.0, T.float32, mask), T.bfloat16)
                S.vsts(out_ub[1, 0], zero, mask, "PK_B32")
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
