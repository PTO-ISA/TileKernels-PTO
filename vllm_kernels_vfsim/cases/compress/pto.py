"""PTO compress VFs. ``vexpdif`` stays; the zero tile is one convert and store.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L116
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

RATIO = 2
DIM = 64


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        score: T.Tensor((RATIO, DIM), T.float32),
        latent: T.Tensor((RATIO, DIM), T.float32),
        out: T.Tensor((3, DIM), T.bfloat16),
    ):
        with T.Kernel(1):
            score_ub = T.alloc_shared((RATIO, DIM), T.float32)
            latent_ub = T.alloc_shared((RATIO, DIM), T.float32)
            packed = T.alloc_shared((RATIO, 2, DIM), T.float32)
            result_ub = T.alloc_shared((3, DIM), T.bfloat16)
            T.copy(score, score_ub)
            T.copy(latent, latent_ub)
            T.copy(score, packed[0:RATIO, 0, 0:DIM])
            T.copy(latent, packed[0:RATIO, 1, 0:DIM])
            with T.SimdVF():
                mask = V.create_mask(DIM, size=DIM)
                one_v = V.vbrc(T.float32(1.0), size=DIM)
                s0 = V.vload(score_ub[0, 0], size=DIM)
                s1 = V.vload(score_ub[1, 0], size=DIM)
                l0 = V.vload(latent_ub[0, 0], size=DIM)
                l1 = V.vload(latent_ub[1, 0], size=DIM)
                tile_max = V.vmax(s0, s1, mask)
                w0 = V.vexpdif(s0, tile_max, mask)
                w1 = V.vexpdif(s1, tile_max, mask)
                tile_sum = V.vadd(w0, w1, mask)
                acc = V.vmula(V.vmul(l0, w0, mask), l1, w1, mask)
                result = V.vmul(acc, V.vdiv(one_v, tile_sum, mask), mask)
                V.vstore(V.vcvt(result, "bfloat16", rounding="R"), result_ub[0, 0])
            with T.SimdVF():
                zero = V.vbrc(T.float32(0.0), size=DIM)
                V.vstore(V.vcvt(zero, "bfloat16", rounding="R"), result_ub[1, 0])
            with T.SimdVF():
                mask = V.create_mask(DIM, size=DIM)
                one_v = V.vbrc(T.float32(1.0), size=DIM)
                s0 = V.vload(packed[0, 0, 0], size=DIM)
                s1 = V.vload(packed[1, 0, 0], size=DIM)
                l0 = V.vload(packed[0, 1, 0], size=DIM)
                l1 = V.vload(packed[1, 1, 0], size=DIM)
                tile_max = V.vmax(s0, s1, mask)
                w0 = V.vexpdif(s0, tile_max, mask)
                w1 = V.vexpdif(s1, tile_max, mask)
                tile_sum = V.vadd(w0, w1, mask)
                acc = V.vmula(V.vmul(l0, w0, mask), l1, w1, mask)
                result = V.vmul(acc, V.vdiv(one_v, tile_sum, mask), mask)
                V.vstore(V.vcvt(result, "bfloat16", rounding="R"), result_ub[2, 0])
            T.copy(result_ub, out)

    return kernel


_KERNEL = None


def launch(score, latent):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    score_n = score.npu()
    out_n = torch.empty((3, DIM), dtype=torch.bfloat16, device=score_n.device)
    _KERNEL(score_n, latent.npu(), out_n)
    return out_n.cpu()
