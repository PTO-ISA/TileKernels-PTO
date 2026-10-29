"""ASC compress VFs: weighted window, zero tile, packed decode window.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L116
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L147
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L374
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

RATIO = 2
DIM = 64


@tilelang.jit(target="ascend")
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
                zero = S.vdup(0.0, T.float32)
                one_v = S.vdup(1.0, T.float32)
                score_reg = S.alloc_local((RATIO,), T.float32)
                latent_reg = S.alloc_local((RATIO,), T.float32)
                tile_max = S.alloc_var(T.float32)
                tile_sum = S.alloc_var(T.float32)
                tile_acc = S.alloc_var(T.float32)
                for row in T.unroll(RATIO, explicit=True):
                    score_reg[row] = S.vld(score_ub[row, 0])
                    latent_reg[row] = S.vld(latent_ub[row, 0])
                tile_max = score_reg[0]
                for row in T.unroll(1, RATIO, explicit=True):
                    tile_max = S.vmax(tile_max, score_reg[row])
                tile_sum = zero
                tile_acc = zero
                for row in T.unroll(RATIO, explicit=True):
                    score_reg[row] = S.vexpdif(score_reg[row], tile_max)
                    tile_sum = S.vadd(tile_sum, score_reg[row])
                    S.vmula(tile_acc, latent_reg[row], score_reg[row])
                inv_sum = S.vdiv(one_v, tile_sum)
                result = S.vmul(tile_acc, inv_sum)
                S.vsts(result_ub[0, 0], S.vcvt(result, T.bfloat16), dist="PK_B32")
            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                S.vsts(result_ub[1, 0], S.vcvt(zero, T.bfloat16), dist="PK_B32")
            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                one_v = S.vdup(1.0, T.float32)
                score_reg = S.alloc_local((RATIO,), T.float32)
                latent_reg = S.alloc_local((RATIO,), T.float32)
                tile_max = S.alloc_var(T.float32)
                tile_sum = S.alloc_var(T.float32)
                tile_acc = S.alloc_var(T.float32)
                for row in T.unroll(RATIO, explicit=True):
                    score_reg[row] = S.vld(packed[row, 0, 0])
                    latent_reg[row] = S.vld(packed[row, 1, 0])
                tile_max = score_reg[0]
                for row in T.unroll(1, RATIO, explicit=True):
                    tile_max = S.vmax(tile_max, score_reg[row])
                tile_sum = zero
                tile_acc = zero
                for row in T.unroll(RATIO, explicit=True):
                    score_reg[row] = S.vexpdif(score_reg[row], tile_max)
                    tile_sum = S.vadd(tile_sum, score_reg[row])
                    S.vmula(tile_acc, latent_reg[row], score_reg[row])
                inv_sum = S.vdiv(one_v, tile_sum)
                result = S.vmul(tile_acc, inv_sum)
                S.vsts(result_ub[2, 0], S.vcvt(result, T.bfloat16), dist="PK_B32")
            T.copy(result_ub, out)

    return kernel


_KERNEL = None


def launch(score, latent):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    score_n = score.npu()
    out_n = torch_empty(score_n)
    _KERNEL(score_n, latent.npu(), out_n)
    return out_n.cpu()


def torch_empty(score_n):
    import torch

    return torch.empty((3, DIM), dtype=torch.bfloat16, device=score_n.device)
