"""ASC image-embed VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L125
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L320
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L342
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L375
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 128
D = 2
D2 = D * D
DIM = 64


@tilelang.jit(target="ascend")
def compile_copy():
    @T.prim_func
    def kernel(src: T.Tensor((HIDDEN,), T.bfloat16), dst: T.Tensor((HIDDEN,), T.bfloat16)):
        with T.Kernel(1):
            src_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            dst_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            T.copy(src, src_ub)
            with T.SimdVF():
                row = S.vld(src_ub[0])
                S.vsts(dst_ub[0], row, dist="NORM_B16")
            T.copy(dst_ub, dst)

    return kernel


@tilelang.jit(target="ascend")
def compile_pool():
    @T.prim_func
    def kernel(
        window: T.Tensor((D2, DIM), T.float32),
        out: T.Tensor((DIM,), T.float32),
        inv: T.float32,
    ):
        with T.Kernel(1):
            win_ub = T.alloc_shared((D2, DIM), T.float32)
            out_ub = T.alloc_shared((DIM,), T.float32)
            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                for i in T.serial(D2):
                    S.vsts(win_ub[i, 0], zero)
            T.copy(window, win_ub)
            with T.SimdVF():
                acc = S.alloc_var(T.float32)
                acc = S.vdup(0.0, T.float32)
                for i in T.serial(D2):
                    acc = S.vadd(acc, S.vld(win_ub[i, 0]))
                S.vsts(out_ub[0], S.vmuls(acc, inv))
            T.copy(out_ub, out)

    return kernel


@tilelang.jit(target="ascend")
def compile_scatter():
    @T.prim_func
    def kernel(
        window: T.Tensor((D2, DIM), T.float32),
        out: T.Tensor((DIM, D, D), T.float32),
    ):
        with T.Kernel(1):
            win_ub = T.alloc_shared((D2, DIM), T.float32)
            out_ub = T.alloc_shared((DIM, D, D), T.float32)
            T.copy(window, win_ub)
            with T.SimdVF():
                for i in T.serial(D2):
                    idx = S.vmuls(S.vci(0, T.int32), T.int32(D2))
                    idx = S.vadds(idx, T.int32(i))
                    S.vscatter(S.vld(win_ub[i, 0]), out_ub[0, 0, 0], T.reinterpret("uint32x64", idx))
            T.copy(out_ub, out)

    return kernel


_COPY = None
_POOL = None
_SCATTER = None


def launch_copy(src):
    global _COPY
    if _COPY is None:
        _COPY = compile_copy()
    src_n = src.npu()
    import torch

    dst = torch.empty(HIDDEN, dtype=torch.bfloat16, device=src_n.device)
    _COPY(src_n, dst)
    return dst.cpu()


def launch_pool(window, inv):
    global _POOL
    if _POOL is None:
        _POOL = compile_pool()
    window_n = window.npu()
    import torch

    out = torch.empty(DIM, dtype=torch.float32, device=window_n.device)
    _POOL(window_n, out, float(inv))
    return out.cpu()


def launch_scatter(window):
    global _SCATTER
    if _SCATTER is None:
        _SCATTER = compile_scatter()
    window_n = window.npu()
    import torch

    out = torch.empty((DIM, D, D), dtype=torch.float32, device=window_n.device)
    _SCATTER(window_n, out)
    return out.cpu()
