"""PTO image-embed VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

HIDDEN = 128
D = 2
D2 = D * D
DIM = 64


@tilelang.jit(target="pto")
def compile_copy():
    @T.prim_func
    def kernel(src: T.Tensor((HIDDEN,), T.bfloat16), dst: T.Tensor((HIDDEN,), T.bfloat16)):
        with T.Kernel(1):
            src_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            dst_ub = T.alloc_shared((HIDDEN,), T.bfloat16)
            T.copy(src, src_ub)
            with T.SimdVF():
                V.vstore(V.vload(src_ub[0], size=HIDDEN), dst_ub[0])
            T.copy(dst_ub, dst)

    return kernel


@tilelang.jit(target="pto")
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
                zero = V.vbrc(T.float32(0.0), size=DIM)
                for i in T.serial(D2):
                    V.vstore(zero, win_ub[i, 0])
            T.copy(window, win_ub)
            with T.SimdVF():
                mask = V.create_mask(DIM, size=DIM)
                acc = V.alloc_var(T.float32, size=DIM)
                acc = V.vbrc(T.float32(0.0), size=DIM)
                for i in T.serial(D2):
                    acc = V.vadd(acc, V.vload(win_ub[i, 0], size=DIM), mask)
                V.vstore(V.vmuls(acc, inv, mask), out_ub[0], mask)
            T.copy(out_ub, out)

    return kernel


@tilelang.jit(target="pto")
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
                mask = V.create_mask(DIM, size=DIM)
                lane = V.vci(T.int32(0), size=DIM)
                for i in T.serial(D2):
                    idx = V.vadds(V.vmuls(lane, T.int32(D2), mask), T.int32(i), mask)
                    V.vscatter(
                        V.vload(win_ub[i, 0], size=DIM),
                        out_ub[0, 0, 0],
                        V.vinterpret_cast(idx, "uint32"),
                        mask,
                    )
            T.copy(out_ub, out)

    return kernel


_COPY = None
_POOL = None
_SCATTER = None


def launch_copy(src):
    global _COPY
    if _COPY is None:
        _COPY = compile_copy()
    import torch

    src_n = src.npu()
    dst = torch.empty(HIDDEN, dtype=torch.bfloat16, device=src_n.device)
    _COPY(src_n, dst)
    return dst.cpu()


def launch_pool(window, inv):
    global _POOL
    if _POOL is None:
        _POOL = compile_pool()
    import torch

    window_n = window.npu()
    out = torch.empty(DIM, dtype=torch.float32, device=window_n.device)
    _POOL(window_n, out, float(inv))
    return out.cpu()


def launch_scatter(window):
    global _SCATTER
    if _SCATTER is None:
        _SCATTER = compile_scatter()
    import torch

    window_n = window.npu()
    out = torch.empty((DIM, D, D), dtype=torch.float32, device=window_n.device)
    _SCATTER(window_n, out)
    return out.cpu()
