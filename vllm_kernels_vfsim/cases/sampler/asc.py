"""ASC sampler VFs: key select, accept/reject copy, and a small scatter/gather.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L181
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L247
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L633

The production remap walks a 43136-wide pool. This driver uses 64 lanes so the
scatter and gather still run, on one core.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 64


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        keys: T.Tensor((N,), T.int32),
        values: T.Tensor((N,), T.float32),
        query: T.int32,
        src: T.Tensor((N,), T.float32),
        accepted: T.int32,
        indices: T.Tensor((N,), T.int32),
        selected: T.Tensor((1,), T.float32),
        finalized: T.Tensor((N,), T.float32),
        gathered: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(1):
            keys_ub = T.alloc_shared((N,), T.int32)
            values_ub = T.alloc_shared((N,), T.float32)
            src_ub = T.alloc_shared((N,), T.float32)
            idx_ub = T.alloc_shared((N,), T.int32)
            pool_ub = T.alloc_shared((N,), T.float32)
            out_ub = T.alloc_shared((N,), T.float32)
            sel_ub = T.alloc_shared((8,), T.float32)
            T.copy(keys, keys_ub)
            T.copy(values, values_ub)
            T.copy(src, src_ub)
            T.copy(indices, idx_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                neg = S.vdup(float("-inf"), T.float32, full)
                pred = S.vcmps(S.vld(keys_ub[0]), query, full, "eq")
                chosen = S.vsel(S.vld(values_ub[0]), neg, pred)
                S.vsts(sel_ub[0], S.vcmax(chosen, full), one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                zero = S.vdup(0.0, T.float32, full)
                flag = S.vdup(accepted, T.int32, full)
                keep = S.vcmps(flag, 0, full, "ne")
                S.vsts(out_ub[0], S.vsel(S.vld(src_ub[0]), zero, keep), full)
            with T.SimdVF():
                full = S.pset(32)
                S.vsts(pool_ub[0], S.vdup(float("-inf"), T.float32, full), full)
                S.mem_bar("VST_VLD")
                S.vscatter(S.vld(values_ub[0]), pool_ub[0], T.reinterpret("uint32x64", S.vld(idx_ub[0])), full)
                S.mem_bar("VST_VLD")
                got = S.vgather2(pool_ub[0], T.reinterpret("uint32x64", S.vld(idx_ub[0])), full)
                S.vsts(values_ub[0], got, full)
            T.copy(sel_ub[0:1], selected)
            T.copy(out_ub, finalized)
            T.copy(values_ub, gathered)

    return kernel


_KERNEL = None


def launch(keys, values, query, src, accepted, indices):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    keys_n = keys.npu()
    device = keys_n.device
    selected = torch.empty(1, device=device)
    finalized = torch.empty(N, device=device)
    gathered = torch.empty(N, device=device)
    _KERNEL(
        keys_n,
        values.npu(),
        int(query),
        src.npu(),
        int(accepted),
        indices.npu(),
        selected,
        finalized,
        gathered,
    )
    return selected.cpu(), finalized.cpu(), gathered.cpu()
