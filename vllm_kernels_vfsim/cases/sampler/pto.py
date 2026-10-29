"""PTO sampler VFs. ``vsel`` takes the mask first; scatter/gather masks are required."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.sampler.asc import N


@tilelang.jit(target="pto")
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
            sel_ub = T.alloc_shared((1,), T.float32)
            T.copy(keys, keys_ub)
            T.copy(values, values_ub)
            T.copy(src, src_ub)
            T.copy(indices, idx_ub)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                neg = V.vbrc(T.float32(float("-inf")), size=N)
                pred = V.vcmps(V.vload(keys_ub[0], size=N), query, mask, "eq")
                chosen = V.vsel(pred, V.vload(values_ub[0], size=N), neg)
                V.vstore(V.vcmax(chosen, mask), sel_ub[0])
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                zero = V.vbrc(T.float32(0.0), size=N)
                flag = V.vbrc(accepted, size=N)
                keep = V.vcmps(flag, T.int32(0), mask, "ne")
                V.vstore(V.vsel(keep, V.vload(src_ub[0], size=N), zero), out_ub[0], mask)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                V.vstore(V.vbrc(T.float32(float("-inf")), size=N), pool_ub[0], mask)
                T.simd.mem_bar("VST_VLD")
                idx = V.vinterpret_cast(V.vload(idx_ub[0], size=N), "uint32")
                vals = V.vload(values_ub[0], size=N)
                V.vscatter(vals, pool_ub[0], idx, mask)
                T.simd.mem_bar("VST_VLD")
                V.vstore(V.vgather(pool_ub[0], idx, mask), values_ub[0], mask)
            T.copy(sel_ub, selected)
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
    _KERNEL(keys_n, values.npu(), int(query), src.npu(), int(accepted), indices.npu(), selected, finalized, gathered)
    return selected.cpu(), finalized.cpu(), gathered.cpu()
