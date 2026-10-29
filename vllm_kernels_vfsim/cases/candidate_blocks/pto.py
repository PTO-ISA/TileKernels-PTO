"""PTO candidate-block VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.candidate_blocks.asc import BLOCKS, BLOCK_BITS, LANES


@T.macro
def _prefix(buf):
    mask = V.create_mask(LANES, size=LANES)
    vals = V.alloc_var(T.int32, size=LANES)
    vals = V.vload(buf[0], size=LANES)
    lanes = V.vci(T.int32(0), size=LANES)
    zero = V.vbrc(T.int32(0), size=LANES)
    for bit in T.unroll(6, explicit=True):
        index = V.vmaxs(V.vadds(lanes, T.int32(-(1 << bit)), mask), T.int32(0), mask)
        taken = V.vselr(vals, V.vinterpret_cast(index, "uint32"))
        include = V.vcmps(lanes, T.int32(1 << bit), mask, "ge")
        vals = V.vadd(vals, V.vsel(include, taken, zero), mask)
    V.vstore(vals, buf[0], mask)


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        lengths: T.Tensor((LANES,), T.int32),
        indices: T.Tensor((LANES,), T.int32),
        candidates: T.Tensor((LANES,), T.int32),
        counts: T.Tensor((LANES,), T.int32),
        prefix: T.Tensor((LANES,), T.int32),
        restored: T.Tensor((LANES,), T.int32),
        total: T.Tensor((1,), T.int32),
        compact: T.Tensor((1,), T.int32),
        row_len: T.int32,
    ):
        with T.Kernel(1):
            len_ub = T.alloc_shared((LANES,), T.int32)
            idx_ub = T.alloc_shared((LANES,), T.int32)
            cand_ub = T.alloc_shared((LANES,), T.int32)
            count_ub = T.alloc_shared((LANES,), T.int32)
            prefix_ub = T.alloc_shared((LANES,), T.int32)
            restored_ub = T.alloc_shared((LANES,), T.int32)
            total_ub = T.alloc_shared((8,), T.int32)
            compact_ub = T.alloc_shared((8,), T.int32)
            T.copy(lengths, len_ub)
            T.copy(indices, idx_ub)
            T.copy(candidates, cand_ub)
            with T.SimdVF():
                mask = V.create_mask(LANES, size=LANES)
                raw = V.vinterpret_cast(V.vload(len_ub[0], size=LANES), "uint32")
                bumped = V.vadds(raw, T.uint32(16383), mask)
                task_counts = V.vshrs(bumped, 14, mask)
                as_int = V.vinterpret_cast(task_counts, "int32")
                V.vstore(as_int, count_ub[0], mask)
                V.vstore(V.vcadd(as_int, mask), total_ub[0])
                V.vstore(as_int, prefix_ub[0], mask)
            with T.SimdVF():
                _prefix(prefix_ub)
            with T.SimdVF():
                mask = V.create_mask(LANES, size=LANES)
                neg_one = V.vbrc(T.int32(-1), size=LANES)
                index = V.vload(idx_ub[0], size=LANES)
                slot_u = V.vshrs(V.vinterpret_cast(index, "uint32"), BLOCK_BITS, mask)
                slot = V.vinterpret_cast(slot_u, "int32")
                nonnegative = V.vcmps(index, T.int32(0), mask, "ge")
                in_bounds = V.vcmps(slot, T.int32(BLOCKS), nonnegative, "lt")
                block_id = V.vgather(cand_ub[0], V.vinterpret_cast(slot_u, "uint32"), in_bounds)
                valid = V.vand(in_bounds, V.vcmps(block_id, T.int32(0), mask, "ge"))
                remainder = V.vand(index, V.vbrc(T.int32(15), size=LANES), mask)
                value = V.vadd(V.vshls(block_id, BLOCK_BITS, mask), remainder, mask)
                V.vstore(V.vsel(valid, value, neg_one), restored_ub[0], mask)
            with T.SimdVF():
                mask = V.create_mask(LANES, size=LANES)
                lanes = V.vci(T.int32(0), size=LANES)
                active = V.vcmps(lanes, T.int32(BLOCKS), mask, "lt")
                ids = V.vload(cand_ub[0], size=LANES)
                start = V.vmuls(ids, T.int32(16), mask)
                remain = V.vsub(V.vbrc(row_len, size=LANES), start, mask)
                capped = V.vmins(V.vmaxs(remain, T.int32(0), mask), T.int32(16), mask)
                capped = V.vsel(active, capped, V.vbrc(T.int32(0), size=LANES))
                V.vstore(V.vcadd(capped, mask), compact_ub[0])
            T.copy(count_ub, counts)
            T.copy(prefix_ub, prefix)
            T.copy(restored_ub, restored)
            T.copy(total_ub[0:1], total)
            T.copy(compact_ub[0:1], compact)

    return kernel


_KERNEL = None


def launch(lengths, indices, candidates, row_len):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    lengths_n = lengths.npu()
    device = lengths_n.device
    counts = torch.empty(LANES, dtype=torch.int32, device=device)
    prefix = torch.empty(LANES, dtype=torch.int32, device=device)
    restored = torch.empty(LANES, dtype=torch.int32, device=device)
    total = torch.empty(1, dtype=torch.int32, device=device)
    compact = torch.empty(1, dtype=torch.int32, device=device)
    _KERNEL(lengths_n, indices.npu(), candidates.npu(), counts, prefix, restored, total, compact, int(row_len))
    return counts.cpu(), prefix.cpu(), restored.cpu(), total.cpu(), compact.cpu()
