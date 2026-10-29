"""ASC candidate-block VFs on one 64-lane tile.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

LANES = 64
BLOCKS = 4
BLOCK_BITS = 4


@T.macro
def _prefix(buf):
    mask = S.pset(32)
    vals = S.alloc_var(T.int32)
    vals = S.vld(buf[0])
    lanes = S.vci(0, "int32")
    zero = S.vdup(0, T.int32, mask)
    for bit in T.unroll(6, explicit=True):
        index = S.vmaxs(S.vadds(lanes, -(1 << bit), mask), 0, mask)
        taken = S.vselr(vals, T.reinterpret("uint32x64", index))
        include = S.vcmps(lanes, 1 << bit, mask, "ge")
        vals = S.vadd(vals, S.vsel(taken, zero, include), mask)
    S.vsts(buf[0], vals, mask)


@tilelang.jit(target="ascend")
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
            compact_ub = T.alloc_shared((LANES,), T.int32)
            T.copy(lengths, len_ub)
            T.copy(indices, idx_ub)
            T.copy(candidates, cand_ub)
            with T.SimdVF():
                full = S.pset(32)
                zero = S.vdup(0, T.uint32, full)
                S.vsts(count_ub[0], T.reinterpret("int32x64", zero), full)
            with T.SimdVF():
                full = S.pset(32)
                lanes = S.vci(0, "int32")
                raw = T.reinterpret("uint32x64", S.vld(len_ub[0]))
                active = S.vcmps(lanes, LANES, full, "lt")
                bumped = S.vadds(raw, T.uint32(16384 - 1), full)
                task_counts = S.vshrs(bumped, 14, full)
                kept = S.vsel(task_counts, S.vdup(0, T.uint32, full), active)
                S.vsts(count_ub[0], T.reinterpret("int32x64", kept), full)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                total_v = S.vcadd(S.vld(count_ub[0]), full)
                S.vsts(total_ub[0], total_v, one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                S.vsts(prefix_ub[0], S.vld(count_ub[0]), full)
            with T.SimdVF():
                _prefix(prefix_ub)
            with T.SimdVF():
                full = S.pset(32)
                neg_one = S.vdup(-1, T.int32, full)
                remainder_mask = S.vdup(15, T.int32, full)
                index = S.vld(idx_ub[0])
                slot = S.vshrs(T.reinterpret("uint32x64", index), BLOCK_BITS, full)
                nonnegative = S.vcmps(index, 0, full, "ge")
                in_bounds = S.vcmps(slot, T.uint32(BLOCKS), nonnegative, "lt")
                block_id = S.vgather2(cand_ub[0], slot, in_bounds)
                valid = S.pand(in_bounds, S.vcmps(block_id, 0, full, "ge"), full)
                remainder = S.vand(index, remainder_mask, full)
                value = S.vadd(S.vshls(block_id, BLOCK_BITS, full), remainder, full)
                S.vsts(restored_ub[0], S.vsel(value, neg_one, valid), full)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                ids = S.vld(cand_ub[0])
                lanes = S.vci(0, "int32")
                active = S.vcmps(lanes, BLOCKS, full, "lt")
                start = S.vmuls(ids, T.int32(16), full)
                remain = S.vsub(S.vdup(row_len, T.int32, full), start, full)
                capped = S.vmins(S.vmaxs(remain, 0, full), 16, full)
                capped = S.vsel(capped, S.vdup(0, T.int32, full), active)
                S.vsts(compact_ub[0], capped, full)
                summed = S.vcadd(capped, full)
                S.vsts(total_ub[1], summed, one, "ONEPT_B32")
            T.copy(count_ub, counts)
            T.copy(prefix_ub, prefix)
            T.copy(restored_ub, restored)
            T.copy(total_ub[0:1], total)
            T.copy(total_ub[1:2], compact)

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
