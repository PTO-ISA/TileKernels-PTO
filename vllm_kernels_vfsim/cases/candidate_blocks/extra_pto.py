"""PTO task map and block max. The max is one grouped reduction."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.candidate_blocks.extra_asc import LANES, ROWS


@tilelang.jit(target="pto")
def compile_map():
    @T.prim_func
    def kernel(
        ends: T.Tensor((LANES,), T.int32),
        lengths: T.Tensor((LANES,), T.int32),
        tasks: T.Tensor((LANES,), T.int32),
        lens_out: T.Tensor((LANES,), T.int32),
    ):
        with T.Kernel(1):
            ends_ub = T.alloc_shared((LANES,), T.int32)
            len_ub = T.alloc_shared((LANES,), T.int32)
            task_ub = T.alloc_shared((LANES,), T.int32)
            lens_ub = T.alloc_shared((LANES,), T.int32)
            T.copy(ends, ends_ub)
            T.copy(lengths, len_ub)
            with T.SimdVF():
                full = V.create_mask(LANES, size=LANES)
                positions = V.vci(T.int32(0), size=LANES)
                local_row = V.alloc_var(T.int32, size=LANES)
                local_row = V.vbrc(T.int32(0), size=LANES)
                zero = V.vbrc(T.int32(0), size=LANES)
                ends_v = V.vload(ends_ub[0], size=LANES)
                for bit in T.unroll(6, explicit=True):
                    next_row = V.vadds(local_row, T.int32(1 << (5 - bit)), full)
                    index = V.vinterpret_cast(V.vmins(V.vadds(next_row, T.int32(-1), full), T.int32(63), full), "uint32")
                    row_end = V.vselr(ends_v, index)
                    in_range = V.vcmps(next_row, T.int32(ROWS), full, "lt")
                    take = V.vand(in_range, V.vcmp(row_end, positions, full, "le"))
                    local_row = V.vsel(take, next_row, local_row)
                previous = V.vmaxs(V.vadds(local_row, T.int32(-1), full), T.int32(0), full)
                row_start = V.vselr(ends_v, V.vinterpret_cast(previous, "uint32"))
                nonfirst = V.vcmps(local_row, T.int32(0), full, "gt")
                tile = V.vsub(positions, V.vsel(nonfirst, row_start, zero), full)
                task = V.vadd(V.vmuls(local_row, T.int32(8), full), tile, full)
                length = V.vselr(V.vload(len_ub[0], size=LANES), V.vinterpret_cast(local_row, "uint32"))
                low, high = V.vintlv(task, length, full)
                V.vstore(low, task_ub[0], full)
                V.vstore(high, lens_ub[0], full)
            T.copy(task_ub, tasks)
            T.copy(lens_ub, lens_out)

    return kernel


@tilelang.jit(target="pto")
def compile_max():
    @T.prim_func
    def kernel(logits: T.Tensor((LANES,), T.bfloat16), maxima: T.Tensor((8,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((LANES,), T.bfloat16)
            out = T.alloc_shared((8,), T.float32)
            T.copy(logits, ub)
            with T.SimdVF():
                mask = V.create_mask(LANES, size=LANES)
                values = V.vcvt(V.vload(ub[0], size=LANES), "float32")
                reduced = V.vcmax(values, mask, group=8)
                V.vstore(reduced, out[0])
            T.copy(out, maxima)

    return kernel


_MAP = None
_MAX = None


def launch_map(ends, lengths):
    global _MAP
    if _MAP is None:
        _MAP = compile_map()
    import torch

    src = ends.npu()
    tasks = torch.empty(LANES, dtype=torch.int32, device=src.device)
    lens = torch.empty(LANES, dtype=torch.int32, device=src.device)
    _MAP(src, lengths.npu(), tasks, lens)
    return tasks.cpu(), lens.cpu()


def launch_max(logits):
    global _MAX
    if _MAX is None:
        _MAX = compile_max()
    import torch

    src = logits.npu()
    out = torch.empty(8, device=src.device)
    _MAX(src, out)
    return out.cpu()
