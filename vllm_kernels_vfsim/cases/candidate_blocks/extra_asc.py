"""Binary-search task map and bf16 block max from candidate_blocks.

https://github.com/deepseek-ai/DeepSeek-AI/DeepInfer is not the path.
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py#L149
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py#L198
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

LANES = 64
ROWS = 4


@tilelang.jit(target="ascend")
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
                full = S.pset(32)
                positions = S.vci(0, T.int32)
                local_row = S.alloc_var(T.int32)
                local_row = S.vdup(0, T.int32, full)
                zero = S.vdup(0, T.int32, full)
                ends_v = S.vld(ends_ub[0])
                for bit in T.unroll(6, explicit=True):
                    next_row = S.vadds(local_row, 1 << (5 - bit), full)
                    index = S.vmins(S.vadds(next_row, -1, full), 63, full)
                    row_end = S.vselr(ends_v, T.reinterpret("uint32x64", index))
                    in_range = S.vcmps(next_row, ROWS, full, "lt")
                    take = S.pand(in_range, S.vcmp(row_end, positions, full, "le"), full)
                    local_row = S.vsel(next_row, local_row, take)
                previous = S.vmaxs(S.vadds(local_row, -1, full), 0, full)
                row_start = S.vselr(ends_v, T.reinterpret("uint32x64", previous))
                nonfirst = S.vcmps(local_row, 0, full, "gt")
                tile = S.vsub(positions, S.vsel(row_start, zero, nonfirst), full)
                task = S.vadd(S.vmuls(local_row, 8, full), tile, full)
                length = S.vselr(S.vld(len_ub[0]), T.reinterpret("uint32x64", local_row))
                low, high = S.vintlv(task, length)
                S.vsts(task_ub[0], low, full)
                S.vsts(lens_ub[0], high, full)
            T.copy(task_ub, tasks)
            T.copy(lens_ub, lens_out)

    return kernel


@tilelang.jit(target="ascend")
def compile_max():
    @T.prim_func
    def kernel(logits: T.Tensor((LANES,), T.bfloat16), maxima: T.Tensor((8,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((LANES,), T.bfloat16)
            out = T.alloc_shared((8,), T.float32)
            T.copy(logits, ub)
            with T.SimdVF():
                full = S.pset(32)
                eight = S.pset(32, "PAT_VL8")
                values = S.vcvt(S.vld(ub[0], dist="UNPK_B16"), T.float32)
                reduced = S.vcgmax(values, full)
                S.vsts(out[0], reduced, eight, extent=8)
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
