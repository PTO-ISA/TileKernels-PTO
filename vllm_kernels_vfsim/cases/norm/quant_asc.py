"""FP32 per-32 quant and the scale-transpose index/gather VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L277
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L382
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L420
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

HIDDEN = 128
GROUPS = 4
CLAMP = 1e-4
QMAX = 448.0


@tilelang.jit(target="ascend")
def compile_quant():
    @T.prim_func
    def kernel(
        x: T.Tensor((HIDDEN,), T.float32),
        q: T.Tensor((HIDDEN,), T.float8_e4m3fn),
        sf: T.Tensor((GROUPS,), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((HIDDEN,), T.float32)
            q_ub = T.alloc_shared((HIDDEN,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((64,), T.float32)
            inv_ub = T.alloc_shared((64,), T.float32)
            T.copy(x, x_ub)
            with T.SimdVF():
                low = S.pset(32, "PAT_VL32")
                all_lanes = S.pset(32, "PAT_ALL")
                high = S.pnot(low, all_lanes)
                zero = S.vdup(0.0, T.float32)
                S.vsts(amax_ub[0], zero)
                v0 = S.vabs(S.vld(x_ub[0]))
                v1 = S.vabs(S.vld(x_ub[64]))
                S.vsts(amax_ub[0], S.vcmax(v0, low), dist="ONEPT_B32")
                S.vsts(amax_ub[1], S.vcmax(v0, high), dist="ONEPT_B32")
                S.vsts(amax_ub[2], S.vcmax(v1, low), dist="ONEPT_B32")
                S.vsts(amax_ub[3], S.vcmax(v1, high), dist="ONEPT_B32")
                S.mem_bar("VST_VLD")
                clamped = S.vmaxs(S.vld(amax_ub[0]), CLAMP)
                qmax = S.vdup(QMAX, T.float32)
                S.vsts(inv_ub[0], S.vdiv(qmax, clamped))
                S.vsts(amax_ub[0], S.vdiv(clamped, qmax))
                S.mem_bar("VST_VLD")
                i0 = S.vld(inv_ub[0], dist="BRC_B32")
                i1 = S.vld(inv_ub[1], dist="BRC_B32")
                i2 = S.vld(inv_ub[2], dist="BRC_B32")
                i3 = S.vld(inv_ub[3], dist="BRC_B32")
                q0 = S.vmul(S.vld(x_ub[0]), S.vsel(i0, i1, low))
                q1 = S.vmul(S.vld(x_ub[64]), S.vsel(i2, i3, low))
                S.vsts(q_ub[0], S.vcvt(q0, T.float8_e4m3fn), dist="PK4_B32")
                S.vsts(q_ub[64], S.vcvt(q1, T.float8_e4m3fn), dist="PK4_B32")
            T.copy(q_ub, q)
            T.copy(amax_ub[0:GROUPS], sf)

    return kernel


@tilelang.jit(target="ascend")
def compile_layout():
    @T.prim_func
    def kernel(
        table: T.Tensor((128,), T.uint16),
        indices: T.Tensor((128,), T.uint16),
        gathered: T.Tensor((128,), T.uint16),
    ):
        with T.Kernel(1):
            table_ub = T.alloc_shared((128,), T.uint16)
            idx_ub = T.alloc_shared((128,), T.uint16)
            out_ub = T.alloc_shared((128,), T.uint16)
            T.copy(table, table_ub)
            with T.SimdVF():
                lane = S.vci(0, T.int16)
                row = S.vand(T.reinterpret(lane, "uint16x128"), S.vdup(7, T.uint16))
                pack = T.reinterpret(S.vshrs(lane, 3), "uint16x128")
                idx = S.vadd(S.vmuls(row, T.uint16(8)), pack)
                S.vsts(idx_ub[0], idx, dist="NORM_B16")
                got = S.vgather2(table_ub[0], idx)
                S.vsts(out_ub[0], got, dist="NORM_B16")
            T.copy(idx_ub, indices)
            T.copy(out_ub, gathered)

    return kernel


_QUANT = None
_LAYOUT = None


def launch_quant(x):
    global _QUANT
    if _QUANT is None:
        _QUANT = compile_quant()
    import torch

    x_n = x.npu()
    q = torch.empty(HIDDEN, dtype=torch.float8_e4m3fn, device=x_n.device)
    sf = torch.empty(GROUPS, dtype=torch.float32, device=x_n.device)
    _QUANT(x_n, q, sf)
    return q.cpu(), sf.cpu()


def launch_layout(table):
    global _LAYOUT
    if _LAYOUT is None:
        _LAYOUT = compile_layout()
    import torch

    table_n = table.npu()
    indices = torch.empty(128, dtype=torch.uint16, device=table_n.device)
    gathered = torch.empty(128, dtype=torch.uint16, device=table_n.device)
    _LAYOUT(table_n, indices, gathered)
    return indices.cpu(), gathered.cpu()
