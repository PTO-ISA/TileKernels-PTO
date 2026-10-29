"""PTO quant and scale-layout VFs. Grouped reduce replaces four half-register maxima."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.norm.quant_asc import CLAMP, GROUPS, HIDDEN, QMAX


@tilelang.jit(target="pto")
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
            amax_ub = T.alloc_shared((GROUPS,), T.float32)
            inv_ub = T.alloc_shared((GROUPS,), T.float32)
            T.copy(x, x_ub)
            with T.SimdVF():
                mask = V.create_mask(HIDDEN, size=HIDDEN)
                mask4 = V.create_mask(GROUPS, size=GROUPS)
                values = V.vload(x_ub[0], size=HIDDEN)
                amax = V.vcmax(V.vabs(values, mask), mask, group=GROUPS)
                clamped = V.vmax(amax, V.vbrc(T.float32(CLAMP), size=GROUPS), mask4)
                qmax = V.vbrc(T.float32(QMAX), size=GROUPS)
                V.vstore(V.vdiv(clamped, qmax, mask4), amax_ub[0])
                V.vstore(V.vdiv(qmax, clamped, mask4), inv_ub[0])
                T.simd.mem_bar("VST_VLD")
                slots = V.vload(inv_ub[0], size=GROUPS, stride=1, group=GROUPS)
                inv = V.vbrc(slots, size=HIDDEN, group=GROUPS)
                quant = V.vmul(values, inv, mask)
                V.vstore(V.vcvt(quant, "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[0])
            T.copy(q_ub, q)
            T.copy(amax_ub, sf)

    return kernel


@tilelang.jit(target="pto")
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
                mask = V.create_mask(128, size=128)
                lane = V.vci(T.int16(0), size=128)
                row = V.vand(V.vinterpret_cast(lane, "uint16"), V.vbrc(T.uint16(7), size=128), mask)
                pack = V.vinterpret_cast(V.vshrs(lane, 3, mask), "uint16")
                idx = V.vadd(V.vmuls(row, T.uint16(8), mask), pack, mask)
                V.vstore(idx, idx_ub[0], mask)
                got = V.vgather(table_ub[0], V.vinterpret_cast(idx, "uint16"), mask)
                V.vstore(got, out_ub[0], mask)
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
