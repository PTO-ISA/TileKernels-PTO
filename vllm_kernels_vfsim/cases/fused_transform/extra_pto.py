"""PTO rope and Hadamard. Rotation is two multiplies and an add, not part-select."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_transform.extra_asc import N


@tilelang.jit(target="pto")
def compile_rope():
    @T.prim_func
    def kernel(
        even: T.Tensor((N,), T.float32),
        odd: T.Tensor((N,), T.float32),
        cos: T.Tensor((N,), T.float32),
        sin: T.Tensor((N,), T.float32),
        out0: T.Tensor((N,), T.float32),
        out1: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(1):
            e = T.alloc_shared((N,), T.float32)
            o = T.alloc_shared((N,), T.float32)
            c = T.alloc_shared((N,), T.float32)
            s = T.alloc_shared((N,), T.float32)
            y0 = T.alloc_shared((N,), T.float32)
            y1 = T.alloc_shared((N,), T.float32)
            T.copy(even, e)
            T.copy(odd, o)
            T.copy(cos, c)
            T.copy(sin, s)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                ev = V.vload(e[0], size=N)
                od = V.vload(o[0], size=N)
                cv = V.vload(c[0], size=N)
                sv = V.vneg(V.vload(s[0], size=N), mask)
                r0 = V.vadd(V.vmul(od, sv, mask), V.vmul(ev, cv, mask), mask)
                r1 = V.vadd(V.vmul(od, cv, mask), V.vmul(ev, V.vneg(sv, mask), mask), mask)
                V.vstore(r0, y0[0], mask)
                V.vstore(r1, y1[0], mask)
            T.copy(y0, out0)
            T.copy(y1, out1)

    return kernel


@tilelang.jit(target="pto")
def compile_hadamard():
    @T.prim_func
    def kernel(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            out = T.alloc_shared((N,), T.float32)
            T.copy(x, ub)
            with T.SimdVF():
                mask = V.create_mask(N, size=N)
                values = V.alloc_var(T.float32, size=N)
                values = V.vload(ub[0], size=N)
                lane = V.vinterpret_cast(V.vci(T.int32(0), size=N), "uint32")
                for stage in T.unroll(6, explicit=True):
                    distance = 1 << stage
                    dist = V.vbrc(T.uint32(distance), size=N)
                    indices = V.vxor(lane, dist, mask)
                    low = V.vcmps(V.vand(lane, dist, mask), T.uint32(0), mask, "eq")
                    partner = V.vselr(values, indices)
                    added = V.vadd(values, partner, mask)
                    flipped = V.vsub(partner, values, mask)
                    values = V.vsel(low, added, flipped)
                V.vstore(values, out[0], mask)
            T.copy(out, y)

    return kernel


_ROPE = None
_HAD = None


def launch_rope(even, odd, cos, sin):
    global _ROPE
    if _ROPE is None:
        _ROPE = compile_rope()
    src = even.npu()
    import torch

    y0 = torch.empty_like(src)
    y1 = torch.empty_like(src)
    _ROPE(src, odd.npu(), cos.npu(), sin.npu(), y0, y1)
    return y0.cpu(), y1.cpu()


def launch_hadamard(x):
    global _HAD
    if _HAD is None:
        _HAD = compile_hadamard()
    src = x.npu()
    import torch

    y = torch.empty_like(src)
    _HAD(src, y)
    return y.cpu()
