"""Rope, Hadamard, and FP8 quant slices of the fused_transform VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477
The reciprocal LUT is compiled only when head_dim is 512 and the cache store
is merged. That arm is attempted in lut_asc.py; a failure is kept, not dropped.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 64


@tilelang.jit(target="ascend")
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
                full = S.pset(32)
                ev = S.vld(e[0])
                od = S.vld(o[0])
                cv = S.vld(c[0])
                sv = S.vneg(S.vld(s[0]), full)
                r0 = S.alloc_var(T.float32)
                r1 = S.alloc_var(T.float32)
                r0 = S.vmul(od, sv, full)
                S.vmula(r0, ev, cv, full)
                r1 = S.vmul(od, cv, full)
                S.vmula(r1, ev, S.vneg(sv, full), full)
                S.vsts(y0[0], r0, full)
                S.vsts(y1[0], r1, full)
            T.copy(y0, out0)
            T.copy(y1, out1)

    return kernel


@tilelang.jit(target="ascend")
def compile_hadamard():
    @T.prim_func
    def kernel(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            out = T.alloc_shared((N,), T.float32)
            T.copy(x, ub)
            with T.SimdVF():
                full = S.pset(32)
                values = S.alloc_var(T.float32)
                values = S.vld(ub[0])
                lane = T.reinterpret(S.vci(0, T.int32), "uint32x64")
                for stage in T.unroll(6, explicit=True):
                    distance = 1 << stage
                    indices = S.vxor(lane, S.vdup(distance, T.uint32, full), full)
                    low = S.vcmps(S.vand(lane, S.vdup(distance, T.uint32, full), full), 0, full, "eq")
                    partner = S.vselr(values, indices)
                    added = S.vadd(values, partner, full)
                    flipped = S.vsub(partner, values, full)
                    values = S.vsel(added, flipped, low)
                S.vsts(out[0], values, full)
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
