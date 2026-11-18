"""Original DeepInfer physical-SIMD FP8 quantization path."""

from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from .common import FLOAT8E4M3_MAX, FP8_AMAX_MARGIN
from .store_common import SF_COL, make_store_tl


@T.macro
def compute_scale(amax: Any):
    clamped = S.vmaxs(amax, FP8_AMAX_MARGIN)
    bits = T.reinterpret(S.vmuls(clamped, 1.0 / FLOAT8E4M3_MAX), "uint32x64")
    exponent = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1)
    inverse = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), exponent), 23), "float32x64")
    return exponent, inverse


@T.macro
def compute_scales(even: Any, odd: Any):
    maxima16 = S.vcgmax(S.vmax(S.vabs(even), S.vabs(odd)))
    lane = T.reinterpret(S.vci(0, T.int32), "uint32x64")
    even_maxima, odd_maxima = S.vdintlv(maxima16, maxima16)
    amax = S.vmax(even_maxima, odd_maxima)
    encoded, inverse = compute_scale(amax)
    return S.vselr(inverse, S.vshrs(lane, 4)), encoded


@T.macro
def quantize_value(even: Any, odd: Any, inverse: Any):
    q0 = S.vcvt(S.vmul(even, inverse), T.float8_e4m3fn, part=0)
    q1 = S.vcvt(S.vmul(odd, inverse), T.float8_e4m3fn, part=1)
    return S.vor(T.reinterpret(q0, "uint32x64"), T.reinterpret(q1, "uint32x64"))


@T.macro
def quantize_tile(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    with T.SimdVF():
        lane = T.reinterpret(S.vci(0, T.int32), "uint32x64")
        full_b32 = S.pset(32)
        for h in T.serial(active):
            sf_row = S.alloc_var(T.uint32)
            sf_row = S.vdup(0, T.uint32)
            for p in T.unroll(4, explicit=True):
                raw = S.vld(x[h, p * 128])
                even = S.vcvt(raw, T.float32, part=0)
                odd = S.vcvt(raw, T.float32, part=1)
                inverse, scale = compute_scales(even, odd)
                packed = quantize_value(even, odd, inverse)
                S.vsts(out[h, p * 128], T.reinterpret(packed, "float8_e4m3fnx256"), full_b32, dist="PK_B32", extent=128)
                relative = S.vsub(lane, S.vdup(p * 4, T.uint32))
                indices = S.vand(relative, S.vdup(3, T.uint32))
                sf_row = S.vsel(S.vselr(scale, indices), sf_row, S.vcmps(relative, 4, op="lt"))
            sf_mask = S.vcmps(lane, 16, op="lt")
            S.vsts(scales[h, SF_COL], T.reinterpret(sf_row, "uint8x256"), sf_mask, dist="PK4_B32", extent=16)


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    },
)
def store_kv_asc() -> Any:
    return make_store_tl(quantize_tile)
