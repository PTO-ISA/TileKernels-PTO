"""PTO VMI FP8 variants used to isolate sequential-store lowering cost."""

from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from .common import FLOAT8E4M3_MAX, FP8_AMAX_MARGIN
from .store_common import OUT_ROW_BYTES, SF_COL, TOKENS_PER_TILE, make_store_tl


def _make_scale(lanes: int):
    @T.macro
    def compute_scale(amax: Any, mask: Any, one: Any, bias: Any):
        clamped = V.vmaxs(amax, T.float32(FP8_AMAX_MARGIN), mask)
        bits = V.vinterpret_cast(V.vmuls(clamped, T.float32(1.0 / FLOAT8E4M3_MAX), mask), "uint32")
        exponent = V.vadds(V.vshrs(V.vsub(bits, one, mask), 23, mask), T.uint32(1), mask)
        inverse = V.vinterpret_cast(V.vshls(V.vsub(bias, exponent, mask), 23, mask), "float32")
        return exponent, inverse

    return compute_scale


compute_scale4 = _make_scale(4)
compute_scale64 = _make_scale(64)


@T.macro
def store_scale_bytes(scale: Any, scales: T.Buffer, row: Any, col: Any, count: int):
    mask = V.create_mask(count, size=count)
    V.vstore(V.vcvt(scale, "uint8"), scales[row, col], mask)


@T.macro
def quantize_native_half(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any, hoisted: bool = False):
    with T.SimdVF():
        if hoisted:
            mask64 = V.create_mask(64, size=64)
            one = V.vbrc(T.uint32(1), size=64)
            bias = V.vbrc(T.uint32(254), size=64)
        for h in T.serial(active):
            for p in T.unroll(4, explicit=True):
                for half in T.unroll(2, explicit=True):
                    offset = p * 128 + half * 64
                    values = V.vcvt(V.vload(x[h, offset], size=64), "float32")
                    mask = V.create_mask(64, size=64)
                    amax = V.vcmax(V.vabs(values, mask), mask, group=2)
                    amax = V.vbrc(amax, size=64, group=2)
                    if hoisted:
                        encoded_dense, inverse = compute_scale64(amax, mask64, one, bias)
                    else:
                        encoded_dense, inverse = compute_scale64(amax, V.create_mask(64, size=64), V.vbrc(T.uint32(1), size=64), V.vbrc(T.uint32(254), size=64))
                    encoded = V.vcmax(encoded_dense, mask, group=2)
                    packed = V.vcvt(V.vmul(values, inverse, mask), "float8_e4m3fn", rounding="R", saturate="SAT")
                    V.vstore(packed, out[h, offset])
                    store_scale_bytes(encoded, scales, h, SF_COL + p * 4 + half * 2, 2)


@T.macro
def quantize_tile_native(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    quantize_native_half(x, out, scales, active, False)


@T.macro
def quantize_tile_hoisted(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    quantize_native_half(x, out, scales, active, True)


@T.macro
def quantize_tile_compact(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    with T.SimdVF():
        for h in T.serial(active):
            for p in T.unroll(4, explicit=True):
                values = V.vcvt(V.vload(x[h, p * 128], size=128), "float32")
                mask = V.create_mask(128, size=128)
                amax = V.vcmax(V.vabs(values, mask), mask, group=4)
                mask4 = V.create_mask(4, size=4)
                encoded, inverse = compute_scale4(amax, mask4, V.vbrc(T.uint32(1), size=4), V.vbrc(T.uint32(254), size=4))
                inv_ub = T.alloc_shared((4,), T.float32)
                V.vstore(inverse, inv_ub[0], mask4)
                T.simd.mem_bar("VST_VLD")
                slots = V.vload(inv_ub[0], size=4, group=4, stride=1)
                broadcast = V.vbrc(slots, size=128, group=4)
                packed = V.vcvt(V.vmul(values, broadcast, mask), "float8_e4m3fn", rounding="R", saturate="SAT")
                V.vstore(packed, out[h, p * 128])
                store_scale_bytes(encoded, scales, h, SF_COL + p * 4, 4)


@T.macro
def quantize_tile_direct_impl(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any, grouped_mask: bool, packed_scale: bool):
    if packed_scale:
        sf_words = T.view(scales, (TOKENS_PER_TILE, OUT_ROW_BYTES // 4), T.uint32)
    with T.SimdVF():
        for h in T.serial(active):
            for p in T.unroll(4, explicit=True):
                values = V.vcvt(V.vload(x[h, p * 128], size=128), "float32")
                mask = V.create_mask(128, size=128)
                amax = V.vcmax(V.vabs(values, mask), mask, group=4)
                if grouped_mask:
                    mask4 = V.create_mask(1, size=4, group=4)
                else:
                    mask4 = V.create_mask(4, size=4)
                encoded, inverse = compute_scale4(amax, mask4, V.vbrc(T.uint32(1), size=4), V.vbrc(T.uint32(254), size=4))
                broadcast = V.vbrc(inverse, size=128, group=4)
                packed = V.vcvt(V.vmul(values, broadcast, mask), "float8_e4m3fn", rounding="R", saturate="SAT")
                V.vstore(packed, out[h, p * 128])
                if packed_scale:
                    _, high16 = V.vunzip(V.vinterpret_cast(encoded, "uint32"), to_dtype="uint16")
                    exponent16 = V.vshrs(high16, 7, mask4)
                    exponent8 = V.vcvt(exponent16, "uint8", saturate="NOSAT")
                    V.vstore(V.vinterpret_cast(exponent8, "uint32"), sf_words[h, (SF_COL + p * 4) // 4])
                else:
                    store_scale_bytes(encoded, scales, h, SF_COL + p * 4, 4)


@T.macro
def quantize_tile_direct(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    quantize_tile_direct_impl(x, out, scales, active, False, False)


@T.macro
def quantize_tile_direct_grouped(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    quantize_tile_direct_impl(x, out, scales, active, True, False)


@T.macro
def quantize_tile_direct_packed(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    quantize_tile_direct_impl(x, out, scales, active, True, True)


@T.macro
def quantize_tile_scale_only(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    with T.SimdVF():
        zero = V.vcvt(V.vbrc(T.float32(0.0), size=64), "float8_e4m3fn")
        for h in T.serial(active):
            for p in T.unroll(4, explicit=True):
                for half in T.unroll(2, explicit=True):
                    offset = p * 128 + half * 64
                    values = V.vcvt(V.vload(x[h, offset], size=64), "float32")
                    mask = V.create_mask(64, size=64)
                    amax = V.vcmax(V.vabs(values, mask), mask, group=2)
                    amax = V.vbrc(amax, size=64, group=2)
                    encoded_dense, unused_inverse = compute_scale64(amax, V.create_mask(64, size=64), V.vbrc(T.uint32(1), size=64), V.vbrc(T.uint32(254), size=64))
                    del unused_inverse
                    encoded = V.vcmax(encoded_dense, mask, group=2)
                    V.vstore(zero, out[h, offset])
                    store_scale_bytes(encoded, scales, h, SF_COL + p * 4 + half * 2, 2)


@T.macro
def quantize_tile_pack_only(x: T.Buffer, out: T.Buffer, scales: T.Buffer, active: Any):
    with T.SimdVF():
        for h in T.serial(active):
            for p in T.unroll(4, explicit=True):
                for half in T.unroll(2, explicit=True):
                    offset = p * 128 + half * 64
                    values = V.vcvt(V.vload(x[h, offset], size=64), "float32")
                    mask = V.create_mask(64, size=64)
                    inverse = V.vbrc(T.float32(1.0), size=64)
                    packed = V.vcvt(V.vmul(values, inverse, mask), "float8_e4m3fn", rounding="R", saturate="SAT")
                    V.vstore(packed, out[h, offset])
                    store_scale_bytes(V.vbrc(T.uint32(0), size=2), scales, h, SF_COL + p * 4 + half * 2, 2)


VARIANTS = {
    "native_half": quantize_tile_native,
    "compact_ub": quantize_tile_compact,
    "direct_brc": quantize_tile_direct,
    "direct_brc_grouped": quantize_tile_direct_grouped,
    "direct_brc_packed": quantize_tile_direct_packed,
    "native_half_hoisted": quantize_tile_hoisted,
    "scale_only": quantize_tile_scale_only,
    "pack_only": quantize_tile_pack_only,
}


@tilelang.jit(
    target="pto",
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    },
)
def store_kv_pto(variant: str) -> Any:
    if variant not in VARIANTS:
        raise ValueError(f"unknown PTO variant {variant!r}")
    return make_store_tl(VARIANTS[variant])
