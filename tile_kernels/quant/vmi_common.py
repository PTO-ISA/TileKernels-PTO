"""Shared MXFP8 / SoftPipeline helpers for VMI quant kernels.

Format and vector geometry live here so each ``*_vmi.py`` stays ASC-sparse at
module top. ``RECIP_SLOT`` is the PTOAS pad tax (compact inv ``vbrc`` fails).
"""

from tilelang.transform import PassConfigKey
from tilelang.ascend import language as T

# MXFP8 / OCP
MX_GROUP = 32
FP8_E4M3_MAX = 448.0
CLAMP_MIN = 1e-4

# Per-channel amax / quant chunk widths
CHUNK_128 = 128
CHUNK_64 = 64

# VMI full-vector strip length (e4m3 / bf16)
VL = 256


LEGAL_MASK_LANES = frozenset({1, 2, 4, 8, 64, 128, 256})


def vmi_strip(hidden: int) -> int:
    """Return the legal VMI strip for ``hidden``.

    Full-vector ``V<256×T>`` when H is 256-aligned. Compact leftover
    H=128/384 cannot use ``V<128×T>`` (PTOAS ``VMI-RESIDUAL-OP``); two
    legal ``V<64×T>`` strips are still logical VMI (design §1.1).
    Host-pad to 256 is not a port.
    """
    if hidden <= 0:
        raise ValueError(f'hidden must be positive, got {hidden}')
    if hidden % VL == 0:
        return VL
    if hidden % CHUNK_128 == 0:
        return CHUNK_64
    raise ValueError(f'hidden must be a multiple of {CHUNK_128}, got {hidden}')


def vmi_half_lanes(vl: int) -> int:
    """Half-strip mask width, snapped to a PTOAS-legal mask size.

    ``VL=64`` has no legal 32-lane mask; callers must treat the full
    strip as the only vector (do not emit a 32-lane half-load).
    """
    half = vl // 2
    if half in LEGAL_MASK_LANES:
        return half
    return vl

# UE8M0 ↔ bf16 bit tricks (ASC inlines the same values)
E8M0_SHIFT = 7
BF16_INF = 0x7F80
BF16_NAN = 0x7FC0
BF16_ABS_MASK = 0x7FFF
BF16_EXP_MASK = 0x7F80
BF16_EXP_STEP = 0x0080
BF16_EXP_BIAS = 0x7F00
FP8_E4M3_MAX_EXP = 0x0400
FP8_E4M3_MAX_MANTISSA = 0x0060  # 448 = 1.75 * 2^8
FP4_E2M1_MAX = 6.0
FP4_E2M1_CLAMP_MIN = FP4_E2M1_MAX * (2.0 ** -126)
BF16_CLAMP_MIN_BITS = 0x38D2  # ~1e-4 as bf16
# Branchless mantissa-round bias: mantissa + MANTISSA_ROUND_BIAS carries into
# bit 7 (exponent field) exactly when mantissa > FP8_E4M3_MAX_MANTISSA (96),
# replacing a vcmp+vsel pair with vadds+vand.
MANTISSA_ROUND_BIAS = 0x80 - FP8_E4M3_MAX_MANTISSA - 1  # 31 = 0x1F

# PTOAS: pad recip to this width before dist_mode='brc' reload
RECIP_SLOT = 32

NUM_STAGES = 2
# Match TileLang ASC's default for a bare S.vdiv: precise FP32 division unless
# a caller explicitly enables fast math for the whole kernel.
FAST_MATH = False
PTO_FAST_MATH = {PassConfigKey.TL_ENABLE_FAST_MATH: FAST_MATH}


def vdiv_0ulp(numer, denom, mask, *, lanes: int):
    """Correctly-rounded fp32 vdiv via the TileLang frontend helper."""
    return T.vmi.vdiv(numer, denom, mask, precision='exact')


def vdiv_0ulp_ftz_true(src0, src1, mask, *, size: int):
    """Alias of ``vdiv_0ulp`` with the main-branch ``size`` keyword."""
    return vdiv_0ulp(src0, src1, mask, lanes=size)


def fast_math_div_core(src0, src1, mask=None, *, size: int):
    """VMI f32 version of CANN ``DivPrecisionImpl`` for finite inputs.

    Selects the quotient or its adjacent f32 bit patterns by minimum residual.
    The no-round MXFP8 caller clamps ``amax`` first, so both operands are
    finite and non-zero; the ASC special-value preservation path is therefore
    intentionally unnecessary here.
    """
    if mask is None:
        mask = T.vmi.create_mask(size, size=size)
    z = T.vmi.vdiv(src0, src1, mask)
    y = T.vmi.vmul(src1, T.vmi.vbrc(T.float32(-1.0), size=size), mask)
    r = T.vmi.vmula(src0, z, y, mask)
    z_bits = T.vmi.vinterpret_cast(z, 'uint32')
    one_u = T.vmi.vbrc(T.uint32(1), size=size)
    z_pre = T.vmi.vinterpret_cast(T.vmi.vsub(z_bits, one_u, mask), 'float32')
    z_next = T.vmi.vinterpret_cast(T.vmi.vadd(z_bits, one_u, mask), 'float32')
    r_pre = T.vmi.vmula(src0, z_pre, y, mask)
    r_next = T.vmi.vmula(src0, z_next, y, mask)
    r = T.vmi.vabs(r, mask)
    r_pre = T.vmi.vabs(r_pre, mask)
    r_next = T.vmi.vabs(r_next, mask)
    choose_pre = T.vmi.vcmp(r, r_pre, mask, 'lt')
    r = T.vmi.vsel(choose_pre, r, r_pre)
    z = T.vmi.vsel(choose_pre, z, z_pre)
    choose_next = T.vmi.vcmp(r_next, r, mask, 'lt')
    z = T.vmi.vsel(choose_next, z_next, z)
    return z


def fast_math_div(src0, src1, mask=None, *, size: int, fast_math: bool):
    """Select the VMI division implementation from the kernel fast-math mode.

    This mirrors ASC ``S.vdiv`` lowering for FP32: fast math selects the
    hardware division, while the default selects CANN's 0-ULP FTZ-true
    division semantics.
    """
    if fast_math:
        return T.vmi.vdiv(src0, src1, mask)
    return fast_math_div_core(src0, src1, mask, size=size)


# Backward-compatible aliases for existing VMI call sites.
vdiv_with_fast_math = fast_math_div
