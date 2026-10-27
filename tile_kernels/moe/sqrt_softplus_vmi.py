from tilelang import language as T
from tilelang.ascend.language import vmi as V


_G0 = 0.8325546383857727
_FORWARD_COEFFS = (
    0.000471802573883906,
    -0.0004226923920214176,
    -8.536428504157811e-05,
    0.002192076062783599,
    0.000292900949716568,
    0.004986039828509092,
    0.011144610121846199,
    0.03287079185247421,
    0.11599518358707428,
)


@T.macro
def _horner_Pt_vmi(t, mask, size):
    poly = V.vbrc(T.float32(_FORWARD_COEFFS[0]), size=size)
    for coeff in _FORWARD_COEFFS[1:]:
        poly = V.vmula(V.vbrc(T.float32(coeff), size=size), poly, t, mask)
    return poly


@T.macro
def _vsqrt_rne_vmi(x, one, mask):
    root0 = V.vsqrt(x, mask)
    recip0 = V.vdiv(one, root0, mask, precision='ftz_true')
    root1 = V.vmuls(V.vadd(root0, V.vmul(x, recip0, mask), mask), T.float32(0.5), mask)
    resid = V.vmula(V.vneg(x, mask), root1, root1, mask)
    corr = V.vmuls(recip0, T.float32(-0.5), mask)
    root2 = V.vmula(root1, resid, corr, mask)
    return root2


@T.macro
def _assemble_sqrt_vmi(x, l, one, mask):
    s = V.vmula(V.vrelu(x, mask), l, l, mask)
    return _vsqrt_rne_vmi(s, one, mask)


@T.macro
def _exp_neg_half_abs_vmi(x, mask):
    neg_half_abs = V.vmuls(V.vabs(x), T.float32(-0.5), mask)
    return V.vexp(neg_half_abs, mask)


@T.macro
def _sqrt_log1p_vmi(a, one, mask, size):
    t = V.vsub(one, V.vmul(a, a, mask), mask)
    p = _horner_Pt_vmi(t, mask, size)
    t = V.vmula(V.vbrc(T.float32(_G0), size=size), t, p, mask)
    return V.vmul(a, t, mask)


@T.macro
def _assemble_softsqrt_vmi(x, l, one, mask):
    y_pos = _assemble_sqrt_vmi(x, l, one, mask)
    return V.vsel(V.vcmps(x, T.float32(0.0), mask, 'gt'), y_pos, l)


@T.macro
def sqrt_softplus_forward_vmi(logit, mask, size):
    one = V.vbrc(T.float32(1.0), size=size)
    a = _exp_neg_half_abs_vmi(logit, mask)
    l = _sqrt_log1p_vmi(a, one, mask, size)
    return _assemble_softsqrt_vmi(logit, l, one, mask)


_BACKWARD_COEFFS = (
    0.000129130945,
    -0.00130112655,
    0.00827621203,
    -0.041647464,
    0.166663632,
    -0.499999821,
)


@T.macro
def sqrt_softplus_backward_vmi(y, mask, size):
    """Recover dy/dx from y=sqrt(softplus(x)); use the polynomial only for y²<1."""
    one = V.vbrc(T.float32(1.0), size=size)
    u = V.vmul(y, y, mask)
    q = V.vbrc(T.float32(_BACKWARD_COEFFS[0]), size=size)
    for coeff in _BACKWARD_COEFFS[1:]:
        q = V.vadds(V.vmul(q, u, mask), T.float32(coeff), mask)
    half_y = V.vmuls(y, T.float32(0.5), mask)
    small = V.vmula(half_y, V.vmul(u, q, mask), half_y, mask)
    numerator = V.vsub(one, V.vexp(V.vneg(u, mask), mask), mask)
    large = V.vmuls(V.vdiv(numerator, y, mask, precision='ftz_true'), T.float32(0.5), mask)
    return V.vsel(V.vcmps(u, T.float32(1.0), mask, 'lt'), small, large)
