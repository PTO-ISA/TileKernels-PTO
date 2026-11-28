"""Scalar counterparts of the ASC sqrt-softplus approximations."""

from tilelang.ascend import language as T


@T.macro
def sqrt_softplus_parallel(x):
    a = T.exp(-0.5 * T.abs(x))
    t = 1.0 - a * a
    p = T.alloc_var(T.float32, init=0.000471802573883906)
    p = p * t - 0.0004226923920214176
    p = p * t - 8.536428504157811e-05
    p = p * t + 0.002192076062783599
    p = p * t + 0.000292900949716568
    p = p * t + 0.004986039828509092
    p = p * t + 0.011144610121846199
    p = p * t + 0.03287079185247421
    p = p * t + 0.11599518358707428
    l = a * (0.8325546383857727 + t * p)
    return T.if_then_else(x > 0.0, T.sqrt(T.max(x, 0.0) + l * l), l)


@T.macro
def sqrt_softplus_backward_parallel(y):
    u = y * y
    q = T.alloc_var(T.float32, init=0.000129130945)
    q = q * u - 0.00130112655
    q = q * u + 0.00827621203
    q = q * u - 0.041647464
    q = q * u + 0.166663632
    q = q * u - 0.499999821
    return T.if_then_else(u < 1.0, (u * q) * (0.5 * y) + 0.5 * y, 0.5 * ((1.0 - T.exp(-u)) / y))
