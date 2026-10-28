from test_parallel_device import NPU, rand, record
import pytest
import torch
from tilelang.ascend import language as T
from tile_kernels.quant.common import CastOutputConfig, CastInputConfig
from tile_kernels.quant.cast_parallel import get_cast_kernel_parallel
from tile_kernels.quant.norm_forward_parallel import get_norm_forward_kernel_parallel
from tile_kernels.quant.norm_backward_parallel import get_norm_backward_kernel_parallel
from tile_kernels.quant.per_token_cast_and_cast_back_parallel import get_per_token_cast_and_cast_back_kernel_parallel
from tile_kernels.quant.swiglu_forward_parallel import get_swiglu_forward_kernel_parallel
from tile_kernels.quant.swiglu_backward_parallel import get_swiglu_backward_kernel_parallel
from tile_kernels.quant.per_block_cast_lossless_parallel import get_per_block_cast_lossless_kernel_parallel


def empty(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=NPU)


def hash_round(x):
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    group = bits.reshape(-1, 4)
    h = (group * torch.tensor([0x5671D42B, 0x9995E499, 0xACE1B8A5, 0xE153538D])).sum(1) + torch.arange(group.shape[0])
    h = h & 0xFFFFFFFF
    h = (h ^ (h >> 23)) * 0x7FEB352D & 0xFFFFFFFF
    h = h ^ (h >> 16)
    r1 = h * 0x846CA68B & 0xFFFFFFFF
    r1 = r1 ^ (r1 >> 11)
    r2 = h * 0xD35A2D97 & 0xFFFFFFFF
    r2 = r2 ^ (r2 >> 11)
    r = torch.stack((r1 & 65535, r1 >> 16, r2 & 65535, r2 >> 16), 1).reshape(-1)
    high = (bits >> 16) + (r < (bits & 65535)).long()
    return high.to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize('n,stochastic', [(256, False), (65536, True), (262144, True)])
def test_cast(n, stochastic):
    x = rand(n, dtype=torch.float32)
    y = empty(n, dtype=torch.bfloat16)
    k = get_cast_kernel_parallel(T.float32, T.bfloat16, stochastic, 4, 4)
    record(k, f'cast_{n}_{stochastic}')
    k(x.to(NPU), y)
    torch.testing.assert_close(y.cpu(), hash_round(x) if stochastic else x.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize('n,h,rounded', [(1, 128, True), (33, 512, False), (65, 1280, True)])
def test_roundtrip(n, h, rounded):
    x = rand(n, h)
    out = x.to(NPU)
    config = CastOutputConfig(torch.float8_e4m3fn, (1, 32), round_sf=rounded, clamp_min_value=1e-4)
    k = get_per_token_cast_and_cast_back_kernel_parallel(h, h, T.bfloat16, config, 4)
    record(k, f'roundtrip_{n}_{h}_{rounded}')
    k(out)
    amax = x.float().reshape(n, h // 32, 32).abs().amax(-1).clamp_min(1e-4)
    if rounded:
        bits = amax.view(torch.int32)
        inv_bits = 0x83000000 - ((bits.to(torch.int64) + 0x1FFFFF) & 0x7F800000)
        inv = inv_bits.to(torch.int32).view(torch.float32).bfloat16().float()
        sf = 1.0 / inv
    else:
        maximum = torch.full_like(amax, 448.0)
        inv = maximum / amax
        sf = amax / maximum
    scaled = x.float().reshape(n, h // 32, 32) * inv[:, :, None]
    if rounded:
        scaled = scaled.bfloat16().float()
    expected = (scaled.to(torch.float8_e4m3fn).float() * sf[:, :, None]).reshape(n, h).bfloat16()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('n,h,weighted,residual', [(1, 128, False, False), (17, 512, True, True), (65, 1280, True, False)])
def test_norm(n, h, weighted, residual):
    x, r, w = rand(1, n, h), rand(1, n, h), rand(h)
    xn, rn, wn = x.to(NPU), r.to(NPU), w.to(NPU)
    y, rb = empty(1, n, h, dtype=torch.bfloat16), empty(1, n)
    rout = empty(1, n, h, dtype=torch.bfloat16)
    dummy = empty(1, 1, 1)
    mega = torch.empty_strided((1, h // 128, 4), (4, 4, 1), device=NPU, dtype=torch.uint8)
    k = get_norm_forward_kernel_parallel(n, h, weighted, residual, 1e-6, 0.7, T.bfloat16, CastOutputConfig(torch.bfloat16), 4)
    record(k, f'norm_fwd_{n}_{h}')
    k(xn, y, dummy, empty(1, n, h, dtype=torch.bfloat16), mega, wn, rb, rn, rout)
    z = (x.float() + r.float()).bfloat16().float() if residual else x.float()
    rstd = (z.square().mean(-1) + 1e-6).rsqrt()
    expected = z * rstd[:, :, None]
    expected = expected * (w.float() * 0.7) if weighted else expected * 0.7
    torch.testing.assert_close(y.cpu(), expected.bfloat16(), rtol=1e-2, atol=8e-3)
    torch.testing.assert_close(rb.cpu(), rstd, rtol=2e-6, atol=2e-6)
    if residual:
        torch.testing.assert_close(rout.cpu(), z.bfloat16(), rtol=0, atol=0)
    dy = rand(n, h)
    dr = rand(n, h)
    out, partial, dw = empty(n, h, dtype=torch.bfloat16), empty(4, h), empty(h)
    k = get_norm_backward_kernel_parallel(n, h, weighted, residual, False, 0.7, T.bfloat16, T.float32, 4)
    record(k, f'norm_bwd_{n}_{h}')
    k(z.squeeze(0).bfloat16().to(NPU), wn, dy.to(NPU), rb.reshape(n), out, partial, dr.to(NPU), dw)
    xhat = z.squeeze(0) * rstd.reshape(n, 1)
    grad = dy.float() * 0.7 * (w.float() if weighted else 1.0)
    dx = (grad - xhat * (grad * xhat).mean(-1, keepdim=True)) * rstd.reshape(n, 1)
    if residual:
        dx = dx + dr.float()
    torch.testing.assert_close(out.cpu(), dx.bfloat16(), rtol=1e-2, atol=8e-3)
    if weighted:
        torch.testing.assert_close(dw.cpu(), (xhat * (dy.float() * 0.7)).sum(0), rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize('n,h,clamp,quant', [(3, 128, False, False), (17, 512, True, False), (32, 256, True, True)])
def test_swiglu(n, h, clamp, quant):
    x = rand(n, 2 * h)
    weight = rand(n, dtype=torch.float32).abs() + 0.1
    xn, wn = x.to(NPU), weight.to(NPU)
    oc = CastOutputConfig(torch.float8_e4m3fn, (1, 32), round_sf=True, clamp_min_value=1e-4) if quant else CastOutputConfig(torch.bfloat16)
    out = empty(n, h, dtype=oc.torch_dtype)
    sf = empty(n, h // 32) if quant else empty(1, 1)
    counts = torch.zeros(4, dtype=torch.int64, device=NPU)
    psum = empty(0, dtype=torch.int32)
    k = get_swiglu_forward_kernel_parallel(h, 0, 128, True, True, 1.2 if clamp else None, clamp, T.bfloat16, oc, 4)
    record(k, f'swiglu_fwd_{n}_{h}_{quant}')
    k(xn, out, sf, psum, wn, 0.7, counts)
    l, r = x[:, :h].float(), x[:, h:].float()
    lc = l.clamp_max(1.2) if clamp else l
    rc = r.clamp(-1.2, 1.2) if clamp else r
    act = (lc / (1.0 + torch.exp(-lc))) * rc
    expected = act * (weight * 0.7)[:, None]
    if quant:
        maximum = expected.reshape(n, h // 32, 32).abs().amax(-1).clamp_min(1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(maximum / 448.0)))
        expected = (expected.reshape(n, h // 32, 32) / scale[:, :, None]).reshape(n, h).to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(sf.cpu(), scale, rtol=0, atol=0)
        torch.testing.assert_close(out.cpu().float(), expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), rtol=1e-2, atol=8e-3)
    if clamp:
        expected_counts = torch.tensor([(l > 1.2).sum(), (r > 1.2).sum(), (r < -1.2).sum(), n * h], dtype=torch.int64)
        torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    # Backward independently exercises recomputation, weights and clamp masks.
    g = rand(n, h)
    dx = empty(n, 2 * h, dtype=torch.bfloat16)
    dw = empty(n)
    recomp = empty(n, h, dtype=torch.bfloat16)
    bc = CastOutputConfig(torch.bfloat16)
    k = get_swiglu_backward_kernel_parallel(h, 0, 128, True, True, clamp, T.bfloat16, T.bfloat16, bc, True, 4)
    record(k, f'swiglu_bwd_{n}_{h}')
    k(xn, empty(1, 1), g.to(NPU), wn, 0.7, psum, recomp, empty(1, 1, dtype=torch.bfloat16), empty(1, 1), dx, dw, 1.2)
    sig = 1.0 / (1.0 + torch.exp(-lc))
    gs = (g.float() * (weight * 0.7)[:, None]) * sig
    dl = (gs * rc) * (1.0 + lc * (1.0 - sig))
    dr = gs * lc
    if clamp:
        dl = torch.where(l > 1.2, 0.0, dl)
        dr = torch.where(r.abs() > 1.2, 0.0, dr)
    torch.testing.assert_close(dx.cpu(), torch.cat((dl, dr), 1).bfloat16(), rtol=1e-2, atol=8e-3)
    torch.testing.assert_close(dw.cpu(), (g.float() * act).sum(-1) * 0.7, rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(recomp.cpu(), (act * (weight * 0.7)[:, None]).bfloat16(), rtol=1e-2, atol=8e-3)


@pytest.mark.parametrize('n,h', [(32, 256), (64, 512)])
def test_lossless(n, h):
    # Every FP4 code and a bounded per-group power-of-two scale spread.
    codes = (torch.arange(n * h).reshape(n, h) % 16).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
    exp = (torch.arange(n * h // 32).reshape(n, h // 32) % 4) - 2
    scales = torch.exp2(exp.float())
    inc = CastInputConfig(torch.int8, (1, 32))
    oc = CastOutputConfig(torch.float8_e4m3fn, (32, 32), round_sf=True)
    out = empty(n, h, dtype=torch.float8_e4m3fn)
    sf = empty(n // 32, h // 32)
    k = get_per_block_cast_lossless_kernel_parallel(h, inc, oc, 4)
    record(k, f'lossless_{n}_{h}')
    k(packed.to(NPU), scales.to(NPU), out, sf)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    values = table[codes.long()] * scales.repeat_interleave(32, 1)
    expected_sf = scales.reshape(n // 32, 32, h // 32).amax(1) / 64.0
    expected = (values.reshape(n // 32, 32, h // 32, 32) / expected_sf[:, None, :, None]).reshape(n, h).to(torch.float8_e4m3fn)
    torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
