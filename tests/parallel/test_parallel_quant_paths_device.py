from test_parallel_device import NPU, rand, record
import pytest
import torch
from tilelang.ascend import language as T
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape
from tile_kernels.quant.norm_forward_parallel import get_norm_forward_kernel_parallel
from tile_kernels.quant.swiglu_backward_parallel import get_swiglu_backward_kernel_parallel


def empty(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=NPU)


@pytest.mark.parametrize('packed,colmajor,mega', [(False, False, False), (True, True, True)])
def test_norm_quant(packed, colmajor, mega):
    n, h = 32, 256
    x = rand(1, n, h)
    w = rand(h)
    config = CastOutputConfig(
        torch.float8_e4m3fn, (1, 32), round_sf=True, use_packed_ue8m0=packed, use_tma_aligned_col_major_sf=colmajor, clamp_min_value=1e-4
    )
    y = empty(1, n, h, dtype=config.torch_dtype)
    sf = empty(1, *get_sf_shape((n, h), config), dtype=config.sf_torch_dtype)
    rb = empty(1, n)
    bf = empty(1, n, h, dtype=torch.bfloat16)
    mega_out = torch.empty_strided((128, h // 128, 4), (4, 512, 1), device=NPU, dtype=torch.uint8)
    k = get_norm_forward_kernel_parallel(n, h, True, False, 1e-6, 0.7, T.bfloat16, config, 4, True, 128 if mega else 0)
    record(k, f'norm_quant_{packed}_{colmajor}_{mega}')
    k(x.to(NPU), y, sf, bf, mega_out, w.to(NPU), rb, x.to(NPU), bf)
    z = (x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (w.float() * 0.7)).bfloat16().float()
    a = z.reshape(n, h // 32, 32).abs().amax(-1).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(a / 448.0)))
    reference = (z.reshape(n, h // 32, 32) / scale[:, :, None]).reshape(1, n, h).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(y.cpu().float(), reference, rtol=0, atol=0)
    result = sf.cpu().squeeze(0)
    if packed:
        if colmajor:
            result = result.reshape(h // 64, n, 2).permute(1, 0, 2).reshape(n, h // 32)
        result = torch.exp2(result.float() - 127)
    torch.testing.assert_close(result, scale, rtol=0, atol=0)
    if mega:
        actual = mega_out.cpu()[torch.arange(n) * 4].reshape(n, h // 32)
        torch.testing.assert_close(torch.exp2(actual.float() - 127), scale, rtol=0, atol=0)


@pytest.mark.parametrize('packed,colmajor', [(False, False), (True, True)])
def test_swiglu_backward_quant(packed, colmajor):
    n, h = 32, 256
    xf = rand(n, 2 * h).to(torch.float8_e4m3fn).float()
    scales = torch.full((n, 2 * h // 32), 0.5)
    x = xf * scales.repeat_interleave(32, 1)
    g = rand(n, h)
    weights = rand(n, dtype=torch.float32).abs() + 0.1
    config = CastOutputConfig(
        torch.float8_e4m3fn, (1, 32), round_sf=True, use_packed_ue8m0=packed, use_tma_aligned_col_major_sf=colmajor, clamp_min_value=1e-4
    )
    quant = empty(n, 2 * h, dtype=torch.float8_e4m3fn)
    sf = empty(*get_sf_shape((n, 2 * h), config), dtype=config.sf_torch_dtype)
    dx = empty(n, 2 * h, dtype=torch.bfloat16)
    recomp = empty(n, h, dtype=torch.bfloat16)
    dw = empty(n)
    k = get_swiglu_backward_kernel_parallel(h, 0, 128, True, True, True, T.float8_e4m3fn, T.bfloat16, config, True, 4)
    record(k, f'swiglu_backward_quant_{packed}_{colmajor}')
    k(
        xf.to(torch.float8_e4m3fn).to(NPU),
        scales.to(NPU),
        g.to(NPU),
        weights.to(NPU),
        0.7,
        empty(0, dtype=torch.int32),
        recomp,
        quant,
        sf,
        dx,
        dw,
        1.2,
    )
    l, r = x[:, :h], x[:, h:]
    lc = l.clamp_max(1.2)
    rc = r.clamp(-1.2, 1.2)
    sig = 1 / (1 + torch.exp(-lc))
    gs = g.float() * (weights * 0.7)[:, None] * sig
    dl = gs * rc * (1 + lc * (1 - sig))
    dr = gs * lc
    dl = torch.where(l > 1.2, 0.0, dl)
    dr = torch.where(r.abs() > 1.2, 0.0, dr)
    grad = torch.cat((dl, dr), 1)
    torch.testing.assert_close(dx.cpu(), grad.bfloat16(), rtol=1e-2, atol=8e-3)
    scale = torch.exp2(torch.ceil(torch.log2(grad.reshape(n, 2 * h // 32, 32).abs().amax(-1).clamp_min(1e-4) / 448.0)))
    ref = (grad.reshape(n, 2 * h // 32, 32) / scale[:, :, None]).reshape(n, 2 * h).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(quant.cpu().float(), ref, rtol=0, atol=0)
    actual = sf.cpu()
    if packed:
        if colmajor:
            actual = actual.reshape(h // 32, n, 2).permute(1, 0, 2).reshape(n, 2 * h // 32)
        actual = torch.exp2(actual.float() - 127)
    torch.testing.assert_close(actual, scale, rtol=0, atol=0)
