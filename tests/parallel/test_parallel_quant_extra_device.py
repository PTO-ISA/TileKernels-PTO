from test_parallel_device import NPU, rand, record
import pytest
import torch
import tilelang
from probe_features import fp4_cast
from tile_kernels.torch.cast import convert_to_e2m1_bits
from tile_kernels.quant.per_block_cast_parallel import get_per_block_cast_kernel_parallel
from tile_kernels.quant.common import CastInputConfig, CastOutputConfig


def test_fp4_midpoints():
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    x = torch.cat((mids, torch.nextafter(mids, torch.full_like(mids, float('inf'))), torch.nextafter(mids, torch.zeros_like(mids))))
    x = torch.cat((x, -x, torch.tensor([0.0, -0.0, 6.0, -6.0, 10.0, -10.0])))
    x = x.repeat(3)[:128].contiguous()
    out = torch.empty((64,), dtype=torch.int8, device=NPU)
    kernel = tilelang.compile(fp4_cast(), target='pto')
    record(kernel, 'fp4_midpoints')
    kernel(x.to(NPU), out)
    bits = convert_to_e2m1_bits(x, 6.0, x.device).reshape(64, 2)
    expected = (bits[:, 0] | (bits[:, 1] << 4)).view(torch.int8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('n,h,dtype,rounded', [(32, 128, torch.bfloat16, False), (96, 512, torch.float32, True)])
def test_per_block(n, h, dtype, rounded):
    x = rand(n, h, dtype=dtype)
    inc = CastInputConfig(dtype)
    outc = CastOutputConfig(torch.float8_e4m3fn, (32, 32), round_sf=rounded, clamp_min_value=1e-4)
    out = torch.empty((n, h), dtype=torch.float8_e4m3fn, device=NPU)
    sf = torch.empty((n // 32, h // 32), device=NPU)
    kernel = get_per_block_cast_kernel_parallel(h, inc, outc, False, False, 4)
    record(kernel, f'per_block_{n}_{h}_{rounded}')
    kernel(x.to(NPU), out, sf)
    xf = x.float().reshape(n // 32, 32, h // 32, 32)
    amax = xf.abs().amax((1, 3)).clamp_min(1e-4)
    maximum = torch.full_like(amax, 448.0)
    expected_sf = amax / maximum
    if rounded:
        expected_sf = torch.exp2(torch.ceil(torch.log2(expected_sf)))
        inverse = 1.0 / expected_sf
    else:
        inverse = maximum / amax
    scaled = xf * inverse[:, None, :, None]
    if dtype == torch.bfloat16 and rounded:
        scaled = scaled.bfloat16().float()
    torch.testing.assert_close(out.cpu().float(), scaled.reshape(n, h).to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
