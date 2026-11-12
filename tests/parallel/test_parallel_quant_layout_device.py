from test_parallel_device import NPU, rand, record
import pytest
import torch
from tilelang.ascend import language as T
from tile_kernels.quant.common import CastOutputConfig, CastInputConfig, get_sf_shape
from tile_kernels.quant.per_token_cast_parallel import get_per_token_cast_kernel_parallel
from tile_kernels.quant.per_block_cast_parallel import get_per_block_cast_kernel_parallel
from tile_kernels.torch.cast import convert_to_e2m1_bits


def allocate_sf(n, h, config):
    shape = tuple(get_sf_shape((n, h), config))
    return torch.empty(shape, dtype=config.sf_torch_dtype, device=NPU)


def unpack_sf(sf, n, h, config):
    exp = sf.cpu()
    if config.use_tma_aligned_col_major_sf:
        exp = (
            exp.reshape(h // config.sf_block[1] // 2, n // config.sf_block[0], 2)
            .permute(1, 0, 2)
            .reshape(n // config.sf_block[0], h // config.sf_block[1])
        )
    return torch.exp2(exp.float() - 127)


@pytest.mark.parametrize('block,fp4,colmajor', [(1, False, False), (1, False, True), (32, False, True), (32, True, False), (32, True, True)])
def test_packed_scale_layout(block, fp4, colmajor):
    n, h = 64, 512
    x = rand(n, h)
    config = CastOutputConfig(
        torch.int8 if fp4 else torch.float8_e4m3fn,
        (block, 32),
        use_packed_ue8m0=True,
        use_tma_aligned_col_major_sf=colmajor,
        round_sf=True,
        clamp_min_value=1e-4,
    )
    out = torch.empty((n, h // 2 if fp4 else h), dtype=config.torch_dtype, device=NPU)
    sf = allocate_sf(n, h, config)
    if block == 1:
        kernel = get_per_token_cast_kernel_parallel(h, h, CastInputConfig(torch.bfloat16), config, False, False, False, 4)
    else:
        kernel = get_per_block_cast_kernel_parallel(h, CastInputConfig(torch.bfloat16), config, False, False, 4)
    record(kernel, f'packed_{block}_{fp4}_{colmajor}')
    if block == 1:
        kernel(x.to(NPU), torch.empty((1, 1), device=NPU), out, sf)
    else:
        kernel(x.to(NPU), out, sf)
    values = x.float().reshape(n // block, block, h // 32, 32)
    amax = values.abs().amax((1, 3)).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / (6.0 if fp4 else 448.0))))
    scaled = (values / scale[:, None, :, None]).bfloat16().float().reshape(n, h)
    torch.testing.assert_close(unpack_sf(sf, n, h, config), scale, rtol=0, atol=0)
    if fp4:
        bits = convert_to_e2m1_bits(scaled, 6.0, scaled.device)
        expected = (bits[:, 0::2] | (bits[:, 1::2] << 4)).view(torch.int8)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out.cpu().float(), scaled.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
