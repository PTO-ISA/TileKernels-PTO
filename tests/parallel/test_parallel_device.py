"""Device acceptance; invoke only through task-submit --device auto.

Inputs and references are constructed on CPU so correctness does not depend on
another TileLang kernel. Every compiled kernel must contain generated PTO VMI.
"""

import os
from pathlib import Path

import pytest
import torch
import torch_npu

if 'TASK_DEVICE' not in os.environ:
    raise RuntimeError('Use task-submit --device auto to run this test file')
DEVICE = 0 if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') else int(os.environ['TASK_DEVICE'].split(',')[0])
torch.npu.set_device(DEVICE)
NPU = f'npu:{DEVICE}'

from tile_kernels.mhc.expand_parallel import expand_to_mhc_fwd_parallel, expand_to_mhc_bwd_parallel
from tile_kernels.mhc.post_parallel import mhc_post_fwd_parallel
from tile_kernels.mhc.pre_apply_mix_parallel import mhc_pre_apply_mix_fwd_parallel
from tile_kernels.quant.common import CastInputConfig, CastOutputConfig
from tile_kernels.quant.per_channel_cast_parallel import get_per_channel_cast_kernel_parallel
from tile_kernels.quant.cast_back_parallel import get_cast_back_kernel_parallel
from tile_kernels.moe.normalize_weight_parallel import get_normalize_weight_kernel_parallel
from tilelang.ascend import language as T


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(20261025)).to(dtype)


def record(kernel, name):
    src = kernel.get_kernel_source()
    assert 'pto.vmi.' in src, 'Kernel did not go through the PTO VMI backend'
    folder = Path(os.environ.get('PARALLEL_GENERATED_DIR', Path.cwd() / 'artifacts' / 'parallel-generated'))
    folder.mkdir(exist_ok=True, parents=True)
    (folder / f'{name}.py').write_text(src)


@pytest.mark.parametrize('n,h,block,stages', [(1, 128, 128, 1), (17, 1280, 256, 3), (129, 4224, 384, 7), (3, 1280, 384, 3)])
def test_expand(n, h, block, stages):
    m = 4
    x = rand(n, h)
    xn = x.to(NPU)
    out = torch.empty((n, m, h), dtype=torch.bfloat16, device=NPU)
    kw = dict(h_blk=block, num_stages=stages, num_vec_cores=4)
    k = expand_to_mhc_fwd_parallel.compile(xn, out, **kw)
    record(k, f'expand_fwd_{h}_{stages}')
    k(xn, out)
    torch.testing.assert_close(out.cpu(), x[:, None, :].expand(n, m, h), rtol=0, atol=0)
    grad = rand(n, m, h)
    gn = grad.to(NPU)
    dx = torch.empty_like(xn)
    k = expand_to_mhc_bwd_parallel.compile(gn, dx, **kw)
    record(k, f'expand_bwd_{h}_{stages}')
    k(gn, dx)
    torch.testing.assert_close(dx.cpu(), grad.float().sum(1).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize('n,h,block', [(1, 128, 128), (17, 1280, 256), (65, 4096, 1024), (3, 1280, 384)])
def test_mix_fwd(n, h, block):
    m = 4
    x = rand(n, h)
    r = rand(n, m, h)
    post = rand(n, m, dtype=torch.float32)
    comb = rand(n, m, m, dtype=torch.float32)
    xn, rn, pn, cn = [t.to(NPU) for t in (x, r, post, comb)]
    out = torch.empty_like(rn)
    k = mhc_post_fwd_parallel.compile(cn, rn, pn, xn, out, h_blk=block, num_vec_cores=4)
    record(k, f'post_fwd_{h}')
    k(cn, rn, pn, xn, out)
    expected = x.float()[:, None, :] * post[:, :, None]
    for mi in range(m):
        expected += r.float()[:, mi, None, :] * comb[:, mi, :, None]
    torch.testing.assert_close(out.cpu(), expected.bfloat16())
    mixout = torch.empty_like(xn)
    k = mhc_pre_apply_mix_fwd_parallel.compile(rn, pn, mixout, h_blk=block, num_vec_cores=4)
    record(k, f'pre_apply_mix_fwd_{h}')
    k(rn, pn, mixout)
    expected = r[:, 0, :].float() * post[:, 0, None]
    for mi in range(1, m):
        expected += r[:, mi, :].float() * post[:, mi, None]
    torch.testing.assert_close(mixout.cpu(), expected.bfloat16())


@pytest.mark.parametrize('n,h', [(128, 128), (256, 512), (384, 1280)])
def test_per_channel(n, h):
    x = rand(n, h)
    xn = x.to(NPU)
    output = torch.empty((n, h), dtype=torch.float8_e4m3fn, device=NPU)
    sf = torch.empty((n // 32, h), device=NPU)
    # The retained ASC signature declares the unused input-scale tensor with
    # sf_block=(1,1), so the lazy adapter still checks its full logical shape.
    dummy = torch.empty((n, h), device=NPU)
    inc = CastInputConfig(torch.bfloat16)
    outc = CastOutputConfig(torch.float8_e4m3fn, (32, 1), clamp_min_value=1e-4)
    k = get_per_channel_cast_kernel_parallel(h, inc, outc, 4)
    record(k, f'per_channel_{h}')
    k(xn, output, sf, dummy)
    amax = x.float().reshape(n // 32, 32, h).abs().amax(1).clamp_min(1e-4)
    # Tensor/tensor division avoids PyTorch's scalar reciprocal/multiply
    # rewrite (also required by the product's torch/cast.py reference).
    maximum = torch.full_like(amax, 448.0)
    expected = (x.float().reshape(n // 32, 32, h) * (maximum / amax)[:, None, :]).reshape(n, h).to(torch.float8_e4m3fn)
    torch.testing.assert_close(output.cpu().float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(sf.cpu(), amax / maximum, rtol=0, atol=0)


@pytest.mark.parametrize('n,h', [(32, 128), (128, 512), (256, 1280)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cast_back(n, h, dtype):
    x = (rand(n, h, dtype=torch.float32) * 3).to(torch.float8_e4m3fn)
    scales = rand(n // 32, h, dtype=torch.float32).abs() + 0.01
    xn, sn = x.to(NPU), scales.to(NPU)
    out = torch.empty((n, h), device=NPU, dtype=dtype)
    inc = CastInputConfig(torch.float8_e4m3fn, (32, 1))
    k = get_cast_back_kernel_parallel(h, inc, T.dtype(dtype), 4)
    record(k, f'cast_back_{h}_{dtype}')
    k(xn, sn, out)
    expected = (x.float().reshape(n // 32, 32, h) * scales[:, None, :]).reshape(n, h).to(dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('n,topk', [(1, 1), (17, 4), (129, 8), (257, 32)])
def test_normalize_weight(n, topk):
    weights = rand(n, topk, dtype=torch.float32).abs()
    weights[0] = 0  # epsilon must make the zero row well-defined.
    wn = weights.to(NPU)
    denom = torch.empty(n, device=NPU)
    out = torch.empty_like(wn)
    k = get_normalize_weight_kernel_parallel(topk, 4)
    record(k, f'normalize_weight_{topk}')
    k(wn, denom, out)
    expected = weights.sum(-1) + 1e-20
    torch.testing.assert_close(denom.cpu(), expected, rtol=2e-6, atol=0)
    torch.testing.assert_close(out.cpu(), weights / expected[:, None], rtol=2e-6, atol=0)
