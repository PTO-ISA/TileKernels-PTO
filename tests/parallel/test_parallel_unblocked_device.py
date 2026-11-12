"""Acceptance for the five paths previously blocked by missing lowering."""

import pytest
import torch
from test_parallel_device import NPU, rand, record
from tile_kernels.quant.common import CastInputConfig, CastOutputConfig
from tile_kernels.quant.per_token_cast_parallel import get_per_token_cast_kernel_parallel
from tile_kernels.mhc.post_parallel import mhc_post_bwd_parallel
from tile_kernels.mhc.pre_apply_mix_parallel import mhc_pre_apply_mix_bwd_parallel
from tile_kernels.mhc.norm_fn_parallel import mhc_reduce_partials_and_rmsnorm_bwd_parallel
from tile_kernels.moe.topk_gate_parallel import get_topk_gate_kernel_parallel


@pytest.mark.parametrize('n,h,stages', [(3, 128, 1), (17, 512, 3), (33, 1280, 7)])
def test_mhc_backward(n, h, stages):
    g, r, x = rand(n, 4, h), rand(n, 4, h), rand(n, h)
    c, p = rand(n, 4, 4, dtype=torch.float32), rand(n, 4, dtype=torch.float32)
    args = [v.to(NPU) for v in (g, c, r, p, x)]
    kernel = mhc_post_bwd_parallel.compile(*args, h_blk=128, num_stages=stages, num_vec_cores=4)
    record(kernel, f'post_bwd_{n}_{h}_{stages}')
    dc, dr, dp, dx = kernel(*args)
    torch.testing.assert_close(dc.cpu(), torch.einsum('noh,nih->nio', g.float(), r.float()), rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(dp.cpu(), (g.float() * x.float()[:, None, :]).sum(-1), rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(dr.cpu(), torch.einsum('noh,nio->nih', g.float(), c).bfloat16())
    torch.testing.assert_close(dx.cpu(), (g.float() * p[:, :, None]).sum(1).bfloat16())
    initial = rand(n, 4, h)
    xgrad = initial.to(NPU)
    args = [x.to(NPU), r.to(NPU), p.to(NPU), xgrad]
    kernel = mhc_pre_apply_mix_bwd_parallel.compile(*args, h_blk=128, num_stages=stages, num_vec_cores=4)
    record(kernel, f'pre_apply_mix_bwd_{n}_{h}_{stages}')
    mixgrad = kernel(*args)
    torch.testing.assert_close(mixgrad.cpu(), (x.float()[:, None, :] * r.float()).sum(-1), rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(xgrad.cpu(), (initial.float() + x.float()[:, None, :] * p[:, :, None]).bfloat16())


@pytest.mark.parametrize('n', [1, 17, 65])
def test_rmsnorm_backward(n):
    g = rand(n, 24, dtype=torch.float32)
    mul = rand(n, 1, 24, dtype=torch.float32)
    sq = rand(n, 1, dtype=torch.float32).abs() + 100
    args = [v.to(NPU) for v in (g, mul, sq)]
    dmul, dsq = torch.empty_like(args[1]), torch.empty_like(args[2])
    kernel = mhc_reduce_partials_and_rmsnorm_bwd_parallel.compile(*args, dmul, dsq, rms_group_size=128, rms_eps=1e-6, num_vec_cores=4)
    record(kernel, f'rmsnorm_bwd_{n}')
    kernel(*args, dmul, dsq)
    rms = (sq / 128 + 1e-6).rsqrt()
    torch.testing.assert_close(dmul.cpu(), g[:, None, :] * rms[:, :, None], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(dsq.cpu(), (g[:, None, :] * mul).sum(-1) * rms / (sq + 128e-6) * -0.5, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize('n,e,k', [(1, 1, 1), (17, 35, 9), (33, 128, 8), (3, 1025, 32)])
def test_topk(n, e, k):
    x = rand(n, e, dtype=torch.float32)
    x[0] = 1  # Stable ties: lowest expert index first.
    out = torch.empty((n, k), dtype=torch.int64, device=NPU)
    kernel = get_topk_gate_kernel_parallel(e, k, 4)
    record(kernel, f'topk_{n}_{e}_{k}')
    kernel(x.to(NPU), out)
    torch.testing.assert_close(out.cpu(), torch.argsort(x, dim=1, descending=True, stable=True)[:, :k], rtol=0, atol=0)


@pytest.mark.parametrize('n,h,rounded', [(1, 128, False), (33, 512, False), (65, 1280, True)])
def test_per_token(n, h, rounded):
    x = rand(n, h)
    out = torch.empty((n, h), dtype=torch.float8_e4m3fn, device=NPU)
    sf = torch.empty((n, h // 32), device=NPU)
    inc = CastInputConfig(torch.bfloat16)
    outc = CastOutputConfig(torch.float8_e4m3fn, (1, 32), round_sf=rounded, clamp_min_value=1e-4)
    kernel = get_per_token_cast_kernel_parallel(h, h, inc, outc, False, False, False, 4)
    record(kernel, f'per_token_{n}_{h}_{rounded}')
    kernel(x.to(NPU), torch.empty((1, 1), device=NPU), out, sf)
    amax = x.float().reshape(n, h // 32, 32).abs().amax(-1).clamp_min(1e-4)
    maximum = torch.full_like(amax, 448.0)
    expected_sf = amax / maximum
    if rounded:
        expected_sf = torch.exp2(torch.ceil(torch.log2(expected_sf)))
        inverse = 1.0 / expected_sf
    else:
        inverse = maximum / amax
    expected = (x.float().reshape(n, h // 32, 32) * inverse[:, :, None]).reshape(n, h).to(torch.float8_e4m3fn)
    torch.testing.assert_close(out.cpu().float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(sf.cpu(), expected_sf, rtol=0, atol=0)
