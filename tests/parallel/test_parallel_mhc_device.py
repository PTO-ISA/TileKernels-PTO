"""Additional mHC acceptance; run under task-submit, like the main matrix."""

from test_parallel_device import NPU, rand, record
import pytest
import torch
from tile_kernels.mhc.norm_fn_parallel import (
    mhc_fn_normw_merge_fwd_parallel,
    mhc_fn_normw_merge_bwd_parallel,
    mhc_reduce_partials_and_rmsnorm_fwd_parallel,
)
from tile_kernels.mhc.multilayer_recompute_parallel import mhc_multilayer_recompute_parallel


@pytest.mark.parametrize('h,block,stages', [(128, 128, 1), (1280, 256, 3), (4096, 128, 5)])
def test_merge(h, block, stages):
    m = 24
    fn, normw, grad, df, dw = [rand(*shape, dtype=torch.float32) for shape in [(m, h), (h,), (m, h), (m, h), (h,)]]
    fnn, wn, gn, dfn, dwn = [t.to(NPU) for t in (fn, normw, grad, df, dw)]
    out = torch.empty_like(fnn)
    kw = dict(mhc_mult3=m, hidden_block=block, num_stages=stages, num_vec_cores=4)
    k = mhc_fn_normw_merge_fwd_parallel.compile(fnn, wn, out, **kw)
    record(k, f'merge_fwd_{h}_{stages}')
    k(fnn, wn, out)
    torch.testing.assert_close(out.cpu(), fn * normw, rtol=0, atol=0)
    k = mhc_fn_normw_merge_bwd_parallel.compile(fnn, wn, gn, dfn, dwn, **kw)
    record(k, f'merge_bwd_{h}_{stages}')
    k(fnn, wn, gn, dfn, dwn)
    torch.testing.assert_close(dfn.cpu(), df + grad * normw)
    expected = dw.clone()
    for row in range(m):
        expected += grad[row] * fn[row]
    torch.testing.assert_close(dwn.cpu(), expected)


@pytest.mark.parametrize('n,splits', [(1, 1), (17, 2), (33, 32)])
def test_reduce_partials(n, splits):
    mul = rand(splits, n, 1, 24, dtype=torch.float32)
    sq = rand(splits, n, 1, dtype=torch.float32).abs() + 0.1
    mn, sn = mul.to(NPU), sq.to(NPU)
    mout = torch.empty((n, 1, 24), device=NPU)
    sout = torch.empty((n, 1), device=NPU)
    out = torch.empty((n, 24), device=NPU)
    k = mhc_reduce_partials_and_rmsnorm_fwd_parallel.compile(
        mn, sn, mout, sout, out, rms_group_size=128, rms_eps=1e-6, n_splits=splits, num_vec_cores=4
    )
    record(k, f'reduce_partials_{n}_{splits}')
    k(mn, sn, mout, sout, out)
    mref, sref = mul.sum(0), sq.sum(0)
    torch.testing.assert_close(mout.cpu(), mref)
    torch.testing.assert_close(sout.cpu(), sref)
    torch.testing.assert_close(out.cpu(), mref[:, 0, :] * torch.rsqrt(sref / 128 + 1e-6))


@pytest.mark.parametrize('n,h,layers,post_layers', [(1, 128, 1, 0), (17, 1280, 3, 2), (5, 256, 2, 2)])
def test_multilayer(n, h, layers, post_layers):
    m = 4
    r = rand(n, m, h)
    pre = [rand(n, m, dtype=torch.float32) / (i + 2) for i in range(layers)]
    x = [rand(n, h) / (i + 2) for i in range(post_layers)]
    post = [rand(n, m, dtype=torch.float32) / (i + 2) for i in range(post_layers)]
    comb = [rand(n, m, m, dtype=torch.float32) / (i + 2) for i in range(post_layers)]
    pre_n, x_n, post_n, comb_n = [[t.to(NPU) for t in group] for group in [pre, x, post, comb]]
    inputs = [torch.empty((n, h), device=NPU, dtype=torch.bfloat16) for _ in range(layers)]
    residuals = [torch.empty((n, m, h), device=NPU, dtype=torch.bfloat16) for _ in range(post_layers)]
    ptrs = [torch.tensor([t.data_ptr() for t in group], dtype=torch.int64, device=NPU) for group in [pre_n, x_n, post_n, comb_n, inputs, residuals]]
    rn = r.to(NPU)
    k = mhc_multilayer_recompute_parallel.compile(rn, *ptrs, h_blk=256 if h > 128 else 128, num_vec_cores=4)
    record(k, f'multilayer_{n}_{h}_{layers}_{post_layers}')
    k(rn, *ptrs)
    for layer in range(layers):
        expected_input = r[:, 0, :].float() * pre[layer][:, 0, None]
        for mi in range(1, m):
            expected_input += r[:, mi, :].float() * pre[layer][:, mi, None]
        torch.testing.assert_close(inputs[layer].cpu(), expected_input.bfloat16())
        if layer < post_layers:
            rr = x[layer].float()[:, None, :] * post[layer][:, :, None]
            for mi in range(m):
                rr += r[:, mi, None, :].float() * comb[layer][:, mi, :, None]
            r = rr.bfloat16()
            torch.testing.assert_close(residuals[layer].cpu(), r)
