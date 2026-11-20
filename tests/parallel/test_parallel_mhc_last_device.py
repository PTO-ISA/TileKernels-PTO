from test_parallel_device import NPU, rand, record
import pytest
import torch
from tilelang.ascend import language as T
from tile_kernels.mhc.pre_split_mixes_parallel import (
    mhc_pre_split_mixes_fwd_simd_mhc4_parallel,
    mhc_pre_split_mixes_bwd_mhc4_parallel,
    mhc_pre_split_mixes_fwd_parallel,
    mhc_pre_split_mixes_bwd_parallel,
)
from tile_kernels.mhc.pre_big_fuse_parallel import mhc_pre_big_fuse_fwd_parallel


def empty(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=NPU)


def splits(x, s, b):
    return (torch.sigmoid(x[:, :4] * s[0] + b[:4]) + 1e-6, 2 * torch.sigmoid(x[:, 4:8] * s[1] + b[4:8]), x[:, 8:] * s[2] + b[8:])


@pytest.mark.parametrize('n,simd', [(1, True), (17, True), (65, True), (17, False)])
def test_split(n, simd):
    x, s, b = [rand(*shape, dtype=torch.float32).requires_grad_() for shape in [(n, 24), (3,), (24,)]]
    gs = [rand(n, k, dtype=torch.float32) for k in [4, 4, 16]]
    ref = splits(x, s, b)
    grads = torch.autograd.grad(ref, (x, s, b), gs)
    xn, sn, bn = [v.detach().to(NPU) for v in (x, s, b)]
    pre, post, comb = [empty(n, k) for k in [4, 4, 16]]
    f = mhc_pre_split_mixes_fwd_simd_mhc4_parallel if simd else mhc_pre_split_mixes_fwd_parallel
    kwargs = dict(mhc_post_mult_value=2.0, mhc_pre_eps=1e-6, token_block_size=8, num_stages=3, num_vec_cores=4)
    if not simd:
        kwargs['dtype'] = T.float32
    k = f.compile(xn, sn, bn, pre, post, comb, **kwargs)
    if simd:
        record(k, f'split_fwd_{n}')
    k(xn, sn, bn, pre, post, comb)
    for actual, expected in zip((pre, post, comb), ref):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-6, atol=3e-7)
    dx, ds, db = empty(n, 24), empty(4, 3), empty(4, 24)
    gn = [g.to(NPU) for g in gs]
    f = mhc_pre_split_mixes_bwd_mhc4_parallel if simd else mhc_pre_split_mixes_bwd_parallel
    kwargs = dict(mhc_post_mult_value=2.0, token_block_size=8, num_vec_cores=4)
    if simd:
        kwargs['num_stages'] = 3
    else:
        kwargs['dtype'] = T.float32
    k = f.compile(*gn, xn, post, sn, bn, dx, ds, db, **kwargs)
    if simd:
        record(k, f'split_bwd_{n}')
    k(*gn, xn, post, sn, bn, dx, ds, db)
    for actual, expected in zip((dx.cpu(), ds.cpu().sum(0), db.cpu().sum(0)), grads):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-5)


@pytest.mark.parametrize('n,h,repeat', [(1, 128, 1), (17, 512, 3), (33, 1280, 10)])
def test_big_fuse(n, h, repeat):
    mul = rand(1, n, 24, dtype=torch.float32)
    sq = rand(1, n, dtype=torch.float32).abs() + 1
    s, b = rand(3, dtype=torch.float32), rand(24, dtype=torch.float32)
    x = rand(n, 4, h)
    tensors = [v.to(NPU) for v in (mul, sq, s, b, x)]
    post, comb, out = empty(n, 4), empty(n, 16), empty(n, h, dtype=torch.bfloat16)
    k = mhc_pre_big_fuse_fwd_parallel.compile(
        *tensors,
        post,
        comb,
        out,
        rms_eps=1e-6,
        mhc_pre_eps=1e-6,
        mhc_sinkhorn_eps=1e-6,
        mhc_post_mult_value=2.0,
        sinkhorn_repeat=repeat,
        h_blk=128,
        n_splits=1,
        token_block=4,
        num_stages=3,
        num_vec_cores=4,
    )
    record(k, f'big_fuse_{n}_{h}_{repeat}')
    k(*tensors, post, comb, out)
    mix = mul[0] * torch.rsqrt(sq[0] / (4 * h) + 1e-6)[:, None]
    pre, p, c = splits(mix, s, b)
    c = c.reshape(n, 4, 4).softmax(-1) + 1e-6
    c = c / (c.sum(-2, keepdim=True) + 1e-6)
    for _ in range(repeat - 1):
        c = c / (c.sum(-1, keepdim=True) + 1e-6)
        c = c / (c.sum(-2, keepdim=True) + 1e-6)
    result = x[:, 0, :].float() * pre[:, 0, None]
    for j in range(1, 4):
        result += x[:, j, :].float() * pre[:, j, None]
    torch.testing.assert_close(post.cpu(), p, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(comb.cpu(), c.reshape(n, 16), rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(out.cpu(), result.bfloat16(), rtol=1e-2, atol=8e-3)
