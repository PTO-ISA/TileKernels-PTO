"""Head mix forward/backward acceptance; TASK_DEVICE guard is imported first."""

from test_parallel_device import NPU, rand, record
import pytest
import torch
from tile_kernels.mhc.head_compute_mix_parallel import (
    mhc_head_compute_mix_fwd_parallel,
    mhc_head_compute_mix_bwd_parallel,
)


@pytest.mark.parametrize('n,m,block', [(1, 4, 16), (17, 4, 16), (129, 8, 32), (73, 16, 16)])
def test_head_mix(n, m, block):
    x = rand(n, m, dtype=torch.float32)
    g = rand(n, m, dtype=torch.float32)
    scale = torch.tensor([0.7])
    base = rand(m, dtype=torch.float32)
    xn, gn, sn, bn = [v.to(NPU) for v in (x, g, scale, base)]
    yn, dxn = torch.empty_like(xn), torch.empty_like(xn)
    ds = torch.empty((4, 1), device=NPU)
    db = torch.empty((4, m), device=NPU)
    kw = dict(token_block_size=block, num_vec_cores=4)
    f = mhc_head_compute_mix_fwd_parallel.compile(xn, sn, bn, yn, mhc_pre_eps=1e-4, **kw)
    record(f, f'head_fwd_{n}_{m}')
    f(xn, sn, bn, yn)
    sig = torch.sigmoid(x * scale + base)
    torch.testing.assert_close(yn.cpu(), sig + 1e-4)
    b = mhc_head_compute_mix_bwd_parallel.compile(gn, xn, sn, bn, dxn, ds, db, **kw)
    record(b, f'head_bwd_{n}_{m}')
    b(gn, xn, sn, bn, dxn, ds, db)
    grad = sig * (1 - sig) * g
    torch.testing.assert_close(dxn.cpu(), grad * scale)
    # Check each core's partition, including cores assigned no token blocks.
    ds_ref = torch.zeros_like(ds.cpu())
    db_ref = torch.zeros_like(db.cpu())
    for tile, start in enumerate(range(0, n, block)):
        part = grad[start : start + block]
        ds_ref[tile % 4, 0] += (part * x[start : start + block]).sum()
        db_ref[tile % 4] += part.sum(0)
    torch.testing.assert_close(ds.cpu(), ds_ref)
    torch.testing.assert_close(db.cpu(), db_ref)
