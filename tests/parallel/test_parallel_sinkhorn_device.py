from test_parallel_device import NPU, rand, record
import pytest
import torch
from tile_kernels.mhc.sinkhorn_parallel import mhc_sinkhorn_fwd_parallel, mhc_sinkhorn_bwd_parallel


@pytest.mark.parametrize('n,repeat', [(1, 1), (17, 3), (33, 10)])
def test_sinkhorn(n, repeat):
    eps = 1e-6
    x = rand(n, 4, 4, dtype=torch.float32).requires_grad_()
    g = rand(n, 4, 4, dtype=torch.float32)
    ref = x.softmax(-1) + eps
    ref = ref / (ref.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        ref = ref / (ref.sum(-1, keepdim=True) + eps)
        ref = ref / (ref.sum(-2, keepdim=True) + eps)
    dx = torch.autograd.grad(ref, x, g)[0]
    xn, gn = x.detach().to(NPU), g.to(NPU)
    out, dout = torch.empty_like(xn), torch.empty_like(xn)
    k = mhc_sinkhorn_fwd_parallel.compile(xn, out, repeat=repeat, eps=eps, token_block_size=4, num_vec_cores=4)
    record(k, f'sinkhorn_fwd_{n}_{repeat}')
    k(xn, out)
    torch.testing.assert_close(out.cpu(), ref, rtol=2e-5, atol=2e-6)
    k = mhc_sinkhorn_bwd_parallel.compile(gn, xn, dout, repeat=repeat, eps=eps, token_block_size=8, num_vec_cores=4)
    record(k, f'sinkhorn_bwd_{n}_{repeat}')
    k(gn, xn, dout)
    torch.testing.assert_close(dout.cpu(), dx, rtol=3e-4, atol=3e-6)
