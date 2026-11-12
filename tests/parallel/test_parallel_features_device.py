"""Feature acceptance on NPU. Run exclusively through task-submit."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T
from test_parallel_device import NPU, record, rand
from probe_features import grouped_broadcast, packed_scale, transpose
from probe_branch_local_bind import make_kernel as branch_kernel


@pytest.mark.parametrize(
    'name,factory', [('grouped', grouped_broadcast), ('bits', packed_scale), ('transpose', transpose), ('branch', branch_kernel)]
)
def test_feature(name, factory):
    kernel = tilelang.compile(factory(), target='pto')
    record(kernel, name)
    if name == 'grouped':
        x, s = rand(128, dtype=torch.float32), rand(4, dtype=torch.float32)
        out = torch.empty_like(x, device=NPU)
        kernel(x.to(NPU), s.to(NPU), out)
        expected = x * s.repeat_interleave(32)
    elif name == 'bits':
        x = torch.linspace(0.001, 3.0, 64)
        out = torch.empty_like(x, device=NPU)
        kernel(x.to(NPU), out)
        expected = ((((x.view(torch.int32) - 1) >> 23) + 1) << 23).view(torch.float32)
    elif name == 'transpose':
        x = rand(32, 32, dtype=torch.float32)
        out = torch.empty_like(x, device=NPU)
        kernel(x.to(NPU), out)
        expected = x.T
    else:
        x = rand(64, dtype=torch.float32)
        out = torch.full_like(x, 17, device=NPU)
        kernel(x.to(NPU), out)
        expected = torch.full_like(x, 17)
        expected[:32] = x[:32].sigmoid()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-6 if name == 'branch' else 0, atol=1e-7 if name == 'branch' else 0)


@tilelang.jit(target='pto')
def reduction_kernel(m, n, dim, kind, dtype):
    count = n if dim == 0 else m

    @T.prim_func
    def main(A: T.Tensor((m, n), dtype), B: T.Tensor((count,), T.float32)):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), dtype)
            b = T.alloc_shared((count,), T.float32)
            T.copy(A, a)
            with T.SimdVF(lanes=64):
                if kind == 'sum':
                    T.reduce_sum(a, b, dim=dim)
                elif kind == 'max':
                    T.reduce_max(a, b, dim=dim)
                elif kind == 'min':
                    T.reduce_min(a, b, dim=dim)
                else:
                    T.reduce_absmax(a, b, dim=dim)
            T.copy(b, B)

    return main


@pytest.mark.parametrize(
    'm,n,dim,kind,dtype',
    [(4, 77, 1, 'sum', torch.float32), (3, 32, 1, 'max', torch.bfloat16), (17, 32, 0, 'min', torch.float32), (4, 128, 1, 'absmax', torch.float32)],
)
def test_reduce(m, n, dim, kind, dtype):
    x = rand(m, n, dtype=dtype)
    out = torch.empty((n if dim == 0 else m,), device=NPU)
    kernel = reduction_kernel(m, n, dim, kind, 'bfloat16' if dtype == torch.bfloat16 else 'float32')
    record(kernel, f'reduce_{kind}_{m}_{n}_{dim}')
    kernel(x.to(NPU), out)
    xf = x.float()
    expected = xf.sum(dim) if kind == 'sum' else xf.amax(dim) if kind == 'max' else xf.amin(dim) if kind == 'min' else xf.abs().amax(dim)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=2e-6)
