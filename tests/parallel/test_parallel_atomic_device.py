from test_parallel_device import NPU
import torch
import pytest
import tilelang
from tilelang.ascend import language as T


@pytest.mark.parametrize('dtype', [torch.int32, torch.uint32, torch.int64, torch.uint64])
@pytest.mark.parametrize('kind', ['add', 'min', 'max'])
def test_integer_atomic_return(dtype, kind):
    dt = T.dtype(dtype)
    op = getattr(T, 'atomic_' + kind)

    @T.prim_func
    def main(C: T.Tensor((1,), dt), P: T.Tensor((1,), dt)):
        with T.Kernel(1), T.SimtVF(threads=1):
            for i in T.Parallel(1):
                P[i] = op(C[i], T.cast(3, dt), return_prev=True)

    width = 32 if dtype in [torch.int32, torch.uint32] else 64
    initial = -7 if dtype in [torch.int32, torch.int64] else (1 << (width - 1)) + 7
    c = torch.tensor([initial], dtype=dtype, device=NPU)
    p = torch.empty_like(c)
    k = tilelang.compile(main, target='pto')
    k(c, p)
    assert p.cpu().item() == initial
    expected = initial + 3 if kind == 'add' else (min(initial, 3) if kind == 'min' else max(initial, 3))
    assert c.cpu().item() == expected
