from test_parallel_device import NPU, record
from parallel_division_kernel import division
import pytest
import torch


@pytest.mark.parametrize('n', [77, 128])
def test_exact_division(n):
    a = torch.tensor([448.0, 448.0, 1.0, -1.0, 0.0, 1e30, 1e-20, float('inf'), float('nan'), -0.0]).repeat((n + 9) // 10)[:n]
    b = torch.tensor([1.59375, 1.96875, 3.0, 7.0, 2.0, 1e20, 1e-10, 2.0, 1.0, 4.0]).repeat((n + 9) // 10)[:n]
    an, bn = a.to(NPU), b.to(NPU)
    cn = torch.empty_like(an)
    kernel = division(n)
    record(kernel, f'division_{n}')
    kernel(an, bn, cn)
    torch.testing.assert_close(cn.cpu(), a / b, rtol=0, atol=0, equal_nan=True)
