"""Direct PTO comparison; the public wrapper deliberately stays unchanged."""
import pytest
import torch

from tile_kernels.config import get_num_vec_cores
from tile_kernels.engram.engram_grad_w_reduce_parallel import get_engram_grad_w_reduce_kernel_parallel
from tile_kernels.engram.engram_grad_w_reduce_vmi import get_engram_grad_w_reduce_kernel_vmi


@pytest.mark.parametrize('hidden,partials,heads', [(64, 1, 1), (192, 3, 4), (1088, 7, 4), (2048, 32, 4), (7168, 32, 4)])
def test_parallel_matches_vmi(hidden, partials, heads):
    torch.manual_seed(42)
    device = 'npu'
    args = [torch.randn(partials, heads, hidden, device=device),
            torch.randn(heads, hidden, device=device).bfloat16(),
            torch.randn(heads, hidden, device=device).bfloat16()]
    initial = [torch.randn(heads, hidden, device=device) for _ in range(2)]
    outputs = []
    for builder in (get_engram_grad_w_reduce_kernel_vmi, get_engram_grad_w_reduce_kernel_parallel):
        out = [x.clone() for x in initial]
        builder(hidden, partials, heads, get_num_vec_cores())(*args, *out)
        outputs.append(out)
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_parallel_preserves_partial_order():
    # A tree reduction returns a different answer for these cancellation terms.
    hidden, partials, heads = 64, 4, 1
    values = torch.tensor([2.0 ** 25, 1.0, -(2.0 ** 25), 1.0], device='npu')
    partial = values[:, None, None].expand(partials, heads, hidden).contiguous()
    weight = torch.ones(heads, hidden, device='npu', dtype=torch.bfloat16)
    out = [torch.zeros(heads, hidden, device='npu') for _ in range(2)]
    get_engram_grad_w_reduce_kernel_parallel(hidden, partials, heads, get_num_vec_cores())(partial, weight, weight, *out)
    for value in out:
        torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)


def test_parallel_fma_matches_vmi_exactly():
    hidden, partials, heads = 64, 1, 1
    partial = torch.full((partials, heads, hidden), 1 + 2 ** -12, device='npu')
    # BF16 weights are exact; choose cancellation that exposes split rounding.
    weight = torch.full((heads, hidden), 1 + 2 ** -7, device='npu', dtype=torch.bfloat16)
    initial = torch.full((heads, hidden), -(1 + 2 ** -7 + 2 ** -12), device='npu')
    results = []
    for builder in (get_engram_grad_w_reduce_kernel_vmi, get_engram_grad_w_reduce_kernel_parallel):
        out = [initial.clone(), initial.clone()]
        builder(hidden, partials, heads, get_num_vec_cores())(partial, weight, weight, *out)
        results.append(out)
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
