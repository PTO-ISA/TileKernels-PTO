"""Direct builder checks; the public wrapper remains unchanged."""
import importlib

import pytest
import torch

from tile_kernels.config import get_num_vec_cores


@pytest.mark.parametrize('hc,hidden', [(1, 256), (4, 768), (4, 2048), (4, 7168)])
def test_parallel_matches_vmi(hc, hidden):
    parallel = importlib.import_module('tile_kernels.engram.engram_fused_weight_parallel')
    from tile_kernels.engram.engram_fused_weight_vmi import get_engram_fused_weight_kernel_vmi

    builders = [get_engram_fused_weight_kernel_vmi, parallel.get_engram_fused_weight_kernel_parallel]
    inputs = [torch.randn(hc, hidden, dtype=torch.bfloat16, device='npu') for _ in range(2)]
    outputs = [torch.empty(hc, hidden, dtype=torch.float32, device='npu') for _ in builders]
    for builder, output in zip(builders, outputs):
        builder(hidden, hc, get_num_vec_cores())(*inputs, output)
    torch.npu.synchronize()
    expected = inputs[0].float() * inputs[1].float()
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
