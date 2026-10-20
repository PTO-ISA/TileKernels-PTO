"""Run the existing SwiGLU tests against the VMI kernels.

Usage (from the repository root):
    TK_USE_SWIGLU_VMI=1 python -m pytest tests/quant/test_swiglu_forward.py \
        tests/quant/test_swiglu_backward.py -q -p vmi_swiglu_plugin

The dispatchers import the ASC getter lazily inside the public API, so
replacing the module attribute before the tests run is enough to exercise the
VMI implementation without modifying the repository.
"""

import os


def pytest_configure(config):
    if os.environ.get('TK_USE_SWIGLU_VMI') != '1':
        return

    import tile_kernels.quant.swiglu_forward_asc as forward_asc
    import tile_kernels.quant.swiglu_backward_asc as backward_asc
    from tile_kernels.quant.swiglu_forward_vmi import get_swiglu_forward_kernel_vmi
    from tile_kernels.quant.swiglu_backward_vmi import get_swiglu_backward_kernel_vmi

    forward_asc.get_swiglu_forward_kernel_asc = get_swiglu_forward_kernel_vmi
    backward_asc.get_swiglu_backward_kernel_asc = get_swiglu_backward_kernel_vmi
