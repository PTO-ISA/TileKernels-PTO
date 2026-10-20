#!/usr/bin/env python3
"""Adversarial ASC-vs-VMI checks for the PTOAS #1616 review risks.

Targets the cases that random data does not reach:
  * forward exact division with subnormal operands (the 'le'/'lt' tie case)
  * forward packed UE8M0 with amax = inf (ASC truncation vs PTODSL SAT)
  * backward exact divisions with subnormal operands

Run inside the A5 container after sourcing activate_vmi_migration.sh.
"""

import torch
import torch_npu

import tile_kernels
import tile_kernels.quant.swiglu_backward_asc as ba
import tile_kernels.quant.swiglu_forward_asc as fa
from tile_kernels.quant.swiglu_backward_vmi import get_swiglu_backward_kernel_vmi
from tile_kernels.quant.swiglu_forward_vmi import get_swiglu_forward_kernel_vmi

H = 2048
T = 1024
of, ob = fa.get_swiglu_forward_kernel_asc, ba.get_swiglu_backward_kernel_asc


def cpu(t):
    t = t.detach().cpu()
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def bits_equal(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(bits_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        x, y = cpu(a), cpu(b)
        if x.dtype.is_floating_point:
            return torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        return torch.equal(x, y)
    return a == b


def run_f(vmi, x, **kw):
    fn = tile_kernels.quant.swiglu_forward_and_per_token_cast if kw.get('fmt') == 'e4m3' else tile_kernels.quant.swiglu_forward
    fa.get_swiglu_forward_kernel_asc = get_swiglu_forward_kernel_vmi if vmi else of
    try:
        return fn(x, **kw)
    finally:
        fa.get_swiglu_forward_kernel_asc = of


def run_b(vmi, x, g, **kw):
    ba.get_swiglu_backward_kernel_asc = get_swiglu_backward_kernel_vmi if vmi else ob
    try:
        return tile_kernels.quant.swiglu_backward(x, g, **kw)
    finally:
        ba.get_swiglu_backward_kernel_asc = ob


def subnormal_bits(lo, hi, count, negative):
    """Evenly sample float32 subnormal bit patterns."""
    step = max(1, (hi - lo + 1) // count)
    idx = torch.arange(lo, hi + 1, step, dtype=torch.int64)
    if idx.numel() < count:
        idx = torch.cat((idx, idx[-1].repeat(count - idx.numel())))
    idx = idx[:count]
    if negative:
        idx = idx - (1 << 32)
    return idx.to(torch.int32).view(torch.float32)


def check(tag, a, b):
    ok = bits_equal(a, b)
    print(f'{tag}:', 'OK' if ok else 'MISMATCH', flush=True)
    return ok


def forward_subnormal(dtype, fmt):
    tile_kernels.set_token_alignment(128)
    x = torch.ones((T, 2 * H), dtype=dtype, device='npu')
    if dtype == torch.float32:
        # All negative float32 subnormals, evenly sampled.
        vals = subnormal_bits(0x80000001, 0x807FFFFF, T * H, negative=True).view(T, H).to('npu')
    else:
        # All negative bf16 subnormals, tiled.
        bits = torch.arange(0x8001, 0x8080, dtype=torch.int64).to(torch.int16)
        vals = bits.view(torch.bfloat16).to('npu')
        vals = vals.repeat((T * H + vals.numel() - 1) // vals.numel())[: T * H].view(T, H)
    x[:, :H] = vals
    x[:, H:] = 1.0
    return check(f'fwd_{fmt}_subnormal', run_f(False, x, fmt=fmt), run_f(True, x, fmt=fmt))


def forward_nonfinite():
    tile_kernels.set_token_alignment(256)
    x = torch.randn((T, 2 * H), dtype=torch.bfloat16, device='npu')
    rows = [
        (3e38, 3e38),  # +inf result -> amax = inf -> exponent 256
        (3e38, -3e38),  # -inf result
        (float('nan'), 1.0),  # NaN result
        (-100.0, 100.0),  # -0 result
        (float('inf'), 1.0),  # inf input
        (float('-inf'), 1.0),  # -inf input
    ]
    for i, (left, right) in enumerate(rows):
        x[i, :H] = left
        x[i, H:] = right
    kw = dict(fmt='e4m3', num_per_channels=32, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    return check('fwd_e4m3_nonfinite', run_f(False, x, **kw), run_f(True, x, **kw))


def forward_sf_only():
    # Focused check for the vcmax(group=...) scale mapping: compare only SF.
    tile_kernels.set_token_alignment(256)
    torch.manual_seed(20261020)
    x = torch.randn((T, 2 * H), dtype=torch.bfloat16, device='npu')
    kw = dict(fmt='e4m3', num_per_channels=32, use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True)
    a = run_f(False, x, **kw)
    b = run_f(True, x, **kw)
    return check('fwd_e4m3_sf_only', a[1], b[1])


def backward_subnormal(dtype, fmt):
    tile_kernels.set_token_alignment(128)
    x = torch.ones((T, 2 * H), dtype=dtype, device='npu')
    if dtype == torch.float32:
        vals = subnormal_bits(0x80000001, 0x807FFFFF, T * H, negative=True).view(T, H).to('npu')
    else:
        bits = torch.arange(0x8001, 0x8080, dtype=torch.int64).to(torch.int16)
        vals = bits.view(torch.bfloat16).to('npu')
        vals = vals.repeat((T * H + vals.numel() - 1) // vals.numel())[: T * H].view(T, H)
    x[:, :H] = vals
    x[:, H:] = 1.0
    g = torch.randn((T, H), dtype=dtype, device='npu')
    return check(f'bwd_{fmt}_subnormal', run_b(False, x, g, fmt=fmt), run_b(True, x, g, fmt=fmt))


if __name__ == '__main__':
    results = [
        forward_subnormal(torch.float32, 'fp32'),
        forward_subnormal(torch.bfloat16, 'bf16'),
        forward_nonfinite(),
        forward_sf_only(),
        backward_subnormal(torch.float32, 'fp32'),
        backward_subnormal(torch.bfloat16, 'bf16'),
    ]
    print('ALL_OK' if all(results) else 'HAS_MISMATCH')
