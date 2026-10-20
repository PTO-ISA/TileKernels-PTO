#!/usr/bin/env python3
"""Adversarial ASC-vs-VMI checks for the PTOAS #1616 review risks.

Targets the cases that random data does not reach:
  * forward/backward exact divisions with subnormal operands (finite-domain
    check; the subnormal sweep does NOT exercise the 'le'/'lt' tie branch:
    b = 2 there, a/2 is exact, and r = 0)
  * forward e4m3/bf16 non-finite outputs, including amax = inf
  * backward e4m3 (packed TMA / packed non-TMA) and bf16 non-finite inputs,
    comparing x_grad, x_grad_fp8, SF and act_out bitwise

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
    fn = tile_kernels.quant.swiglu_backward_and_per_token_cast if kw.get('fmt') == 'e4m3' else tile_kernels.quant.swiglu_backward
    ba.get_swiglu_backward_kernel_asc = get_swiglu_backward_kernel_vmi if vmi else ob
    try:
        return fn(x, g, **kw)
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
    # For tiny |x|, b = 1 + exp(-x) = 2 exactly and a/2 is exact, so r = 0.
    # This sweep checks finite subnormal handling, not the le/lt tie branch.
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
        (20.0, 448.0 / 20.0),  # finite control, ~448
        (20.0, 448.00003 / 20.0),  # finite control, just above 448
        (20.0, 464.0 / 20.0),  # finite control, RNE overflow threshold
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


def forward_nonfinite_bf16():
    tile_kernels.set_token_alignment(128)
    x = torch.randn((T, 2 * H), dtype=torch.bfloat16, device='npu')
    rows = [
        (float('inf'), 1.0),
        (float('-inf'), 1.0),
        (float('nan'), 1.0),
        (-0.0, 1.0),
        (-1e-40, 1.0),  # bf16 subnormal
        (3e38, 1.0),  # bf16 finite max vicinity
        (2.0, 3.0),  # finite control
    ]
    for i, (left, right) in enumerate(rows):
        x[i, :H] = left
        x[i, H:] = right
    return check('fwd_bf16_nonfinite', run_f(False, x, fmt='bf16'), run_f(True, x, fmt='bf16'))


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


def backward_nonfinite(tma):
    tile_kernels.set_token_alignment(256)
    torch.manual_seed(20261021)
    x_f32 = torch.randn((T, 2 * H), dtype=torch.float32, device='npu')
    x_f32[0, :H] = float('nan')
    x_f32[0, H:] = 1.0
    x_f32[1, :H] = -0.0
    x_f32[1, H:] = 1.0
    x_f32[2, :H] = -1e-40  # underflows to -0 in e4m3
    x_f32[2, H:] = 1.0
    x_data = x_f32.to(torch.float8_e4m3fn)
    x_sf = torch.ones((T, 2 * H // 32), dtype=torch.float32, device='npu')
    g = torch.randn((T, H), dtype=torch.float32, device='npu')
    g[0, :] = float('inf')
    g[1, :] = float('-inf')
    g[2, :] = float('nan')
    g[3, :] = -0.0
    g[4, :] = -1e-40
    kw = dict(
        fmt='e4m3',
        num_per_channels=32,
        use_tma_aligned_col_major_sf=tma,
        round_sf=True,
        use_packed_ue8m0=True,
        do_recompute=True,
    )
    tag = 'tma' if tma else 'packed'
    return check(f'bwd_e4m3_nonfinite_{tag}', run_b(False, (x_data, x_sf), g, **kw), run_b(True, (x_data, x_sf), g, **kw))


def backward_nonfinite_bf16():
    tile_kernels.set_token_alignment(128)
    torch.manual_seed(20261022)
    x = torch.randn((T, 2 * H), dtype=torch.bfloat16, device='npu')
    x[0, :H] = float('inf')
    x[1, :H] = float('-inf')
    x[2, :H] = float('nan')
    x[3, :H] = -0.0
    x[4, :H] = -1e-40  # bf16 subnormal
    x[5, :H] = 3e38
    for row in range(6):
        x[row, H:] = 1.0
    g = torch.randn((T, H), dtype=torch.bfloat16, device='npu')
    g[0, :] = float('inf')
    g[1, :] = float('-inf')
    g[2, :] = float('nan')
    g[3, :] = -0.0
    g[4, :] = -1e-40
    g[5, :] = 3e38
    return check('bwd_bf16_nonfinite', run_b(False, x, g, fmt='bf16', do_recompute=True), run_b(True, x, g, fmt='bf16', do_recompute=True))


if __name__ == '__main__':
    results = [
        forward_subnormal(torch.float32, 'fp32'),
        forward_subnormal(torch.bfloat16, 'bf16'),
        forward_nonfinite(),
        forward_nonfinite_bf16(),
        forward_sf_only(),
        backward_subnormal(torch.float32, 'fp32'),
        backward_subnormal(torch.bfloat16, 'bf16'),
        backward_nonfinite(True),
        backward_nonfinite(False),
        backward_nonfinite_bf16(),
    ]
    print('ALL_OK' if all(results) else 'HAS_MISMATCH')
