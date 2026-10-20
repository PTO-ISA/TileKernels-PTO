#!/usr/bin/env python3
"""Extra ASC-vs-VMI coverage requested in review."""

import torch
import torch_npu
import tile_kernels
import tile_kernels.quant.swiglu_forward_asc as fa
import tile_kernels.quant.swiglu_backward_asc as ba
from tile_kernels.quant.swiglu_forward_vmi import get_swiglu_forward_kernel_vmi
from tile_kernels.quant.swiglu_backward_vmi import get_swiglu_backward_kernel_vmi

of, ob = fa.get_swiglu_forward_kernel_asc, ba.get_swiglu_backward_kernel_asc
H = 2048
PSUM = torch.tensor([600, 1600], dtype=torch.int32, device='npu')


def cpu(t):
    t = t.detach().cpu()
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def cmp_tree(a, b, rows):
    if isinstance(a, tuple):
        return all(cmp_tree(x, y, rows) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(cpu(a)[rows], cpu(b)[rows])
    return a == b


def layout(use_psum):
    if not use_psum:
        return None, torch.ones(1024, dtype=torch.bool)
    mask = torch.zeros(1792, dtype=torch.bool)
    mask[0:600] = True
    mask[768:1600] = True
    return PSUM, mask


def run_f(vmi, x, fn, **kw):
    fa.get_swiglu_forward_kernel_asc = get_swiglu_forward_kernel_vmi if vmi else of
    try:
        return fn(x=x, **kw)
    finally:
        fa.get_swiglu_forward_kernel_asc = of


def run_b(vmi, x, g, fn, **kw):
    ba.get_swiglu_backward_kernel_asc = get_swiglu_backward_kernel_vmi if vmi else ob
    try:
        return fn(x, g, **kw)
    finally:
        ba.get_swiglu_backward_kernel_asc = ob


def sf_kwargs(fmt, packed, round_sf, tma):
    if fmt != 'e4m3':
        return {}
    return dict(
        fmt=fmt,
        num_per_channels=32,
        use_tma_aligned_col_major_sf=tma,
        round_sf=round_sf,
        use_packed_ue8m0=packed,
    )


def fwd(tag, dtype, use_psum, fmt, weights=False, rsf=None, clamp=None, packed=False, round_sf=False, tma=False):
    tile_kernels.set_token_alignment(256 if use_psum else 128)
    psum, mask = layout(use_psum)
    torch.manual_seed(101)
    T = 1792 if use_psum else 1024
    x = torch.randn((T, 2 * H), device='npu', dtype=dtype)
    w = torch.rand((T,), device='npu', dtype=torch.float32)
    if use_psum:
        m = mask.to('npu')
        x[~m] = 0
        w[~m] = 0
    kw = sf_kwargs(fmt, packed, round_sf, tma)
    if fmt != 'e4m3':
        kw['fmt'] = fmt
    if weights:
        kw.update(topk_weights=w, routed_scaling_factor=rsf)
    if clamp is not None:
        kw.update(clamp_value=clamp)
    if fmt == 'e4m3':
        fn = tile_kernels.quant.swiglu_forward_and_per_token_cast
    else:
        fn = tile_kernels.quant.swiglu_forward
    a = run_f(False, x, fn, psum_num_tokens_per_expert=psum, **kw)
    b = run_f(True, x, fn, psum_num_tokens_per_expert=psum, **kw)
    torch.npu.synchronize()
    print(f'{tag}:', 'OK' if cmp_tree(a, b, mask) else 'MISMATCH')


def bwd(tag, fmt, use_psum, dtype, weights=False, rsf=None, clamp=None, recompute=False, packed=False, round_sf=False, tma=False):
    tile_kernels.set_token_alignment(256 if (use_psum or fmt == 'e4m3') else 128)
    psum, mask = layout(use_psum)
    torch.manual_seed(102)
    T = 1792 if use_psum else 1024
    g_dtype = torch.bfloat16 if fmt == 'e4m3' else dtype
    x_raw = torch.randn((T, 2 * H), device='npu', dtype=torch.bfloat16)
    g = torch.randn((T, H), device='npu', dtype=g_dtype)
    w = torch.rand((T,), device='npu', dtype=torch.float32)
    if use_psum:
        m = mask.to('npu')
        x_raw[~m] = 0
        g[~m] = 0
        w[~m] = 0
    kw = sf_kwargs(fmt, packed, round_sf, tma)
    if fmt != 'e4m3':
        kw['fmt'] = fmt
    if weights:
        kw.update(topk_weights=w, routed_scaling_factor=rsf)
    if clamp is not None:
        kw.update(clamp_value=clamp)
    kw.update(do_recompute=recompute)
    if fmt == 'e4m3':
        x = tile_kernels.quant.per_token_cast(x_raw, 'e4m3', num_per_channels=32)
        fn = tile_kernels.quant.swiglu_backward_and_per_token_cast
    else:
        x = x_raw.to(dtype)
        fn = tile_kernels.quant.swiglu_backward
    a = run_b(False, x, g, fn, psum_num_tokens_per_expert=psum, **kw)
    b = run_b(True, x, g, fn, psum_num_tokens_per_expert=psum, **kw)
    torch.npu.synchronize()
    print(f'{tag}:', 'OK' if cmp_tree(a, b, mask) else 'MISMATCH')


fwd('fwd_fp32_in_bf16_out_no_psum', torch.float32, False, 'bf16')
fwd('fwd_fp32_in_fp32_out_no_psum_clamp', torch.float32, False, 'fp32', clamp=0.5)
fwd('fwd_fp32_in_bf16_out_weights_rsf', torch.float32, False, 'bf16', weights=True, rsf=2.5)
fwd('fwd_fp32_in_e4m3_packed_no_tma', torch.float32, False, 'e4m3', packed=True, round_sf=True, tma=False)
fwd('fwd_e4m3_packed_no_tma_psum_weights', torch.bfloat16, True, 'e4m3', weights=True, rsf=2.5, packed=True, round_sf=True, tma=False)
fwd('fwd_e4m3_dense_exact_no_psum', torch.bfloat16, False, 'e4m3', round_sf=False, tma=False)
fwd('fwd_e4m3_tma_weights_clamp', torch.bfloat16, True, 'e4m3', weights=True, rsf=2.5, clamp=0.5, packed=True, round_sf=True, tma=True)

bwd('bwd_bf16_no_psum_no_weights', 'bf16', False, torch.bfloat16)
bwd('bwd_bf16_psum_weights_no_rsf', 'bf16', True, torch.bfloat16, weights=True, rsf=None)
bwd('bwd_bf16_psum_weights_recompute_clamp', 'bf16', True, torch.bfloat16, weights=True, rsf=2.5, clamp=0.5, recompute=True)
bwd('bwd_fp32_no_psum_weights_recompute', 'fp32', False, torch.float32, weights=True, rsf=2.5, recompute=True)
bwd(
    'bwd_e4m3_no_psum_tma_weights_recompute',
    'e4m3',
    False,
    torch.bfloat16,
    weights=True,
    rsf=2.5,
    recompute=True,
    packed=True,
    round_sf=True,
    tma=True,
)
bwd('bwd_e4m3_psum_packed_no_tma_weights', 'e4m3', True, torch.bfloat16, weights=True, rsf=2.5, packed=True, round_sf=True, tma=False)
bwd('bwd_e4m3_psum_dense_exact_recompute', 'e4m3', True, torch.bfloat16, weights=True, rsf=2.5, recompute=True, round_sf=False, tma=False)
bwd('bwd_e4m3_psum_tma_no_weights_recompute', 'e4m3', True, torch.bfloat16, recompute=True, packed=True, round_sf=True, tma=True)
print('REVIEW_COVERAGE_DONE')
