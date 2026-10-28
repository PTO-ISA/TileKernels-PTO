import math

import tilelang
from tilelang.ascend import language as T

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_swiglu_forward_kernel_parallel(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    clamp_value,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    has_clamp = clamp_value is not None
    has_psum = num_experts > 0

    assert not count_clamp or has_clamp
    assert not with_routed_scaling or with_weight
    assert hidden % 128 == 0, f'Ascend swiglu requires hidden % 128 == 0, got {hidden}'
    assert not out_config.use_tma_aligned_col_major_sf or out_config.use_packed_ue8m0
    if out_config.with_sf:
        assert out_config.sf_block == (1, 32), 'Ascend swiglu only supports sf_block=(1, 32)'

    token_group = 32 if out_config.with_sf and out_config.use_tma_aligned_col_major_sf else 1
    if has_psum:
        assert alignment % token_group == 0, f'Ascend swiglu col-major path requires alignment % {token_group} == 0, got {alignment}'
    use_compact_groups = has_psum and token_group > 1
    # Fast certificate is only enabled for bf16 -> e4m3 with power-of-two sf.
    # It checks that fast and exact division give the same e4m3/sf result;
    # unsafe tokens fall back to exact division.
    use_fast_certificate = False  # T.Parallel division uses the exact VMI mode.
    num_stages = 3
    # Multiplication by a weight requires a larger certificate error bound.
    certificate_error = 5 if with_weight else 2
    vec_size = 64
    num_vecs = hidden // vec_size
    hidden_aligned = align(hidden, 256)

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')
    sf_shape = get_sf_shape((num_expanded_tokens, hidden), out_config) if out_config.with_sf else (1, 1)
    num_groups = hidden // 32
    pack_factor = get_packed_ue8m0_pack_factor()

    @T.macro
    def init_transpose_indices(indices):
        with T.SimdVF(lanes=64):
            T.evaluate(0)

    @T.macro
    def transpose_packed_scales(src, dst, indices):
        src_u16 = T.Tensor((token_group, align(num_groups, 64) // pack_factor), T.uint16, src.data)
        with T.SimdVF(lanes=64):
            for group, row in T.Parallel(num_groups // pack_factor, token_group):
                dst[group, row] = src_u16[row, group]

    @T.macro
    def compute_swiglu(xl_ub, xr_ub, values_ub, out_ub, weight, precision, amax_ub, with_certificate, pacc_buf, count_clamp, acc=None):
        for j in T.Parallel(hidden):
            left = T.float32(xl_ub[j])
            right = T.float32(xr_ub[j])
            if has_clamp:
                left = T.min(left, clamp_value)
                right = T.min(T.max(right, -clamp_value), clamp_value)
            value = (left / (1.0 + T.exp(-left))) * right
            if with_weight:
                value = value * weight
            if out_config.with_sf:
                values_ub[j] = value
            else:
                out_ub[j] = T.cast(value, out_config.dtype)
        if out_config.with_sf:
            for group in T.Parallel(num_groups):
                maximum = T.alloc_var(T.float32, init=0.0)
                for k in T.serial(32):
                    maximum = T.max(maximum, T.abs(values_ub[group * 32 + k]))
                amax_ub[group] = maximum
        if count_clamp:
            for j in T.Parallel(vec_size):
                c0 = T.alloc_var(T.int32, init=pacc_buf[0, j])
                c1 = T.alloc_var(T.int32, init=pacc_buf[1, j])
                c2 = T.alloc_var(T.int32, init=pacc_buf[2, j])
                for k in T.serial(num_vecs):
                    left = T.float32(xl_ub[k * vec_size + j])
                    right = T.float32(xr_ub[k * vec_size + j])
                    c0 = c0 + T.if_then_else(left > clamp_value, 1, 0)
                    c1 = c1 + T.if_then_else(right > clamp_value, 1, 0)
                    c2 = c2 + T.if_then_else(right < -clamp_value, 1, 0)
                pacc_buf[0, j] = c0
                pacc_buf[1, j] = c1
                pacc_buf[2, j] = c2
        return None, None

    @T.macro
    def quantize_f32(values_ub, amax_ub, inverse_ub, sf_dense_ub, out_ub, token_offset):
        with T.SimdVF(lanes=64):
            for group in T.Parallel(num_groups):
                maximum = T.max(amax_ub[group], out_config.clamp_min_value)
                if out_config.round_sf:
                    bits = T.reinterpret(maximum * (1.0 / 448.0), T.uint32)
                    exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
                    scale = T.reinterpret(exponent << 23, T.float32)
                    inverse = T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
                else:
                    scale = maximum / 448.0
                    inverse = 448.0 / maximum
                if out_config.use_packed_ue8m0:
                    sf_dense_ub[token_offset, group] = T.cast(exponent, T.uint8)
                else:
                    sf_dense_ub[token_offset, group] = scale
                inverse_ub[group] = inverse
            for j in T.Parallel(hidden):
                out_ub[j] = T.cast(values_ub[j] * inverse_ub[j // 32], out_config.dtype)

    @T.macro
    def load_weight(topk_weights, token_id, routed_scaling_factor):
        weight = T.alloc_var(T.float32, init=1.0)
        if with_weight:
            weight = topk_weights[token_id]
            if with_routed_scaling:
                weight = weight * routed_scaling_factor
        return weight

    @T.macro
    def is_valid_token(token_id, expert_id, psum_ub):
        valid = T.alloc_var(T.bool, init=token_id < num_expanded_tokens)
        if has_psum:
            valid = valid and (expert_id < num_experts) and (token_id < psum_ub[expert_id])
        return valid

    @T.macro
    def store_sf_row(sf_dense_ub, out_sf, row, token_id):
        if out_config.with_sf and not out_config.use_tma_aligned_col_major_sf:
            T.copy(sf_dense_ub[row, 0:num_groups], out_sf[token_id, 0])

    @T.macro
    def load_inputs(x_gm, xl, xr, topk_w, rsf, tid):
        weight = load_weight(topk_w, tid, rsf)
        T.copy(x_gm[tid, 0], xl)
        T.copy(x_gm[tid, hidden], xr)
        return weight

    @T.macro
    def finish_token(vals, out, amax, inv, sf, out_gm, out_sf_gm, off, tid, acc_buf, count_clamp):
        if out_config.with_sf:
            quantize_f32(vals, amax, inv, sf, out, off)
        T.copy(out, out_gm[tid, 0])
        store_sf_row(sf, out_sf_gm, off, tid)
        if count_clamp:
            acc_buf[3] += hidden

    @T.prim_func
    def swiglu_forward_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        clamped_count: T.Tensor[(4,), T.int64],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            if count_clamp:
                acc_ub = T.alloc_shared((4,), T.int32)
                pacc_ub = T.alloc_shared((3, vec_size), T.int32)  # per lane accumulator pacc_ub. use vcadd at the last moment

                for i in T.serial(4):
                    acc_ub[i] = 0
                with T.SimdVF(lanes=64):
                    for i, j in T.Parallel(3, vec_size):
                        pacc_ub[i, j] = 0
            else:
                acc_ub = None
                pacc_ub = None

            if has_psum:
                psum_ub = T.alloc_shared((num_experts,), T.int32)
                T.copy(psum_num_tokens_per_expert, psum_ub)
                expert_id = T.alloc_var(T.int32, init=0)
                if use_compact_groups:
                    group_psum_ub = T.alloc_shared((num_experts,), T.int32)
                    group_psum_ub[0] = T.ceildiv(psum_ub[0], token_group)
                    for expert in T.serial(1, num_experts):
                        expert_start = (psum_ub[expert - 1] + alignment - 1) // alignment * alignment
                        expert_groups = T.ceildiv(psum_ub[expert] - expert_start, token_group)
                        group_psum_ub[expert] = group_psum_ub[expert - 1] + expert_groups

            xl_ub = T.alloc_shared((hidden_aligned,), in_dtype)
            xr_ub = T.alloc_shared((hidden_aligned,), in_dtype)
            out_ub = T.alloc_shared((hidden_aligned,), out_config.dtype)
            values_ub = T.alloc_shared((hidden_aligned,), T.float32)
            if use_fast_certificate:
                xl_fb_ub = T.alloc_shared((hidden_aligned,), in_dtype)
                xr_fb_ub = T.alloc_shared((hidden_aligned,), in_dtype)
                out_fb_ub = T.alloc_shared((hidden_aligned,), out_config.dtype)
                values_fb_ub = T.alloc_shared((hidden_aligned,), T.float32)
            if out_config.with_sf:
                amax_ub = T.alloc_shared((align(num_groups, 64),), T.float32)
                if use_fast_certificate:
                    certificate_ub = T.alloc_shared((token_group, 2), T.uint32)
                inverse_ub = T.alloc_shared((align(num_groups, 64),), T.float32)
                sf_dense_ub = T.alloc_shared(
                    (token_group, align(num_groups, 64)),
                    T.uint8 if out_config.use_packed_ue8m0 else T.float32,
                )
                if out_config.use_tma_aligned_col_major_sf:
                    sf_ub = T.alloc_shared((num_groups // pack_factor, token_group), T.uint16)
                    transpose_indices_ub = T.alloc_shared((128,), T.uint16)
            else:
                amax_ub = None
                inverse_ub = None
                sf_dense_ub = None
            T.annotate_buffer_versions({xl_ub: num_stages, xr_ub: num_stages, out_ub: num_stages})

            if out_config.with_sf and out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)

            for pid in T.Persistent(
                [T.ceildiv(num_expanded_tokens, token_group)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_base = T.alloc_var(T.int32, init=pid * token_group)
                if use_compact_groups:
                    while expert_id < num_experts and pid >= group_psum_ub[expert_id]:
                        expert_id += 1
                    expert_group_base = T.alloc_var(T.int32, init=0)
                    expert_token_base = T.alloc_var(T.int32, init=0)
                    if expert_id > 0:
                        expert_group_base = group_psum_ub[expert_id - 1]
                        expert_token_base = (psum_ub[expert_id - 1] + alignment - 1) // alignment * alignment
                    token_base = expert_token_base + (pid - expert_group_base) * token_group
                elif has_psum:
                    while expert_id < num_experts and token_base >= (psum_ub[expert_id] + alignment - 1) // alignment * alignment:
                        expert_id += 1
                if has_psum:
                    valid_group = token_base < num_expanded_tokens and expert_id < num_experts and token_base < psum_ub[expert_id]
                else:
                    valid_group = token_base < num_expanded_tokens

                if valid_group:
                    for token_offset in T.serial(token_group):
                        token_id = T.alloc_var(T.int32, init=token_base + token_offset)
                        valid = is_valid_token(token_id, expert_id if has_psum else 0, psum_ub if has_psum else None)
                        if valid:
                            weight = load_inputs(x, xl_ub, xr_ub, topk_weights, routed_scaling_factor, token_id)
                            vf_weight = weight
                            with T.SimdVF(lanes=64):
                                compute_swiglu(xl_ub, xr_ub, values_ub, out_ub, vf_weight, 'exact', amax_ub, False, pacc_ub, count_clamp)
                            finish_token(
                                values_ub, out_ub, amax_ub, inverse_ub, sf_dense_ub, out, out_sf, token_offset, token_id, acc_ub, count_clamp
                            )

                    if use_fast_certificate:  # fall back to exact vdiv for unsafe tokens
                        for fallback_offset in T.serial(token_group):
                            token_id = T.alloc_var(T.int32, init=token_base + fallback_offset)
                            valid = is_valid_token(token_id, expert_id if has_psum else 0, psum_ub if has_psum else None)
                            unsafe = T.alloc_var(
                                T.bool,
                                init=(
                                    certificate_ub[fallback_offset, 0] <= certificate_error
                                    or certificate_ub[fallback_offset, 1] >= 0x80000 - certificate_error
                                ),
                            )
                            if valid and unsafe:
                                weight = load_inputs(x, xl_fb_ub, xr_fb_ub, topk_weights, routed_scaling_factor, token_id)
                                with T.SimdVF():
                                    # Clamp counts were already accumulated in the main path; fallback must set count_clamp to False
                                    _, _ = compute_swiglu(xl_fb_ub, xr_fb_ub, values_fb_ub, out_fb_ub, weight, 'exact', amax_ub, False, None, False)
                                finish_token(
                                    values_fb_ub, out_fb_ub, amax_ub, inverse_ub, sf_dense_ub, out, out_sf, fallback_offset, token_id, None, False
                                )

                if out_config.with_sf and out_config.use_tma_aligned_col_major_sf:
                    if valid_group:
                        transpose_packed_scales(sf_dense_ub, sf_ub, transpose_indices_ub)
                        output_sf = T.Tensor((sf_shape[0], sf_shape[1] // pack_factor), T.uint16, out_sf.data)
                        valid_token_end = psum_ub[expert_id] if has_psum else num_expanded_tokens
                        valid_tokens = T.alloc_var(T.int32, init=T.min(token_group, valid_token_end - token_base))
                        T.copy(sf_ub[:, 0:valid_tokens], output_sf[:, token_base : token_base + valid_tokens])

            if count_clamp:
                with T.SimdVF(lanes=64):
                    for i in T.unroll(3, explicit=True):
                        T.reduce_sum(pacc_ub[i, :], acc_ub[i : i + 1], dim=1)

                with T.SimtVF(threads=1):
                    for i in T.serial(4):
                        T.atomic_add(clamped_count[i], T.int64(acc_ub[i]))

    return swiglu_forward_kernel
