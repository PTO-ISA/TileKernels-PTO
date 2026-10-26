import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_swiglu_forward_kernel_vmi(
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
    use_fast_certificate = in_dtype == T.bfloat16 and out_config.with_sf and out_config.round_sf and out_config.use_tma_aligned_col_major_sf
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
    def get_scale_and_inverse(amax):
        full = V.create_mask(64, size=64)
        clamped_amax = V.vmaxs(amax, T.float32(out_config.clamp_min_value), full)
        scale_bits = V.vinterpret_cast(V.vmuls(clamped_amax, T.float32(1.0 / 448.0), full), 'uint32')
        exponent = V.vadds(
            V.vshrs(V.vsub(scale_bits, V.vbrc(T.uint32(1), size=64)), T.uint32(23), full),
            T.uint32(1),
            full,
        )
        inverse = V.vinterpret_cast(
            V.vshls(V.vsub(V.vbrc(T.uint32(254), size=64), exponent), T.uint32(23), full),
            'float32',
        )
        if out_config.use_packed_ue8m0:
            return exponent, inverse
        return V.vinterpret_cast(V.vshls(exponent, T.uint32(23), full), 'float32'), inverse

    @T.macro
    def init_transpose_indices(indices):
        # Precompute gather indices shared by every token group when converting the
        # dense token-major scale buffer to the packed col-major output layout.
        with T.SimdVF():
            lane = V.vci(T.int16(0), size=128)
            output_row = V.vand(V.vinterpret_cast(lane, 'uint16'), V.vbrc(T.uint16(token_group - 1), size=128))
            group_in_vector = V.vinterpret_cast(
                V.vshr(lane, V.vbrc(T.int16(int(math.log2(token_group))), size=128)),
                'uint16',
            )
            source_indices = V.vadd(
                V.vmul(output_row, V.vbrc(T.uint16(align(num_groups, 64) // pack_factor), size=128)),
                group_in_vector,
            )
            V.vstore(source_indices, indices[0])

    @T.macro
    def transpose_packed_scales(src, dst, indices):
        # Transpose one token group's packed UE8M0 scales.
        with T.SimdVF():
            groups_per_vector = 128 // token_group
            num_output_values = num_groups // pack_factor * token_group
            num_full_vectors = num_output_values // 128
            num_tail_values = num_output_values % 128
            source_indices = V.vload(indices[0], size=128)
            full_mask = V.create_mask(128, size=128)
            # Read byte-owned SF storage without introducing a typed pointer alias.
            byte_indices = V.vmul(source_indices, V.vbrc(T.uint16(2), size=128))

            if num_full_vectors > 0:
                for output_vector in T.unroll(num_full_vectors, explicit=True):
                    group_base = output_vector * groups_per_vector
                    low = V.vgather(src[0, group_base * 2], byte_indices, full_mask)
                    high = V.vgather(src[0, group_base * 2 + 1], byte_indices, full_mask)
                    values = V.vor(low, V.vshls(high, T.uint16(8), full_mask))
                    V.vstore(values, dst[group_base, 0])

            if num_tail_values > 0:
                tail_group_base = num_full_vectors * groups_per_vector
                tail_mask = V.create_mask(num_tail_values, size=128)
                low = V.vgather(src[0, tail_group_base * 2], byte_indices, tail_mask)
                high = V.vgather(src[0, tail_group_base * 2 + 1], byte_indices, tail_mask)
                values = V.vor(low, V.vshls(high, T.uint16(8), tail_mask))
                V.vstore(values, dst[tail_group_base, 0], tail_mask)

    @T.macro
    def compute_swiglu(xl_ub, xr_ub, values_ub, out_ub, weight, precision, amax_ub, with_certificate, pacc_buf, count_clamp, acc=None):
        # Compute SiLU(x_l) * x_r, optionally collecting clamp counters,
        # quantization amax values, and the fast-division safety certificate.
        # Logical compute width is local to SIMD; ASC buffers and scheduler
        # retain their original 64-lane geometry and certificate storage.
        logical_size = (256 if hidden % 256 == 0 else 128) if out_config.with_sf and not count_clamp else 64
        logical_vectors = hidden // logical_size
        compute_unroll = 1 if logical_size > 64 else 4
        full = V.create_mask(logical_size, size=logical_size)
        zeros = V.vbrc(T.float32(0.0), size=logical_size)
        if count_clamp:
            if acc is None:
                acc = V.alloc_local((3,), V.vreg(logical_size, T.float32))
                for i in T.unroll(3, explicit=True):
                    acc[i] = zeros
        if with_certificate:
            # Loop-carried across tile_id: alloc_var is what lets the eager
            # builder rebind these accumulators inside the loop.
            certificate_min = V.alloc_var(T.uint32, size=logical_size)
            certificate_max = V.alloc_var(T.uint32, size=logical_size)
            certificate_min = V.vbrc(T.uint32(0xFFFFFFFF), size=logical_size)
            certificate_max = V.vbrc(T.uint32(0), size=logical_size)
        else:
            certificate_min = None
            certificate_max = None
        for tile_id in T.serial(T.ceildiv(logical_vectors, compute_unroll)):
            for sub in T.unroll(compute_unroll, explicit=True):
                vector_id = tile_id * compute_unroll + sub
                if vector_id < logical_vectors:
                    col = vector_id * logical_size
                    if in_dtype == T.bfloat16:
                        val_l = V.vcvt(V.vload(xl_ub[col], size=logical_size), 'float32')
                        val_r = V.vcvt(V.vload(xr_ub[col], size=logical_size), 'float32')
                    else:
                        val_l = V.vload(xl_ub[col], size=logical_size)
                        val_r = V.vload(xr_ub[col], size=logical_size)
                    if has_clamp:
                        silu_l = V.vmins(val_l, T.float32(clamp_value), full)
                        mul_r = V.vmaxs(V.vmins(val_r, T.float32(clamp_value), full), T.float32(-clamp_value), full)
                    else:
                        silu_l = val_l
                        mul_r = val_r
                    denominator = V.vadds(V.vexpdif(zeros, silu_l, full), T.float32(1.0), full)
                    if precision == 'ftz_true':
                        quotient = V.vdiv(silu_l, denominator)
                    elif precision == 'exact':
                        quotient = V.vdiv(silu_l, denominator, full, precision='exact')
                    else:
                        raise ValueError(f"unsupported vdiv precision {precision!r}; expected 'ftz_true' or 'exact'")
                    val = V.vmul(quotient, mul_r)
                    out_val = V.vmuls(val, weight, full) if with_weight else val

                    if out_config.with_sf:
                        V.vstore(out_val, values_ub[col])
                    elif out_config.dtype == T.bfloat16:
                        V.vstore(V.vcvt(out_val, 'bfloat16', saturate='NOSAT'), out_ub[col])
                    else:
                        V.vstore(out_val, out_ub[col])

                    if with_certificate:
                        # Keep the low 19 mantissa bits; values too close to either end of this window use the exact fallback.
                        metric = V.vand(V.vinterpret_cast(out_val, 'uint32'), V.vbrc(T.uint32(0x7FFFF), size=logical_size))
                        certificate_min = V.vmin(certificate_min, metric)
                        certificate_max = V.vmax(certificate_max, metric)
                    if count_clamp:
                        clamped_mask_0 = V.vcmps(val_l, T.float32(clamp_value), full, 'gt')
                        clamped_mask_1 = V.vcmps(val_r, T.float32(clamp_value), full, 'gt')
                        clamped_mask_2 = V.vcmps(val_r, T.float32(-clamp_value), full, 'lt')
                        # Merge-mode accumulation: add 1 on active lanes, keep the
                        # previous value elsewhere (PTO vadds only supports zeroing).
                        one = V.vbrc(T.float32(1.0), size=logical_size)
                        acc[0] = V.vadd(acc[0], V.vsel(clamped_mask_0, one, zeros))
                        acc[1] = V.vadd(acc[1], V.vsel(clamped_mask_1, one, zeros))
                        acc[2] = V.vadd(acc[2], V.vsel(clamped_mask_2, one, zeros))
                    if out_config.with_sf:
                        amax = V.vcmax(V.vabs(out_val), full, group=logical_size // 32)
                        V.vstore(amax, amax_ub[vector_id * (logical_size // 32)])
        if count_clamp:
            for i in T.unroll(3, explicit=True):
                V.vstore(V.vadd(V.vload(pacc_buf[i, 0], size=logical_size), acc[i]), pacc_buf[i, 0])
        if with_certificate and logical_size != 64:
            # The public helper contract remains two 64-lane values, allowing
            # ASC's outer certificate buffers and scalar fallback to stay exact.
            certificate_min_out = V.vbrc(V.vcmin(certificate_min, full), size=64, group=1)
            certificate_max_out = V.vbrc(V.vcmax(certificate_max, full), size=64, group=1)
        else:
            certificate_min_out = certificate_min
            certificate_max_out = certificate_max
        return certificate_min_out, certificate_max_out

    @T.macro
    def quantize_f32(values_ub, amax_ub, inverse_ub, sf_dense_ub, out_ub, token_offset):
        # Get inverse scale and quantize FP32 SwiGLU result to FP8.
        with T.SimdVF():
            full = V.create_mask(vec_size, size=vec_size)
            full_scale = V.create_mask(64, size=64)
            for group_batch in T.serial(T.ceildiv(num_groups, 64)):
                group = group_batch * 64
                amax = V.vload(amax_ub[group], size=64)
                if out_config.round_sf:
                    scales_or_exponents, inverses = get_scale_and_inverse(amax)
                    if out_config.use_packed_ue8m0:
                        # int-narrow (uint32 -> uint8): NOSAT is exactly ASC's
                        # reinterpret-truncation, unrelated to the FP8 output
                        # saturate decision below. Spell it out because PTODSL
                        # otherwise defaults to SAT for narrowing, which differs
                        # for exponents >= 256 (amax = inf).
                        V.vstore(V.vcvt(scales_or_exponents, 'uint8', saturate='NOSAT'), sf_dense_ub[token_offset, group])
                    else:
                        V.vstore(scales_or_exponents, sf_dense_ub[token_offset, group])
                    V.vstore(inverses, inverse_ub[group])
                else:
                    clamped_amax = V.vmaxs(amax, T.float32(out_config.clamp_min_value), full_scale)
                    quant_max = V.vbrc(T.float32(448.0), size=64)
                    V.vstore(V.vdiv(clamped_amax, quant_max, full_scale, precision='exact'), sf_dense_ub[token_offset, group])
                    V.vstore(V.vdiv(quant_max, clamped_amax, full_scale, precision='exact'), inverse_ub[group])

            S.mem_bar('VST_VLD')
            for group_batch in T.unroll(T.ceildiv(num_groups, 64), explicit=True):
                group_base = group_batch * 64
                vector_base = group_batch * 32
                for vector in T.serial(T.min(32, num_vecs - vector_base)):
                    col = (vector_base + vector) * vec_size
                    # Broadcast each 32-channel inverse to its 32 lanes.
                    inverse_slots = V.vload(
                        inverse_ub[group_base + vector * 2],
                        size=2,
                        group=2,
                        stride=1,
                    )
                    expanded_inverse = V.vbrc(inverse_slots, size=vec_size, group=2)
                    quantized = V.vcvt(
                        V.vmul(V.vload(values_ub[col], size=vec_size), expanded_inverse),
                        'float8_e4m3fn',
                        rounding='R',
                        saturate='NOSAT',
                    )
                    V.vstore(quantized, out_ub[col])

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
                pacc_ub = T.alloc_shared((3, vec_size), T.float32)  # per lane accumulator pacc_ub. use vcadd at the last moment

                for i in T.serial(4):
                    acc_ub[i] = 0
                with T.SimdVF():
                    T.clear(pacc_ub)
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
                            with T.SimdVF():
                                vdiv_precision = 'ftz_true' if use_fast_certificate else 'exact'
                                cert_min, cert_max = compute_swiglu(
                                    xl_ub, xr_ub, values_ub, out_ub, weight, vdiv_precision, amax_ub, use_fast_certificate, pacc_ub, count_clamp
                                )
                                if use_fast_certificate:
                                    V.vstore(V.vcmin(cert_min, V.create_mask(vec_size, size=vec_size)), certificate_ub[token_offset, 0])
                                    V.vstore(V.vcmax(cert_max, V.create_mask(vec_size, size=vec_size)), certificate_ub[token_offset, 1])
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
                with T.SimdVF():
                    for i in T.unroll(3, explicit=True):
                        V.vstore(V.vcvt(V.vcadd(V.vload(pacc_ub[i, 0], size=vec_size), V.create_mask(vec_size, size=vec_size), reassoc=True), 'int32', saturate='NOSAT'), acc_ub[i])

                with T.SimtVF(threads=1):
                    for i in T.serial(4):
                        T.atomic_add(clamped_count[i], T.int64(acc_ub[i]))

    return swiglu_forward_kernel
