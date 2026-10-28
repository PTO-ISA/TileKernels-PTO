import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_swiglu_backward_kernel_vmi(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    use_clamp: bool,
    x_dtype: T.dtype,
    act_x_grad_dtype: T.dtype,
    out_config: CastOutputConfig,
    do_recompute: bool,
    num_vec_cores: int,
):
    num_stages = 2

    with_quant = out_config.with_sf
    out_dtype = T.bfloat16 if with_quant else out_config.dtype

    if with_quant:
        assert x_dtype == T.float8_e4m3fn
        assert out_config.dtype == T.float8_e4m3fn
        assert out_config.sf_block == (1, 32)
        assert not out_config.use_packed_ue8m0 or out_config.round_sf
    else:
        assert x_dtype in (T.bfloat16, T.float32)
        assert out_dtype in (T.bfloat16, T.float32)
    assert hidden % 128 == 0

    # Keep each SF column at the 32-byte DMA granularity.
    block_m = (16 if out_config.use_packed_ue8m0 else 8) if with_quant else 1
    if num_experts:
        assert alignment % block_m == 0, f'Ascend swiglu backward requires alignment % {block_m} == 0, got {alignment}'

    vec_size = 64
    block_k = min(hidden, 4096)
    while hidden % block_k != 0:
        block_k -= vec_size
    assert block_k >= 128 and hidden % block_k == 0

    pack_factor = get_packed_ue8m0_pack_factor() if with_quant and out_config.use_packed_ue8m0 else 1
    num_hidden_blocks = hidden // block_k
    num_groups = hidden // 32 if with_quant else 0
    num_block_groups = block_k // 32 if with_quant else 0
    num_stored_groups = num_groups // pack_factor
    num_stored_block_groups = num_block_groups // pack_factor
    # Pad each hidden block to one full vector store.
    sf_block_stride = align(num_stored_block_groups, vec_size // pack_factor) if with_quant else 0
    sf_half_stride = sf_block_stride * num_hidden_blocks
    sf_row_stride = sf_half_stride * 2
    num_scale_vectors = (num_block_groups + vec_size - 1) // vec_size if with_quant else 0
    sf_storage_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.float32
    stored_groups_per_vector = vec_size * pack_factor // block_m
    # Pad the transpose UB so every VMI gather/store uses a full vector width.
    aligned_num_stored_block_groups = align(num_stored_block_groups, stored_groups_per_vector)

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    x_sf_shape = (num_expanded_tokens, num_groups * 2) if with_quant else (1, 1)
    out_sf_shape = get_sf_shape((num_expanded_tokens, hidden * 2), out_config) if with_quant else (1, 1)
    x_grad_fp8_shape = (num_expanded_tokens, hidden * 2) if with_quant else (1, 1)

    @T.macro
    def vdiv_exact_0ulp(a, b):
        """Correct a / b to 0 ULP with ASC backward ``vdiv_0ulp`` semantics.

        ASC backward uses the ``le`` tie-break, unlike both the built-in
        ``V.vdiv(precision='exact')`` expansion and the forward kernel, which
        use ``lt``. Keep ``le`` for bitwise parity with
        swiglu_backward_asc.py; do not unify the two helpers.

        The built-in expansion and this version perform the same neighbor
        search; the omitted denominator prescaling is a power of two and does
        not change residual ordering in this operand domain.
        """
        full = V.create_mask(vec_size, size=vec_size)
        z = V.vdiv(a, b)
        y = V.vneg(b)
        r = V.vmula(a, z, y, full)
        z_bits = V.vinterpret_cast(z, 'int32')
        z_pre = V.vinterpret_cast(V.vadds(z_bits, T.int32(-1), full), 'float32')
        z_next = V.vinterpret_cast(V.vadds(z_bits, T.int32(1), full), 'float32')
        r_pre = V.vmula(a, z_pre, y, full)
        r_next = V.vmula(a, z_next, y, full)
        r = V.vabs(r)
        r_pre = V.vabs(r_pre)
        r_next = V.vabs(r_next)
        take_pre = V.vcmp(r_pre, r, full, 'le')
        best_r = V.vsel(take_pre, r_pre, r)
        best_z = V.vsel(take_pre, z_pre, z)
        take_next = V.vcmp(r_next, best_r, full, 'lt')
        corrected = V.vsel(take_next, z_next, best_z)
        is_zero = V.vcmps(z, T.float32(0.0), full, 'eq')
        return V.vsel(is_zero, z, corrected)

    @T.macro
    def sigmoid_vdiv_exact_0ulp(a, b):
        """Correct a / b for sigmoid to 0 ULP (the ASC sigmoid algorithm).

        Starts from the hardware quotient and checks the adjacent representable,
        matching the ASC ``sigmoid_vdiv_0ulp`` algorithm. This intentionally
        does not call ``V.vdiv(precision='exact')``; the ASC source uses this
        cheaper specialized correction for the sigmoid. PTO ``vmi.vsel`` only
        accepts vector-register operands, so the predicate xor is materialized
        as int32 vectors.
        """
        full = V.create_mask(vec_size, size=vec_size)
        one_i32 = V.vbrc(T.int32(1), size=vec_size)
        zero_i32 = V.vbrc(T.int32(0), size=vec_size)
        z = V.vdiv(a, b)
        neg_b = V.vneg(b)

        residual = V.vmula(a, z, neg_b, full)

        residual_bits = V.vinterpret_cast(residual, 'int32')
        sign_mask = V.vshrs(residual_bits, T.int32(31), full)
        use_bits_minus = V.vcmps(residual_bits, T.int32(0), full, 'lt')
        delta = V.vor(sign_mask, one_i32)
        adjacent = V.vinterpret_cast(V.vadd(V.vinterpret_cast(z, 'int32'), delta), 'float32')

        adjacent_residual = V.vmula(a, adjacent, neg_b, full)
        residual_sum = V.vadd(residual, adjacent_residual)
        take_bits_plus = V.vcmps(residual_sum, T.float32(0.0), full, 'gt')
        is_nonzero = V.vcmps(z, T.float32(0.0), full, 'ne')

        take_plus_i = V.vsel(take_bits_plus, one_i32, zero_i32)
        use_minus_i = V.vsel(use_bits_minus, one_i32, zero_i32)
        nonzero_i = V.vsel(is_nonzero, one_i32, zero_i32)
        take_adjacent_i = V.vand(V.vxor(take_plus_i, use_minus_i), nonzero_i)
        take_adjacent = V.vcmps(take_adjacent_i, T.int32(0), full, 'ne')
        return V.vsel(take_adjacent, adjacent, z)

    @T.macro
    def get_scale_and_inverse(amax):
        # Compute one vector of output scales and inverse scales from per-group amax.
        full = V.create_mask(vec_size, size=vec_size)
        clamped_amax = V.vmaxs(amax, T.float32(out_config.clamp_min_value), full)
        if out_config.round_sf:
            exponent_mask = V.vbrc(T.uint32(0x7F800000), size=vec_size)
            exponent_carry = V.vbrc(T.uint32(0x001FFFFF), size=vec_size)
            inverse_bias = V.vbrc(T.uint32(0x83000000), size=vec_size)
            scale_bias = V.vbrc(T.uint32(0x04000000), size=vec_size)
            exponent = V.vand(V.vadd(V.vinterpret_cast(clamped_amax, 'uint32'), exponent_carry), exponent_mask)
            scale = V.vinterpret_cast(V.vsub(exponent, scale_bias), 'float32')
            inverse = V.vinterpret_cast(V.vsub(inverse_bias, exponent), 'float32')
            return scale, inverse
        quant_max = V.vbrc(T.float32(448.0), size=vec_size)
        return (
            V.vdiv(clamped_amax, quant_max, full, precision='exact'),
            V.vdiv(quant_max, clamped_amax, full, precision='exact'),
        )

    @T.macro
    def init_transpose_indices(indices):
        # Precompute gather indices shared by every token block when converting
        # dense token-major scales to the output layout.
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                lane = V.vci(T.int16(0), size=128)
                row = V.vand(V.vinterpret_cast(lane, 'uint16'), V.vbrc(T.uint16(block_m - 1), size=128))
                group = V.vinterpret_cast(V.vshr(lane, V.vbrc(T.int16(block_m.bit_length() - 1), size=128)), 'uint16')
                source = V.vadd(V.vmul(row, V.vbrc(T.uint16(sf_row_stride), size=128)), group)
                V.vstore(source, indices[0])
            else:
                lane = V.vci(T.int32(0), size=64)
                row = V.vand(V.vinterpret_cast(lane, 'uint32'), V.vbrc(T.uint32(block_m - 1), size=64))
                group = V.vinterpret_cast(V.vshr(lane, V.vbrc(T.int32(block_m.bit_length() - 1), size=64)), 'uint32')
                source = V.vadd(V.vmul(row, V.vbrc(T.uint32(sf_row_stride), size=64)), group)
                V.vstore(source, indices[0])

    @T.macro
    def transpose_sf(src, dst, indices, source_col):
        # Transpose one hidden block of dense scales into the stored layout.
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                vector_lanes = 128
            else:
                vector_lanes = 64
            source_indices = V.vload(indices[0], size=vector_lanes)
            full_mask = V.create_mask(vector_lanes, size=vector_lanes)
            for output_vector in T.unroll((num_stored_block_groups + stored_groups_per_vector - 1) // stored_groups_per_vector, explicit=True):
                group_base = output_vector * stored_groups_per_vector
                values = V.vgather(src[0, source_col + group_base], source_indices, full_mask)
                V.vstore(values, dst[group_base, 0])

    @T.prim_func
    def swiglu_backward_kernel_vmi(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), x_dtype],
        x_sf: T.Tensor[x_sf_shape, T.float32],
        act_x_grad: T.Tensor[(num_expanded_tokens, hidden), act_x_grad_dtype],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        psum_num_tokens_per_expert: T.Tensor[(num_experts,), T.int32],
        out: T.Tensor[(num_expanded_tokens, hidden), out_dtype],
        x_grad_fp8: T.Tensor[x_grad_fp8_shape, out_config.dtype],
        x_grad_fp8_sf: T.StridedTensor[out_sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
        x_grad: T.Tensor[(num_expanded_tokens, hidden * 2), out_dtype],
        weight_grad: T.Tensor[(num_expanded_tokens,), T.float32],
        clamp_value: T.float32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_k,), x_dtype)
            y_ub = T.alloc_shared((block_k,), x_dtype)
            grad_ub = T.alloc_shared((block_k,), act_x_grad_dtype)
            x_grad_ub = T.alloc_shared((block_k,), out_dtype)
            y_grad_ub = T.alloc_shared((block_k,), out_dtype)
            if with_quant:
                x_sf_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                y_sf_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                x_grad_f32_ub = T.alloc_shared((block_k,), T.float32)
                y_grad_f32_ub = T.alloc_shared((block_k,), T.float32)
                x_grad_fp8_ub = T.alloc_shared((block_k,), T.float8_e4m3fn)
                y_grad_fp8_ub = T.alloc_shared((block_k,), T.float8_e4m3fn)
                # Reused for compact amax and inverse scales (one value per 32 channels).
                # VMI overwrites in place, so no fourfold expansion is needed.
                x_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                y_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor,), T.float32)
                # Packed SF is stored as uint16 words: quantization writes bytes
                # through sf_dense_u8, and the transpose reads the same buffer as uint16.
                sf_dense_ub = T.alloc_shared((block_m, sf_row_stride), sf_storage_dtype)
                if out_config.use_packed_ue8m0:
                    sf_dense_u8 = T.Tensor((block_m, sf_row_stride * 2), T.uint8, sf_dense_ub.data)
                if out_config.use_tma_aligned_col_major_sf:
                    x_sf_out_ub = T.alloc_shared((aligned_num_stored_block_groups, block_m), sf_storage_dtype)
                    y_sf_out_ub = T.alloc_shared((aligned_num_stored_block_groups, block_m), sf_storage_dtype)
                    transpose_indices_ub = T.alloc_shared((vec_size * pack_factor,), T.uint16 if out_config.use_packed_ue8m0 else T.uint32)

            if with_weight:
                weight_acc_ub = T.alloc_shared((block_m, vec_size), T.float32)
                weight_out_ub = T.alloc_shared((vec_size,), T.float32)
            if do_recompute:
                act_out_ub = T.alloc_shared((block_k,), out_dtype)
            if num_experts:
                psum_ub = T.alloc_shared((num_experts,), T.int32)

            versions = {
                x_ub: num_stages,
                y_ub: num_stages,
                grad_ub: num_stages,
                x_grad_ub: num_stages,
                y_grad_ub: num_stages,
            }
            if with_quant:
                versions.update(
                    {
                        x_sf_ub: num_stages,
                        y_sf_ub: num_stages,
                        # Vector-local scratch does not need MTE overlap versions.
                        x_grad_f32_ub: 1,
                        y_grad_f32_ub: 1,
                        x_grad_fp8_ub: num_stages,
                        y_grad_fp8_ub: num_stages,
                        x_quant_scratch_ub: 1,
                        y_quant_scratch_ub: 1,
                        sf_dense_ub: 1,
                    }
                )
            if with_quant and out_config.use_tma_aligned_col_major_sf:
                versions[x_sf_out_ub] = 1
                versions[y_sf_out_ub] = 1
                versions[transpose_indices_ub] = 1
            if with_weight:
                versions[weight_acc_ub] = 1
                versions[weight_out_ub] = num_stages
            if do_recompute:
                versions[act_out_ub] = num_stages
            if num_experts:
                versions[psum_ub] = 1
            T.annotate_buffer_versions(versions)

            if with_quant and out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)
            if num_experts:
                T.copy(psum_num_tokens_per_expert, psum_ub[:num_experts])
                expert_id = T.alloc_var(T.int32, init=0)
                valid_token_end = T.alloc_var(T.int32, init=0)

            num_token_blocks = num_expanded_tokens // block_m
            blocks_per_core = num_token_blocks // num_vec_cores
            num_long_cores = num_token_blocks % num_vec_cores
            # Assign contiguous block ranges per core to spread concurrent GM traffic across L2 home paths.
            core_block_start = core_id * blocks_per_core + T.min(core_id, num_long_cores)

            for scheduler_pid in T.Persistent(
                [num_token_blocks],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                wave = scheduler_pid // num_vec_cores
                token_block = core_block_start + wave
                token_base = token_block * block_m
                if num_experts:
                    while expert_id < num_experts and token_base >= align(psum_ub[expert_id], alignment):
                        expert_id += 1
                    if expert_id < num_experts:
                        valid_token_end = psum_ub[expert_id]
                    else:
                        valid_token_end = 0

                for hidden_block in range(num_hidden_blocks):
                    col_offset = hidden_block * block_k
                    sf_offset = hidden_block * num_block_groups
                    sf_block_offset = hidden_block * sf_block_stride

                    for row in T.serial(block_m):
                        token_id = token_base + row
                        valid_token_f32 = T.alloc_var(T.float32, init=1.0)
                        if num_experts:
                            # PTO codegen does not support scalar bool -> float32 casts.
                            valid_token_f32 = T.float32(0.0)
                            if expert_id < num_experts and token_id < valid_token_end:
                                valid_token_f32 = T.float32(1.0)

                        effective_weight = T.alloc_var(T.float32, init=valid_token_f32)
                        if with_weight:
                            effective_weight = topk_weights[token_id] * valid_token_f32
                            if with_routed_scaling:
                                effective_weight = effective_weight * routed_scaling_factor

                        T.copy(x[token_id, col_offset : col_offset + block_k], x_ub)
                        T.copy(x[token_id, hidden + col_offset : hidden + col_offset + block_k], y_ub)
                        if with_quant:
                            T.copy(x_sf[token_id, sf_offset : sf_offset + num_block_groups], x_sf_ub[:num_block_groups])
                            T.copy(x_sf[token_id, num_groups + sf_offset : num_groups + sf_offset + num_block_groups], y_sf_ub[:num_block_groups])
                        T.copy(act_x_grad[token_id, col_offset : col_offset + block_k], grad_ub)

                        with T.SimdVF():
                            full = V.create_mask(vec_size, size=vec_size)
                            zeros = V.vbrc(T.float32(0.0), size=vec_size)
                            ones = V.vbrc(T.float32(1.0), size=vec_size)
                            if with_weight:
                                weight_acc = V.alloc_var(T.float32, size=vec_size)
                                # hidden_block is a runtime TIR loop variable; use an
                                # if statement so PTO does not see a vector select.
                                if hidden_block == 0:
                                    weight_acc = zeros
                                else:
                                    weight_acc = V.vload(weight_acc_ub[row, 0], size=vec_size)

                            # VMI exposes separate widen/narrow conversions. PTO
                            # codegen already supports fused T.simd dist='UNPK/PK'
                            # forms, but #1616 forbids using ASC physical layouts
                            # here; a VMI widen/narrow load/store is the upstream
                            # feature request described in the PR.
                            for vector in T.serial(block_k // vec_size):
                                col = vector * vec_size
                                if with_quant:
                                    x_scale = V.vbrc(V.vload(x_sf_ub[vector * 2], size=2, group=2, stride=1), size=vec_size, group=2)
                                    y_scale = V.vbrc(V.vload(y_sf_ub[vector * 2], size=2, group=2, stride=1), size=vec_size, group=2)
                                    x_raw = V.vcvt(V.vload(x_ub[col], size=vec_size), 'float32')
                                    y_raw = V.vcvt(V.vload(y_ub[col], size=vec_size), 'float32')
                                    x_value = V.vmul(x_raw, x_scale)
                                    y_value = V.vmul(y_raw, y_scale)
                                elif x_dtype == T.bfloat16:
                                    x_value = V.vcvt(V.vload(x_ub[col], size=vec_size), 'float32')
                                    y_value = V.vcvt(V.vload(y_ub[col], size=vec_size), 'float32')
                                else:
                                    x_value = V.vload(x_ub[col], size=vec_size)
                                    y_value = V.vload(y_ub[col], size=vec_size)
                                if act_x_grad_dtype == T.bfloat16:
                                    grad = V.vcvt(V.vload(grad_ub[col], size=vec_size), 'float32')
                                else:
                                    grad = V.vload(grad_ub[col], size=vec_size)

                                if use_clamp:
                                    is_clamped_x = V.vcmps(x_value, T.float32(clamp_value), full, 'gt')
                                    is_clamped_y = V.vcmps(V.vabs(y_value), T.float32(clamp_value), full, 'gt')
                                    x_compute_value = V.vmins(x_value, T.float32(clamp_value), full)
                                    y_compute_value = V.vmins(V.vmaxs(y_value, T.float32(-clamp_value), full), T.float32(clamp_value), full)
                                else:
                                    x_compute_value = x_value
                                    y_compute_value = y_value

                                sigmoid_denominator = V.vadds(V.vexpdif(zeros, x_compute_value, full), T.float32(1.0), full)
                                sigmoid = sigmoid_vdiv_exact_0ulp(ones, sigmoid_denominator)
                                if with_weight or do_recompute:
                                    # Keep a separate 0-ULP division for the required FP32 rounding.
                                    act_value = V.vmul(vdiv_exact_0ulp(x_compute_value, sigmoid_denominator), y_compute_value)
                                if with_weight:
                                    weight_contribution = V.vmul(grad, act_value)
                                    weight_value = V.vmuls(weight_contribution, valid_token_f32, full) if num_experts else weight_contribution
                                    weight_acc = V.vadd(weight_acc, weight_value)
                                grad_ws = V.vmul(V.vmuls(grad, effective_weight, full), sigmoid)
                                inner = V.vadd(ones, V.vmul(x_compute_value, V.vsub(ones, sigmoid)))
                                x_grad_unclamped = V.vmul(V.vmul(grad_ws, y_compute_value), inner)
                                y_grad_unclamped = V.vmul(grad_ws, x_compute_value)
                                x_grad_value = V.vsel(is_clamped_x, zeros, x_grad_unclamped) if use_clamp else x_grad_unclamped
                                y_grad_value = V.vsel(is_clamped_y, zeros, y_grad_unclamped) if use_clamp else y_grad_unclamped
                                if not with_quant:
                                    if out_dtype == T.bfloat16:
                                        # Explicit NOSAT: same bug A/B rationale as
                                        # the FP8 output; do not rely on PTODSL's
                                        # implicit SAT default for narrowing.
                                        V.vstore(V.vcvt(x_grad_value, 'bfloat16', saturate='NOSAT'), x_grad_ub[col])
                                        V.vstore(V.vcvt(y_grad_value, 'bfloat16', saturate='NOSAT'), y_grad_ub[col])
                                    else:
                                        V.vstore(x_grad_value, x_grad_ub[col])
                                        V.vstore(y_grad_value, y_grad_ub[col])
                                else:
                                    V.vstore(x_grad_value, x_grad_f32_ub[col])
                                    V.vstore(y_grad_value, y_grad_f32_ub[col])
                                    x_abs = V.vabs(x_grad_value)
                                    y_abs = V.vabs(y_grad_value)
                                    V.vstore(V.vcmax(x_abs, full, group=2), x_quant_scratch_ub[vector * 2])
                                    V.vstore(V.vcmax(y_abs, full, group=2), y_quant_scratch_ub[vector * 2])
                                if do_recompute:
                                    act_out_value = V.vmuls(act_value, effective_weight, full)
                                    if out_dtype == T.bfloat16:
                                        # Explicit NOSAT, same bug A/B rationale.
                                        V.vstore(V.vcvt(act_out_value, 'bfloat16', saturate='NOSAT'), act_out_ub[col])
                                    else:
                                        V.vstore(act_out_value, act_out_ub[col])

                            if with_weight:
                                if hidden_block + 1 == num_hidden_blocks:
                                    # ASC S.vcadd is a pairwise reduction.
                                    # PTODSL only accepts reassoc=True here, so
                                    # weight_grad can differ in the last mantissa
                                    # bits; it is validated at the repo's 1e-4.
                                    one_mask = V.create_mask(1, size=1)
                                    reduced_weight_sum = V.vcadd(weight_acc, full, reassoc=True)
                                    weight_sum = (
                                        V.vmuls(reduced_weight_sum, T.float32(routed_scaling_factor), one_mask)
                                        if with_routed_scaling
                                        else reduced_weight_sum
                                    )
                                    V.vstore(weight_sum, weight_out_ub[0], group=1, stride=1)
                                else:
                                    V.vstore(weight_acc, weight_acc_ub[row, 0])

                            if with_quant:
                                S.mem_bar('VST_VLD')
                                if num_experts:
                                    valid_token_vec = V.vbrc(valid_token_f32, size=vec_size)
                                # Each iteration reads its amax before overwriting
                                # the same slot with the inverse, so order is free.
                                for scale_vector in range(num_scale_vectors):
                                    scale_offset = scale_vector * vec_size
                                    x_scale_value, x_inv = get_scale_and_inverse(V.vload(x_quant_scratch_ub[scale_offset], size=vec_size))
                                    y_scale_value, y_inv = get_scale_and_inverse(V.vload(y_quant_scratch_ub[scale_offset], size=vec_size))
                                    V.vstore(x_inv, x_quant_scratch_ub[scale_offset])
                                    V.vstore(y_inv, y_quant_scratch_ub[scale_offset])
                                    x_scale_out = V.vmul(x_scale_value, valid_token_vec) if num_experts else x_scale_value
                                    y_scale_out = V.vmul(y_scale_value, valid_token_vec) if num_experts else y_scale_value
                                    if out_config.use_packed_ue8m0:
                                        x_exp = V.vshrs(V.vinterpret_cast(x_scale_out, 'uint32'), T.uint32(23), full)
                                        y_exp = V.vshrs(V.vinterpret_cast(y_scale_out, 'uint32'), T.uint32(23), full)
                                        # int-narrow (uint32 -> uint8): NOSAT is
                                        # exactly ASC's reinterpret-truncation;
                                        # unrelated to the FP8 output saturate
                                        # decision below. Spell it out because PTODSL
                                        # otherwise defaults to SAT for narrowing,
                                        # which differs for exponents >= 256
                                        # (amax = inf).
                                        V.vstore(
                                            V.vcvt(x_exp, 'uint8', saturate='NOSAT'),
                                            sf_dense_u8[row, sf_block_offset * 2 + scale_offset],
                                        )
                                        V.vstore(
                                            V.vcvt(y_exp, 'uint8', saturate='NOSAT'),
                                            sf_dense_u8[row, sf_half_stride * 2 + sf_block_offset * 2 + scale_offset],
                                        )
                                    else:
                                        V.vstore(x_scale_out, sf_dense_ub[row, sf_block_offset + scale_offset])
                                        V.vstore(y_scale_out, sf_dense_ub[row, sf_half_stride + sf_block_offset + scale_offset])

                                S.mem_bar('VST_VLD')
                                if num_experts:
                                    valid_token_mask = V.vcmps(valid_token_vec, T.float32(0.0), full, 'gt')
                                for vector in T.serial(block_k // vec_size):
                                    col = vector * vec_size
                                    x_inv_vec = V.vbrc(V.vload(x_quant_scratch_ub[vector * 2], size=2, group=2, stride=1), size=vec_size, group=2)
                                    y_inv_vec = V.vbrc(V.vload(y_quant_scratch_ub[vector * 2], size=2, group=2, stride=1), size=vec_size, group=2)
                                    x_grad_quant_value = V.vload(x_grad_f32_ub[col], size=vec_size)
                                    y_grad_quant_value = V.vload(y_grad_f32_ub[col], size=vec_size)
                                    x_scaled_value = V.vmul(x_grad_quant_value, x_inv_vec)
                                    y_scaled_value = V.vmul(y_grad_quant_value, y_inv_vec)
                                    # Explicit NOSAT, same bug A/B rationale.
                                    V.vstore(V.vcvt(x_grad_quant_value, 'bfloat16', saturate='NOSAT'), x_grad_ub[col])
                                    V.vstore(V.vcvt(y_grad_quant_value, 'bfloat16', saturate='NOSAT'), y_grad_ub[col])
                                    x_scaled = V.vsel(valid_token_mask, x_scaled_value, zeros) if num_experts else x_scaled_value
                                    y_scaled = V.vsel(valid_token_mask, y_scaled_value, zeros) if num_experts else y_scaled_value
                                    # FP8 output saturate: explicit NOSAT. ASC's
                                    # RS_ENABLE is a no-op for f32->e4m3 here (bug A),
                                    # while VMI SAT maps NaN to 0x00 (bug B). NOSAT is
                                    # the only mode that both reproduces ASC today and
                                    # preserves NaN; switch back to SAT after bug A is
                                    # fixed upstream. Both bugs: <ISSUE_URL_TBD>
                                    # (draft ready; to be filed), repro_fp8_saturate.py.
                                    # Finite values are
                                    # unaffected: value <= 448*(1+2^-23) << 464.
                                    V.vstore(V.vcvt(x_scaled, 'float8_e4m3fn', rounding='R', saturate='NOSAT'), x_grad_fp8_ub[col])
                                    V.vstore(V.vcvt(y_scaled, 'float8_e4m3fn', rounding='R', saturate='NOSAT'), y_grad_fp8_ub[col])

                        T.copy(x_grad_ub, x_grad[token_id, col_offset : col_offset + block_k])
                        T.copy(y_grad_ub, x_grad[token_id, hidden + col_offset : hidden + col_offset + block_k])
                        if with_quant:
                            T.copy(x_grad_fp8_ub, x_grad_fp8[token_id, col_offset : col_offset + block_k])
                            T.copy(y_grad_fp8_ub, x_grad_fp8[token_id, hidden + col_offset : hidden + col_offset + block_k])
                        if do_recompute:
                            T.copy(act_out_ub, out[token_id, col_offset : col_offset + block_k])
                        if with_weight and hidden_block + 1 == num_hidden_blocks:
                            T.copy(weight_out_ub[:1], weight_grad[token_id : token_id + 1])

                if with_quant:
                    if out_config.use_packed_ue8m0:
                        # PTO's DMA requires matching source/destination element widths.
                        # Write packed SF through uint8 views instead of ASC's uint16 view.
                        # tilelang PR #14 (codex/unify-asc-mte-pointer-lowering) removes
                        # this restriction, but it targets a newer pto-dev and does not
                        # apply to our pinned 5f105e99 snapshot; keep the byte views.
                        if out_config.use_tma_aligned_col_major_sf:
                            output_sf = T.Tensor((num_stored_groups * 2, num_expanded_tokens * 2), T.uint8, x_grad_fp8_sf.data)
                        else:
                            output_sf = T.Tensor((num_expanded_tokens, num_stored_groups * 4), T.uint8, x_grad_fp8_sf.data)
                    else:
                        output_sf = x_grad_fp8_sf

                    for output_block in range(num_hidden_blocks):
                        sf_source_offset = output_block * sf_block_stride
                        sf_output_offset = output_block * num_stored_block_groups
                        if out_config.use_tma_aligned_col_major_sf:
                            if out_config.use_packed_ue8m0:
                                x_sf_out_u8 = T.Tensor((aligned_num_stored_block_groups, block_m * 2), T.uint8, x_sf_out_ub.data)
                                y_sf_out_u8 = T.Tensor((aligned_num_stored_block_groups, block_m * 2), T.uint8, y_sf_out_ub.data)
                                transpose_sf(sf_dense_ub, x_sf_out_ub, transpose_indices_ub, sf_source_offset)
                                T.copy(
                                    x_sf_out_u8[0:num_stored_block_groups, 0 : block_m * 2],
                                    output_sf[sf_output_offset, token_base * 2],
                                )
                                transpose_sf(sf_dense_ub, y_sf_out_ub, transpose_indices_ub, sf_half_stride + sf_source_offset)
                                T.copy(
                                    y_sf_out_u8[0:num_stored_block_groups, 0 : block_m * 2],
                                    output_sf[num_stored_groups + sf_output_offset, token_base * 2],
                                )
                            else:
                                transpose_sf(sf_dense_ub, x_sf_out_ub, transpose_indices_ub, sf_source_offset)
                                T.copy(x_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[sf_output_offset, token_base])
                                transpose_sf(sf_dense_ub, y_sf_out_ub, transpose_indices_ub, sf_half_stride + sf_source_offset)
                                T.copy(y_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[num_stored_groups + sf_output_offset, token_base])
                        else:
                            if out_config.use_packed_ue8m0:
                                T.copy(
                                    sf_dense_u8[0:block_m, sf_source_offset * 2 : (sf_source_offset + num_stored_block_groups) * 2],
                                    output_sf[token_base, sf_output_offset * 2],
                                )
                                T.copy(
                                    sf_dense_u8[
                                        0:block_m,
                                        (sf_half_stride + sf_source_offset) * 2 : (sf_half_stride + sf_source_offset + num_stored_block_groups) * 2,
                                    ],
                                    output_sf[token_base, (num_stored_groups + sf_output_offset) * 2],
                                )
                            else:
                                T.copy(
                                    sf_dense_ub[0:block_m, sf_source_offset : sf_source_offset + num_stored_block_groups],
                                    output_sf[token_base, sf_output_offset],
                                )
                                T.copy(
                                    sf_dense_ub[
                                        0:block_m,
                                        sf_half_stride + sf_source_offset : sf_half_stride + sf_source_offset + num_stored_block_groups,
                                    ],
                                    output_sf[token_base, num_stored_groups + sf_output_offset],
                                )

    return swiglu_backward_kernel_vmi
