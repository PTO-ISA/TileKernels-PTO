import tilelang
from tilelang.ascend import language as T

from tile_kernels.quant.common import CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_swiglu_backward_kernel_parallel(
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
    # Pad transpose UB to avoid unsupported partial-vector PAT_VL masks.
    aligned_num_stored_block_groups = align(num_stored_block_groups, stored_groups_per_vector)

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    x_sf_shape = (num_expanded_tokens, num_groups * 2) if with_quant else (1, 1)
    out_sf_shape = get_sf_shape((num_expanded_tokens, hidden * 2), out_config) if with_quant else (1, 1)
    x_grad_fp8_shape = (num_expanded_tokens, hidden * 2) if with_quant else (1, 1)

    @T.macro
    def get_scale_and_inverse(amax):
        maximum = T.max(amax, out_config.clamp_min_value)
        if out_config.round_sf:
            bits = T.reinterpret(maximum * (1.0 / 448.0), T.uint32)
            exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
            return T.reinterpret(exponent << 23, T.float32), T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
        else:
            return maximum / 448.0, 448.0 / maximum

    @T.macro
    def init_transpose_indices(indices):
        with T.SimdVF(lanes=64):
            T.evaluate(0)

    @T.macro
    def transpose_sf(src, dst, indices, source_col):
        with T.SimdVF(lanes=64):
            for group, row in T.Parallel(num_stored_block_groups, block_m):
                dst[group, row] = src[row, source_col + group]

    @T.prim_func
    def swiglu_backward_kernel_parallel(
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
                # Reused for expanded input scales, amax, and inverse scales.
                x_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor * 4,), T.float32)
                y_quant_scratch_ub = T.alloc_shared((sf_block_stride * pack_factor * 4,), T.float32)
                # Packed SF is written as bytes and transposed through a uint16 view.
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
                            valid_token_f32 = T.cast(expert_id < num_experts and token_id < valid_token_end, T.float32)

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

                        vf_weight = effective_weight
                        vf_valid = valid_token_f32
                        with T.SimdVF(lanes=64):
                            if with_weight:
                                weight_products = T.alloc_shared((block_k,), T.float32)
                                weight_total = T.alloc_shared((1,), T.float32)
                            for j in T.Parallel(block_k):
                                left = T.float32(x_ub[j])
                                right = T.float32(y_ub[j])
                                if with_quant:
                                    left = left * x_sf_ub[j // 32]
                                    right = right * y_sf_ub[j // 32]
                                if use_clamp:
                                    xc = T.min(left, clamp_value)
                                    yc = T.min(T.max(right, -clamp_value), clamp_value)
                                else:
                                    xc = left
                                    yc = right
                                denominator = 1.0 + T.exp(-xc)
                                sigmoid = 1.0 / denominator
                                grad = T.float32(grad_ub[j])
                                grad_ws = (grad * vf_weight) * sigmoid
                                dx = (grad_ws * yc) * (1.0 + xc * (1.0 - sigmoid))
                                dy = grad_ws * xc
                                if use_clamp:
                                    dx = T.if_then_else(left > clamp_value, 0.0, dx)
                                    dy = T.if_then_else(T.abs(right) > clamp_value, 0.0, dy)
                                x_grad_ub[j] = T.cast(dx, out_dtype)
                                y_grad_ub[j] = T.cast(dy, out_dtype)
                                if with_quant:
                                    x_grad_f32_ub[j] = dx
                                    y_grad_f32_ub[j] = dy
                                if with_weight or do_recompute:
                                    act = (xc / denominator) * yc
                                if with_weight:
                                    weight_products[j] = (grad * act) * vf_valid
                                if do_recompute:
                                    act_out_ub[j] = T.cast(act * vf_weight, out_dtype)
                            if with_weight:
                                T.reduce_sum(weight_products, weight_total, dim=0)
                                for point in T.Parallel(1):
                                    accumulated = weight_total[point] + T.if_then_else(hidden_block == 0, 0.0, weight_acc_ub[row, point])
                                    weight_acc_ub[row, point] = accumulated
                                    if hidden_block + 1 == num_hidden_blocks:
                                        weight_out_ub[point] = accumulated * (routed_scaling_factor if with_routed_scaling else 1.0)
                            if with_quant:
                                for group in T.Parallel(num_block_groups):
                                    x_max = T.alloc_var(T.float32, init=0.0)
                                    y_max = T.alloc_var(T.float32, init=0.0)
                                    for k in T.serial(32):
                                        x_max = T.max(x_max, T.abs(x_grad_f32_ub[group * 32 + k]))
                                        y_max = T.max(y_max, T.abs(y_grad_f32_ub[group * 32 + k]))
                                    x_scale, x_inv = get_scale_and_inverse(x_max)
                                    y_scale, y_inv = get_scale_and_inverse(y_max)
                                    x_quant_scratch_ub[group] = x_inv
                                    y_quant_scratch_ub[group] = y_inv
                                    if out_config.use_packed_ue8m0:
                                        sf_dense_u8[row, sf_block_offset * 2 + group] = T.cast(
                                            T.reinterpret(x_scale * vf_valid, T.uint32) >> 23, T.uint8
                                        )
                                        sf_dense_u8[row, sf_half_stride * 2 + sf_block_offset * 2 + group] = T.cast(
                                            T.reinterpret(y_scale * vf_valid, T.uint32) >> 23, T.uint8
                                        )
                                    else:
                                        sf_dense_ub[row, sf_block_offset + group] = x_scale * vf_valid
                                        sf_dense_ub[row, sf_half_stride + sf_block_offset + group] = y_scale * vf_valid
                                for j in T.Parallel(block_k):
                                    x_grad_fp8_ub[j] = T.cast(
                                        T.if_then_else(vf_valid > 0.0, x_grad_f32_ub[j] * x_quant_scratch_ub[j // 32], 0.0), out_config.dtype
                                    )
                                    y_grad_fp8_ub[j] = T.cast(
                                        T.if_then_else(vf_valid > 0.0, y_grad_f32_ub[j] * y_quant_scratch_ub[j // 32], 0.0), out_config.dtype
                                    )

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
                        if out_config.use_tma_aligned_col_major_sf:
                            output_sf = T.Tensor((num_stored_groups * 2, num_expanded_tokens), T.uint16, x_grad_fp8_sf.data)
                        else:
                            output_sf = T.Tensor((num_expanded_tokens, num_stored_groups * 2), T.uint16, x_grad_fp8_sf.data)
                    else:
                        output_sf = x_grad_fp8_sf

                    for output_block in range(num_hidden_blocks):
                        sf_source_offset = output_block * sf_block_stride
                        sf_output_offset = output_block * num_stored_block_groups
                        if out_config.use_tma_aligned_col_major_sf:
                            transpose_sf(sf_dense_ub, x_sf_out_ub, transpose_indices_ub, sf_source_offset)
                            T.copy(x_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[sf_output_offset, token_base])
                            transpose_sf(sf_dense_ub, y_sf_out_ub, transpose_indices_ub, sf_half_stride + sf_source_offset)
                            T.copy(y_sf_out_ub[0:num_stored_block_groups, 0:block_m], output_sf[num_stored_groups + sf_output_offset, token_base])
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

    return swiglu_backward_kernel_parallel
