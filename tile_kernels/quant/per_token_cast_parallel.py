"""Grouped reduction/scale/cast expressions with ASC scheduling and formats."""

import math

import tilelang
from tilelang.ascend import language as T

from tile_kernels.quant.common import CastInputConfig, CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_per_token_cast_kernel_parallel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    stochastic_cast: bool,
    num_vec_cores: int,
):
    num_per_channels = out_config.sf_block[1]
    assert not stochastic_cast
    assert hidden % 128 == 0, 'Ascend per_token_cast hidden must be a multiple of 128'
    assert num_per_channels == 32, 'Ascend per_token_cast currently only supports 32 channels per scale'
    assert not in_config.with_sf or in_config.sf_block[1] == 32, 'Ascend per_token_cast expects 32 channels per input scale'

    quant_max = 6.0 if out_config.dtype == T.float4_e2m1fn else 448.0
    inv_quant_max = 1.0 / quant_max
    pack_factor = get_packed_ue8m0_pack_factor()

    value_dtype = T.float32
    if (
        hidden % 256 == 0
        and not cast_only
        and out_config.use_packed_ue8m0
        and out_config.round_sf
        and (in_config.dtype == T.bfloat16 or in_config.with_sf and in_config.use_packed_ue8m0)
    ):
        value_dtype = T.bfloat16
    if in_config.dtype == T.float8_e4m3fn and value_dtype == T.bfloat16:
        block_m = 16
        block_k_limit = 2048
    else:
        block_m = 32
        block_k_limit = 1024 if value_dtype == T.bfloat16 else 512
    block_k = math.gcd(block_k_limit, hidden)
    log2_block_m = block_m.bit_length() - 1
    num_stages = 2
    num_groups = block_k // num_per_channels
    num_hidden_tiles = hidden // block_k
    sf_stride = 64

    num_tokens = T.dynamic('num_tokens')
    out_sf_stride = T.dynamic('out_sf_stride')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config) if in_config.with_sf else (1, 1)

    @T.macro
    def compute_scale(amax):
        clamped = T.max(T.float32(amax), out_config.clamp_min_value)
        if not out_config.round_sf:
            return clamped / quant_max, quant_max / clamped
        bits = T.reinterpret(clamped * inv_quant_max, T.uint32)
        exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
        inverse = T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
        if out_config.use_packed_ue8m0:
            return T.cast(exponent, T.uint8), inverse
        return T.reinterpret(exponent << 23, T.float32), inverse

    @T.macro
    def init_transpose_indices(indices):
        # Retain the ASC buffer/DMA interface. Element indexing expresses the
        # transpose directly, so no explicit vector gather table is needed.
        with T.SimdVF(lanes=64):
            for j in T.Parallel(128 if out_config.use_packed_ue8m0 else 64):
                indices[j] = T.cast(j, T.uint16 if out_config.use_packed_ue8m0 else T.uint32)

    @T.macro
    def transpose_output_sf(src, dst, indices):
        with T.SimdVF(lanes=64):
            if out_config.use_packed_ue8m0:
                for i, j in T.Parallel(num_groups // pack_factor, block_m * pack_factor):
                    dst[i, j] = src[j // pack_factor, i * pack_factor + j % pack_factor]
            else:
                for i, j in T.Parallel(num_groups, block_m):
                    dst[i, j] = src[j, i]

    @T.macro
    def dequantize_input(x_ub, scales_ub, out_ub):
        with T.SimdVF(lanes=64):
            for i, j in T.Parallel(block_m, block_k):
                group = j // num_per_channels
                if in_config.use_packed_ue8m0:
                    packed = T.cast(scales_ub[group // pack_factor], T.uint32)
                    exponent = (packed >> ((group % pack_factor) * 8)) & T.uint32(255)
                    scale = T.reinterpret(exponent << 23, T.float32)
                else:
                    scale = scales_ub[group]
                out_ub[i, j] = T.cast(T.float32(x_ub[i, j]) * scale, value_dtype)

    @T.macro
    def quantize_values(x_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub):
        with T.SimdVF(lanes=64):
            if not cast_only:
                values = T.reshape(x_ub, (block_m, num_groups, num_per_channels))
                T.reduce_absmax(values, amax_ub[:, 0:num_groups], dim=2)
                for i, j in T.Parallel(block_m, num_groups):
                    scale, inverse = compute_scale(amax_ub[i, j])
                    sf_dense_ub[i, j] = scale
                    sf_inv_ub[i, j] = T.cast(inverse, value_dtype)
            else:
                for i, j in T.Parallel(block_m, num_groups):
                    sf_inv_ub[i, j] = T.cast(1.0 / sf_dense_ub[i, j], value_dtype)
            if not sf_only:
                for i, j in T.Parallel(block_m, block_k):
                    scaled = T.float32(x_ub[i, j]) * T.float32(sf_inv_ub[i, j // num_per_channels])
                    if value_dtype == T.bfloat16:
                        # Preserve the ASC BF16 multiply rounding before FP8/FP4.
                        scaled = T.float32(T.cast(scaled, T.bfloat16))
                    out_ub[i, j] = T.cast(scaled, out_config.dtype)

    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.Tensor[x_sf_shape, in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages}
            if not sf_only:
                versions[out_ub] = num_stages

            if in_config.with_sf:
                dequant_ub = T.alloc_shared((block_m, block_k), value_dtype)
                if in_config.use_packed_ue8m0:
                    sf_in_ub = T.alloc_shared((align(num_groups, 32) // pack_factor,), T.uint16)
                else:
                    sf_in_ub = T.alloc_shared((align(num_groups, 8),), T.float32)
                versions[sf_in_ub] = num_stages
                versions[dequant_ub] = 1
            values_ub = dequant_ub if in_config.with_sf else x_ub

            sf_dense_ub = T.alloc_shared((block_m, sf_stride), T.uint8 if out_config.use_packed_ue8m0 else out_config.sf_dtype)
            if out_config.use_tma_aligned_col_major_sf:
                sf_out_ub = T.alloc_shared(
                    (num_groups // pack_factor, block_m * pack_factor) if out_config.use_packed_ue8m0 else (num_groups, block_m),
                    out_config.sf_dtype,
                )
                transpose_index_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
                transpose_index_lanes = 128 if out_config.use_packed_ue8m0 else 64
                transpose_indices_ub = T.alloc_shared((transpose_index_lanes,), transpose_index_dtype)
            else:
                sf_out_ub = sf_dense_ub

            amax_ub = T.alloc_shared((block_m, sf_stride), T.float32)
            scale_inv_ub = T.alloc_shared((block_m, sf_stride), value_dtype)
            if not in_config.with_sf:
                if not cast_only:
                    versions[amax_ub] = num_stages
                versions[scale_inv_ub] = num_stages
                if out_config.use_tma_aligned_col_major_sf:
                    versions[sf_dense_ub] = num_stages
            T.annotate_buffer_versions(versions)

            if out_config.use_tma_aligned_col_major_sf:
                init_transpose_indices(transpose_indices_ub)

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), num_hidden_tiles],
                num_vec_cores,
                core_id,
                # Full last dimension: keeps the traversal row-major (token outer,
                # hidden tile inner). A smaller group size makes the realigned
                # PersistentFor walk the domain column-major, which costs DMA
                # locality on the wide shapes.
                group_size=num_hidden_tiles,
                num_stages=num_stages,
            ):
                col_offset = pid_hidden * block_k
                group_offset = pid_hidden * num_groups

                T.copy(x[pid_token * block_m, col_offset], x_ub)

                if in_config.with_sf:
                    input_sf_row = pid_token * block_m // in_config.sf_block[0]
                    if in_config.use_tma_aligned_col_major_sf:
                        if in_config.use_packed_ue8m0:
                            sf_in_matrix = T.Tensor((align(num_groups, 32) // pack_factor, pack_factor), T.uint8, sf_in_ub.data)
                            T.copy(
                                x_sf[group_offset // pack_factor, input_sf_row * pack_factor],
                                sf_in_matrix[0 : num_groups // pack_factor, 0:pack_factor],
                            )
                        else:
                            sf_in_matrix = T.Tensor((align(num_groups, 8), 1), T.float32, sf_in_ub.data)
                            T.copy(
                                x_sf[group_offset : group_offset + num_groups, input_sf_row : input_sf_row + 1],
                                sf_in_matrix[0:num_groups, 0:1],
                            )
                    else:
                        sf_in_flat = T.Tensor((align(num_groups, 32),), T.uint8, sf_in_ub.data) if in_config.use_packed_ue8m0 else sf_in_ub
                        T.copy(x_sf[input_sf_row, group_offset], sf_in_flat[0:num_groups])
                    dequantize_input(x_ub, sf_in_ub, dequant_ub)

                if cast_only:
                    T.copy(out_sf[pid_token * block_m, group_offset], sf_out_ub[0:block_m, 0:num_groups])

                if value_dtype == T.bfloat16:
                    quantize_values(values_ub, amax_ub, scale_inv_ub, sf_dense_ub, out_ub)
                else:
                    quantize_values(values_ub, amax_ub, scale_inv_ub, sf_dense_ub, out_ub)

                if out_config.use_tma_aligned_col_major_sf:
                    transpose_output_sf(sf_dense_ub, sf_out_ub, transpose_indices_ub)
                if not sf_only:
                    T.copy(out_ub, out[pid_token * block_m, col_offset])
                if not cast_only:
                    if out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0:
                        T.copy(sf_out_ub, out_sf[group_offset // pack_factor, pid_token * block_m * pack_factor])
                    elif out_config.use_tma_aligned_col_major_sf:
                        T.copy(sf_out_ub, out_sf[group_offset, pid_token * block_m])
                    else:
                        T.copy(sf_out_ub[0:block_m, 0:num_groups], out_sf[pid_token * block_m, group_offset])

    return per_token_cast_kernel
