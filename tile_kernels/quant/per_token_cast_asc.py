import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.quant.common import CastInputConfig, CastOutputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_per_token_cast_kernel_asc(
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
    def compute_scale(amax, lanes):
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(out_config.clamp_min_value), size=lanes))
        if not out_config.round_sf:
            max_value = V.vbrc(T.float32(quant_max), size=lanes)
            return V.vdiv(clamped, max_value, mask, precision='exact'), V.vdiv(max_value, clamped, mask, precision='exact')
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(V.vmul(clamped, V.vbrc(T.float32(inv_quant_max), size=lanes)), 'uint32')
        exponent = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        inverse = V.vinterpret_cast(V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), exponent), shift), 'float32')
        if out_config.use_packed_ue8m0:
            return exponent, inverse
        return V.vinterpret_cast(V.vshl(exponent, shift), 'float32'), inverse

    @T.macro
    def f32_to_bf16_round_odd(x):
        low, high = V.vunzip(x, 'uint16')
        return V.vinterpret_cast(V.vor(high, V.vmin(low, V.vbrc(T.uint16(1), size=128))), 'bfloat16')

    @T.macro
    def load_packed_input_scale_bf16(scales_ub, group, sf_shift, sf_exp_mask):
        packed = V.vload(scales_ub[group // pack_factor], size=256, dist_mode='brc', group=2, stride=1)
        scale_bits = V.vand(V.vshl(V.vinterpret_cast(packed, 'uint32'), sf_shift), sf_exp_mask)
        scale_u16 = V.vcvt(V.vshr(scale_bits, V.vbrc(T.uint32(16), size=128)), 'uint16', saturate='NOSAT')
        return V.vinterpret_cast(scale_u16, 'bfloat16')

    @T.macro
    def init_transpose_indices(indices):
        with T.SimdVF():
            if out_config.use_packed_ue8m0:
                lane = V.vci(T.int16(0), size=128)
                output_row = V.vand(V.vinterpret_cast(lane, 'uint16'), V.vbrc(T.uint16(block_m - 1), size=128))
                group_in_vector = V.vinterpret_cast(V.vshr(lane, V.vbrc(T.int16(log2_block_m), size=128)), 'uint16')
                V.vstore(V.vadd(V.vmul(output_row, V.vbrc(T.uint16(sf_stride // pack_factor), size=128)), group_in_vector), indices[0])
            else:
                lane = V.vci(T.int32(0), size=64)
                output_row = V.vand(V.vinterpret_cast(lane, 'uint32'), V.vbrc(T.uint32(block_m - 1), size=64))
                group_in_vector = V.vinterpret_cast(V.vshr(lane, V.vbrc(T.int32(log2_block_m), size=64)), 'uint32')
                V.vstore(V.vadd(V.vmul(output_row, V.vbrc(T.uint32(sf_stride), size=64)), group_in_vector), indices[0])

    @T.macro
    def transpose_vectors(src, dst, indices, vector_lanes, num_output_groups):
        with T.SimdVF():
            groups_per_vector = vector_lanes // block_m
            num_output_values = num_output_groups * block_m
            num_full_vectors = num_output_values // vector_lanes
            num_tail_values = num_output_values % vector_lanes
            source_indices = V.vload(indices[0], size=vector_lanes)
            full_mask = V.create_mask(vector_lanes, size=vector_lanes)
            for output_vector in T.unroll(num_full_vectors, explicit=True):
                full_group_base = output_vector * groups_per_vector
                V.vstore(V.vgather(src[0, full_group_base], source_indices, full_mask), dst[full_group_base, 0])
            if num_tail_values > 0:
                tail_group_base = num_full_vectors * groups_per_vector
                tail_mask = V.create_mask(num_tail_values, size=vector_lanes)
                V.vstore(V.vgather(src[0, tail_group_base], source_indices, tail_mask), dst[tail_group_base, 0], tail_mask)

    @T.macro
    def transpose_output_sf(src, dst, indices):
        if out_config.use_packed_ue8m0:
            src_u16 = T.Tensor((block_m, sf_stride // pack_factor), T.uint16, src.data)
            dst_u16 = T.Tensor((num_groups // pack_factor, block_m), T.uint16, dst.data)
            transpose_vectors(src_u16, dst_u16, indices, 128, num_groups // pack_factor)
        else:
            transpose_vectors(src, dst, indices, 64, num_groups)

    @T.macro
    def dequantize_input(x_ub, scales_ub, out_ub):
        with T.SimdVF():
            if in_config.use_packed_ue8m0:
                mask_low = V.create_mask(32, size=128)
                sf_shift = V.vsel(mask_low, V.vbrc(T.uint32(23), size=128), V.vbrc(T.uint32(15), size=128))
                sf_exp_mask = V.vbrc(T.uint32(0x7F800000), size=128)
            for pair in T.serial(block_k // 128):
                col = pair * 128
                group = pair * 4
                if in_config.use_packed_ue8m0:
                    scale = load_packed_input_scale_bf16(scales_ub, group, sf_shift, sf_exp_mask)
                    if value_dtype != T.bfloat16:
                        scale = V.vcvt(scale, 'float32')
                else:
                    scale = V.vload(scales_ub[group], size=128, dist_mode='brc', group=4, stride=1)
                    if value_dtype == T.bfloat16:
                        scale = V.vcvt(scale, 'bfloat16')
                for row in T.serial(block_m):
                    raw = V.vload(x_ub[row, col], size=128)
                    if value_dtype == T.bfloat16:
                        values = V.vcvt(V.vcvt(raw, 'float32'), 'bfloat16') if in_config.dtype == T.float8_e4m3fn else V.vcvt(raw, 'bfloat16')
                    else:
                        values = V.vcvt(raw, 'float32') if in_config.dtype == T.float8_e4m3fn else V.vcvt(V.vcvt(raw, 'bfloat16'), 'float32')
                    V.vstore(V.vmul(values, scale), out_ub[row, col])

    @T.macro
    def quantize_bf16(x_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub):
        with T.SimdVF():
            mask256 = V.create_mask(256, size=256)
            abs_mask = V.vbrc(T.uint16(0x7FFF), size=256)
            for row in T.serial(block_m):
                for tile in T.unroll(block_k // 256, explicit=True):
                    col = tile * 256
                    amax_u = V.vcmax(V.vand(V.vinterpret_cast(V.vload(x_ub[row, col], size=256), 'uint16'), abs_mask), mask256, group=8)
                    V.vstore(V.vcvt(V.vinterpret_cast(amax_u, 'bfloat16'), 'float32'), amax_ub[row, tile * 8], group=8, stride=1)
            T.simd.mem_bar('VST_VLD')
            for row in T.serial(block_m):
                scale, inverse = compute_scale(V.vload(amax_ub[row, 0], size=64), 64)
                V.vstore(V.vcvt(scale, 'uint8'), sf_dense_ub[row, 0])
                V.vstore(V.vcvt(inverse, 'bfloat16'), sf_inv_ub[row, 0])
            if not sf_only:
                T.simd.mem_bar('VST_VLD')
                for row in T.serial(block_m):
                    for tile in T.serial(block_k // 256):
                        col = tile * 256
                        inverse = V.vload(sf_inv_ub[row, tile * 8], size=256, dist_mode='brc', group=8, stride=1)
                        values = V.vmul(V.vload(x_ub[row, col], size=256), inverse)
                        if out_config.dtype == T.float4_e2m1fn:
                            V.vstore(V.vcvt(values, 'float4_e2m1fn', rounding='R'), out_ub[row, col])
                        else:
                            V.vstore(V.vcvt(V.vcvt(values, 'float32'), 'float8_e4m3fn', rounding='R', saturate='SAT'), out_ub[row, col])

    @T.macro
    def load_f32_pair(x_ub, row, col):
        if in_config.dtype == T.bfloat16:
            return V.vcvt(V.vload(x_ub[row, col], size=128), 'float32')
        return V.vload(x_ub[row, col], size=128)

    @T.macro
    def quantize_f32(x_ub, amax_ub, sf_inv_ub, sf_dense_ub, out_ub):
        with T.SimdVF():
            mask64 = V.create_mask(64, size=64)
            mask128 = V.create_mask(128, size=128)
            if cast_only:
                one = V.vbrc(T.float32(1.0), size=64)
                for row in T.serial(block_m):
                    V.vstore(V.vdiv(one, V.vload(sf_dense_ub[row, 0], size=64), mask64, precision='exact'), sf_inv_ub[row, 0])
            else:
                for row in T.serial(block_m):
                    for pair in T.serial(block_k // 128):
                        col = pair * 128
                        V.vstore(V.vcmax(V.vabs(load_f32_pair(x_ub, row, col)), mask128, group=4), amax_ub[row, pair * 4])
                T.simd.mem_bar('VST_VLD')
                for row in T.serial(block_m):
                    scale, inverse = compute_scale(V.vload(amax_ub[row, 0], size=64), 64)
                    if out_config.use_packed_ue8m0:
                        V.vstore(V.vcvt(scale, 'uint8'), sf_dense_ub[row, 0])
                    else:
                        V.vstore(scale, sf_dense_ub[row, 0])
                    V.vstore(inverse, sf_inv_ub[row, 0])
            if not sf_only:
                T.simd.mem_bar('VST_VLD')
                for row in T.serial(block_m):
                    for pair in T.serial(block_k // 128):
                        col = pair * 128
                        inverse = V.vload(sf_inv_ub[row, pair * 4], size=128, dist_mode='brc', group=4, stride=1)
                        quantized = V.vmul(load_f32_pair(x_ub, row, col), inverse)
                        if out_config.dtype == T.float4_e2m1fn:
                            V.vstore(V.vcvt(f32_to_bf16_round_odd(quantized), 'float4_e2m1fn', rounding='R'), out_ub[row, col])
                        else:
                            V.vstore(V.vcvt(quantized, 'float8_e4m3fn', rounding='R', saturate='SAT'), out_ub[row, col])

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
                    quantize_bf16(values_ub, amax_ub, scale_inv_ub, sf_dense_ub, out_ub)
                else:
                    quantize_f32(values_ub, amax_ub, scale_inv_ub, sf_dense_ub, out_ub)

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
