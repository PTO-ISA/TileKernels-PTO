import math
import tilelang

from tile_kernels.utils import align
from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def get_per_block_cast_kernel_parallel(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    num_vec_cores: int,
):
    num_per_tokens, num_per_channels = out_config.sf_block
    assert (num_per_tokens, num_per_channels) == (32, 32)
    assert out_config.dtype in (T.float4_e2m1fn, T.float8_e4m3fn)
    quant_max = 6.0 if out_config.dtype == T.float4_e2m1fn else 448.0

    block_k_limit = 512 if in_config.dtype == T.float32 else 1024
    block_k = align(math.gcd(block_k_limit, hidden), 128)
    block_m = num_per_tokens
    num_stages = 2

    num_sf_cols_per_block = block_k // num_per_channels
    pack_factor = get_packed_ue8m0_pack_factor()
    assert num_sf_cols_per_block % pack_factor == 0
    packed_sf_cols_per_block = num_sf_cols_per_block // pack_factor
    num_stored_sf_cols_per_block = packed_sf_cols_per_block if out_config.use_packed_ue8m0 else num_sf_cols_per_block
    sf_ub_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.float32

    token_group = 4 if out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0 else 1

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    @T.macro
    def quant_block(x_ub, sf_ub, out_ub, sf_row):
        with T.SimdVF(lanes=128 if out_config.dtype == T.float4_e2m1fn else 64):
            column_max = T.alloc_shared((block_k,), T.float32)
            maxima = T.alloc_shared((num_sf_cols_per_block,), T.float32)
            scales = T.alloc_shared((align(num_sf_cols_per_block, 64),), T.float32)
            inverse = T.alloc_shared((align(num_sf_cols_per_block, 64),), T.float32)
            if not cast_only:
                for j in T.Parallel(block_k):
                    amax = T.alloc_var(T.float32, init=0.0)
                    for row in T.serial(block_m):
                        amax = T.max(amax, T.abs(T.float32(x_ub[row, j])))
                    column_max[j] = amax
                T.reduce_max(T.reshape(column_max, (num_sf_cols_per_block, 32)), maxima, dim=1)
                for j in T.Parallel(num_sf_cols_per_block):
                    amax = T.max(maxima[j], out_config.clamp_min_value)
                    if out_config.round_sf:
                        bits = T.reinterpret(amax * (1.0 / quant_max), T.uint32)
                        exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
                        scales[j] = T.reinterpret(exponent << 23, T.float32)
                        inverse[j] = T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
                    else:
                        scales[j] = amax / quant_max
                        inverse[j] = quant_max / amax
                for j in T.serial(num_stored_sf_cols_per_block):
                    for point in T.Parallel(1):
                        if out_config.use_packed_ue8m0:
                            low = T.reinterpret(scales[j * 2 + point], T.uint32) >> 23
                            high = T.reinterpret(scales[j * 2 + 1 + point], T.uint32) >> 23
                            packed = T.cast(low | (high << 8), T.uint16)
                            if out_config.use_tma_aligned_col_major_sf:
                                sf_ub[j, sf_row + point] = packed
                            else:
                                sf_ub[sf_row, j + point] = packed
                        elif out_config.use_tma_aligned_col_major_sf:
                            sf_ub[j, sf_row + point] = scales[j + point]
                        else:
                            sf_ub[sf_row, j + point] = scales[j + point]
            else:
                for j in T.Parallel(num_sf_cols_per_block):
                    scales[j] = sf_ub[0, j]
                    inverse[j] = 1.0 / sf_ub[0, j]
            if not sf_only:
                for row, j in T.Parallel(block_m, block_k):
                    value = T.float32(x_ub[row, j]) * inverse[j // 32]
                    if in_config.dtype == T.bfloat16 and out_config.round_sf:
                        value = T.float32(T.cast(value, T.bfloat16))
                    out_ub[row, j] = T.cast(value, out_config.dtype)

    @T.prim_func
    def per_block_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_cols = max(num_stored_sf_cols_per_block, 16 // token_group if out_config.use_packed_ue8m0 else 8)
            if out_config.use_tma_aligned_col_major_sf:
                sf_ub = T.alloc_shared((sf_cols, align(token_group, 32 // sf_ub_dtype.bytes)), sf_ub_dtype)
            else:
                sf_ub = T.alloc_shared((token_group, sf_cols), sf_ub_dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)

            versions = {x_ub: num_stages, sf_ub: num_stages}
            if not sf_only:
                versions[out_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_x, pid_y in T.Persistent(
                [T.ceildiv(num_tokens, block_m * token_group), T.ceildiv(hidden, block_k)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                sf_m_base = T.alloc_var(T.int32, init=pid_x * token_group)
                sf_k_base = T.alloc_var(T.int32, init=pid_y * num_sf_cols_per_block)
                stored_sf_k_base = T.alloc_var(T.int32, init=pid_y * num_stored_sf_cols_per_block)
                for token_offset in T.serial(token_group):
                    token_block = sf_m_base + token_offset
                    T.copy(x[token_block * block_m, pid_y * block_k], x_ub[0:block_m, 0:block_k])
                    if cast_only:
                        T.copy(out_sf[token_block, sf_k_base], sf_ub[0:1, 0:num_sf_cols_per_block])
                    quant_block(x_ub, sf_ub, out_ub, token_offset)
                    if not sf_only:
                        T.copy(out_ub[0:block_m, 0:block_k], out[token_block * block_m, pid_y * block_k])

                if not cast_only:
                    if out_config.use_packed_ue8m0:
                        output_sf = T.StridedTensor(
                            (sf_shape[0], sf_shape[1] // pack_factor), (sf_stride // pack_factor, 1), T.uint16, data=out_sf.data
                        )
                    else:
                        output_sf = out_sf
                    if out_config.use_tma_aligned_col_major_sf:
                        T.copy(sf_ub[0:num_stored_sf_cols_per_block, 0:token_group], output_sf[stored_sf_k_base, sf_m_base])
                    else:
                        T.copy(sf_ub[0:token_group, 0:num_stored_sf_cols_per_block], output_sf[sf_m_base, stored_sf_k_base])

    return per_block_cast_kernel
