import math

import tilelang

from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T


def get_sf_storage(config, dtype):
    if config.use_packed_ue8m0:
        return T.uint16, get_packed_ue8m0_pack_factor()
    return dtype, 1


@T.macro
def init_fp4_to_fp8_map(fp4_to_fp8_map_ub, invalid_exp_ub):
    with T.SimdVF(lanes=128):
        for j in T.Parallel(16):
            invalid_exp_ub[j] = T.uint16(0)


@tilelang.jit(target='pto')
def get_per_block_cast_lossless_kernel_parallel(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    assert in_config.dtype == T.float4_e2m1fn and out_config.dtype == T.float8_e4m3fn
    assert in_config.sf_block == (1, 32) and out_config.sf_block == (32, 32)
    assert out_config.round_sf
    assert not in_config.use_tma_aligned_col_major_sf or in_config.use_packed_ue8m0
    assert not out_config.use_tma_aligned_col_major_sf or out_config.use_packed_ue8m0
    assert hidden % 256 == 0

    block_k = math.gcd(2048, hidden)
    block_m = 32
    num_stages = 2
    num_per_channels = in_config.sf_block[1]
    num_sf_cols_per_block = block_k // num_per_channels
    vec_size = 64
    assert vec_size * num_per_channels >= block_k

    sf_dtype, in_pack_factor = get_sf_storage(in_config, T.uint32)
    num_sf_load_cols = num_sf_cols_per_block // in_pack_factor
    out_sf_dtype, out_pack_factor = get_sf_storage(out_config, T.float32)
    num_sf_store_cols = num_sf_cols_per_block // out_pack_factor
    if in_config.use_tma_aligned_col_major_sf:
        sf_load_ub_shape = (num_sf_load_cols, block_m * in_pack_factor)
    else:
        sf_load_ub_shape = (block_m, vec_size * in_pack_factor)
    if out_config.use_tma_aligned_col_major_sf:
        num_batched_tiles = 2
        out_sf_shape_ub = (max(num_sf_store_cols, 4 * out_pack_factor), num_batched_tiles * out_pack_factor)
    else:
        num_batched_tiles = 1
        out_sf_shape_ub = (1, max(num_sf_store_cols, 8 * out_pack_factor) * out_pack_factor)
    num_tokens = T.dynamic('num_tokens')
    in_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    if in_config.use_tma_aligned_col_major_sf:
        in_sf_stride = T.dynamic('in_sf_stride')
    else:
        in_sf_stride = in_sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        out_sf_stride = T.dynamic('out_sf_stride')
    else:
        out_sf_stride = out_sf_shape[1]

    @T.macro
    def copy_sf(x_sf, sf_load_ub, row_start, pid_k):
        if in_config.use_tma_aligned_col_major_sf:
            col_start = T.alloc_var(T.int32, init=pid_k * num_sf_load_cols)
            T.copy(x_sf[col_start, row_start * in_pack_factor], sf_load_ub[:num_sf_load_cols, : block_m * in_pack_factor])
        else:
            col_start = T.alloc_var(T.int32, init=pid_k * num_sf_cols_per_block)
            T.copy(x_sf[row_start, col_start], sf_load_ub[:block_m, :num_sf_cols_per_block])

    @T.macro
    def reduce_sf(sf_load_ub, sf_exp_ub, out_sf_ub, invalid_exp_ub, store_row):
        with T.SimdVF(lanes=128):
            maxima = T.alloc_shared((num_sf_cols_per_block,), T.int32)
            invalid = T.alloc_shared((num_sf_cols_per_block,), T.int32)
            invalid_total = T.alloc_shared((1,), T.int32)
            for j in T.Parallel(num_sf_cols_per_block):
                maximum = T.alloc_var(T.int32, init=0)
                minimum = T.alloc_var(T.int32, init=255)
                for row in T.serial(block_m):
                    if in_config.use_tma_aligned_col_major_sf:
                        exp = T.int32(sf_load_ub[j // in_pack_factor, row * in_pack_factor + j % in_pack_factor])
                    elif in_config.use_packed_ue8m0:
                        exp = T.int32(sf_load_ub[row, j])
                    else:
                        exp = T.int32(T.reinterpret(sf_load_ub[row, j], T.uint32) >> 23)
                    sf_exp_ub[row, j] = T.uint16(exp)
                    maximum = T.max(maximum, exp)
                    minimum = T.min(minimum, exp)
                out_exp = T.max(maximum, 6) - 6
                sf_exp_ub[block_m, j] = T.uint16(out_exp)
                maxima[j] = out_exp
                invalid[j] = out_exp - T.min(out_exp, minimum + 5)
            T.reduce_max(invalid, invalid_total, dim=0)
            for j in T.Parallel(16):
                invalid_exp_ub[j] = T.max(invalid_exp_ub[j], T.uint16(invalid_total[0]))
            for group in T.serial(num_sf_store_cols):
                for point in T.Parallel(1):
                    if out_config.use_packed_ue8m0:
                        packed = T.uint16(T.uint32(maxima[group * 2 + point]) | (T.uint32(maxima[group * 2 + 1 + point]) << 8))
                        out_words = T.view(out_sf_ub, (out_sf_shape_ub[0], out_sf_shape_ub[1] // 2), T.uint16)
                        if out_config.use_tma_aligned_col_major_sf:
                            out_words[group, store_row + point] = packed
                        else:
                            out_words[0, group + point] = packed
                    else:
                        scale = T.reinterpret(T.uint32(maxima[group + point]) << 23, T.float32)
                        out_sf_ub[0, group + point] = scale

    @T.macro
    def quant(sf_exp_ub, x_ub, out_ub, fp4_to_fp8_map_ub):
        with T.SimdVF(lanes=128):
            for row, j in T.Parallel(block_m, block_k):
                delta = T.int32(sf_exp_ub[row, j // 32]) - T.int32(sf_exp_ub[block_m, j // 32])
                factor = T.reinterpret(T.uint32(delta + 127) << 23, T.float32)
                out_ub[row, j] = T.cast(T.float32(x_ub[row, j]) * factor, T.float8_e4m3fn)

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[in_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[out_sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_exp_ub = T.alloc_shared((block_m + 1, max(num_sf_cols_per_block, 16)), T.uint16)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            fp4_to_fp8_map_ub = T.alloc_shared((256,), T.uint8)
            sf_load_ub = T.alloc_shared(sf_load_ub_shape, in_config.sf_dtype)
            out_sf_ub = T.alloc_shared(out_sf_shape_ub, out_config.sf_dtype)
            invalid_exp_ub = T.alloc_shared((16,), T.uint16)
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    sf_load_ub: num_stages,
                    sf_exp_ub: num_stages,
                    out_ub: num_stages,
                    out_sf_ub: num_stages,
                    invalid_exp_ub: 1,
                }
            )
            init_fp4_to_fp8_map(fp4_to_fp8_map_ub, invalid_exp_ub)
            for pid_m, pid_k in T.Persistent(
                [num_tokens // (block_m * num_batched_tiles), hidden // block_k],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                for batch in T.serial(num_batched_tiles):
                    tile_m = T.alloc_var(T.int32, init=pid_m * num_batched_tiles + batch)
                    row_start = T.alloc_var(T.int32, init=tile_m * block_m)
                    T.copy(x[row_start, pid_k * block_k], x_ub[:block_m, :])
                    copy_sf(x_sf, sf_load_ub, row_start, pid_k)
                    reduce_sf(sf_load_ub, sf_exp_ub, out_sf_ub, invalid_exp_ub, batch)
                    quant(sf_exp_ub, x_ub, out_ub, fp4_to_fp8_map_ub)
                    T.copy(out_ub, out[row_start, pid_k * block_k])
                first_tile = T.alloc_var(T.int32, init=pid_m * num_batched_tiles)
                store_base = T.alloc_var(T.int32, init=pid_k * num_sf_store_cols)
                if out_config.use_tma_aligned_col_major_sf:
                    T.copy(out_sf_ub[:num_sf_store_cols, : num_batched_tiles * out_pack_factor], out_sf[store_base, first_tile * out_pack_factor])
                else:
                    T.copy(out_sf_ub[0, :num_sf_cols_per_block], out_sf[first_tile, store_base * out_pack_factor])
            T.device_assert(invalid_exp_ub[0] == 0)

    return kernel
