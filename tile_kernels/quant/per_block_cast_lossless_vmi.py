import math

import tilelang

from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


def get_sf_storage(config, dtype):
    if config.use_packed_ue8m0:
        return T.uint16, get_packed_ue8m0_pack_factor()
    return dtype, 1


@T.macro
def init_fp4_to_fp8_map(fp4_to_fp8_map_ub, invalid_exp_ub):
    with T.SimdVF():
        mask = V.create_mask(256, size=256)
        lanes = V.vci(T.uint8(0), size=256)
        code = V.vand(lanes, V.vbrc(T.uint8(0x0F), size=256))
        magnitude = V.vand(code, V.vbrc(T.uint8(0x07), size=256))

        # E4M3 magnitude encodings for FP4 magnitudes 0, 0.5, 1, 1.5,
        # 2, 3, 4, 6.
        base = V.vshls(magnitude, 2, mask)
        base = V.vadd(base, V.vbrc(T.uint8(0x30), size=256))
        is_one = V.vcmps(magnitude, T.uint8(1), mask, 'eq')
        base = V.vsel(is_one, V.vbrc(T.uint8(0x30), size=256), base)

        # The lookup index packs exponent delta in the high nibble and the
        # E2M1 code in the low nibble. Bias by -5 output exponents here.
        delta = V.vand(V.vshrs(lanes, 1, mask), V.vbrc(T.uint8(0x78), size=256), mask)
        delta = V.vadds(delta, 216, mask)
        scaled = V.vadd(base, delta, mask)
        is_nonzero = V.vcmps(magnitude, T.uint8(0), mask, 'ne')
        scaled = V.vsel(is_nonzero, scaled, V.vbrc(T.uint8(0), size=256))
        sign = V.vshls(V.vand(code, V.vbrc(T.uint8(0x08), size=256)), 4, mask)
        V.vstore(V.vor(scaled, sign), fp4_to_fp8_map_ub[0])
        V.vstore(V.vbrc(T.uint16(0), size=8), invalid_exp_ub[0])


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_per_block_cast_lossless_kernel_vmi(
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
    def load_sf(sf_load_ub, row, gather_base, gather_mask):
        if in_config.use_tma_aligned_col_major_sf:
            return V.vgather(sf_load_ub[0, 0], V.vadds(gather_base, row * in_pack_factor, gather_mask), gather_mask)
        elif sf_dtype == T.uint16:
            return V.vcvt(V.vload(sf_load_ub[row, 0], size=64), 'uint16')
        else:
            values = V.vload(sf_load_ub[row, 0], size=64)
        return V.vcvt(V.vshrs(V.vinterpret_cast(values, 'uint32'), 23, gather_mask), 'uint16', saturate='NOSAT')

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
        with T.SimdVF():
            mask = V.create_mask(num_sf_cols_per_block, size=64)
            zero = V.vbrc(T.uint16(0), size=64)
            six = V.vbrc(T.uint16(6), size=64)
            gather_base = V.alloc_var(T.uint16, size=64)
            gather_mask = mask
            if in_config.use_tma_aligned_col_major_sf:
                lanes = V.vci(T.uint16(0), size=64)
                if in_config.use_packed_ue8m0:
                    pair_index = V.vshrs(lanes, 1, mask)
                    byte_index = V.vand(lanes, V.vbrc(T.uint16(1), size=64))
                    gather_base = V.vadd(V.vmuls(pair_index, block_m * in_pack_factor, mask), byte_index, mask)
                else:
                    gather_base = V.vmuls(lanes, block_m, mask)

            max_exp = V.alloc_var(T.uint16, size=64)
            min_exp = V.alloc_var(T.uint16, size=64)
            max_exp = zero
            min_exp = V.vbrc(T.max_value(T.uint16), size=64)
            for row in T.serial(block_m):
                in_exp = load_sf(sf_load_ub, row, gather_base, gather_mask)
                V.vstore(in_exp, sf_exp_ub[row, 0], mask)
                max_exp = V.vmax(max_exp, in_exp, mask)
                min_exp = V.vmin(min_exp, in_exp, mask)
            out_exp = V.vsub(V.vmax(max_exp, six, mask), six, mask)
            V.vstore(V.vsub(out_exp, V.vbrc(T.uint16(5), size=64), mask), sf_exp_ub[block_m, 0], mask)
            invalid = V.vsub(out_exp, V.vmin(out_exp, V.vadd(min_exp, V.vbrc(T.uint16(5), size=64), mask), mask), mask)
            max_invalid = V.vcmax(invalid, mask)
            acc_invalid = V.vmax(max_invalid, V.vload(invalid_exp_ub[0], size=1))
            V.vstore(acc_invalid, invalid_exp_ub[0])
            if out_sf_dtype == T.uint16:
                out_sf_value = V.vcvt(out_exp, 'uint8', saturate='NOSAT')
            else:
                wide_exp = V.vzip(out_exp, zero, to_dtype='uint32')
                out_sf_value = V.vinterpret_cast(V.vshls(wide_exp, 23, mask), 'float32')

            out_store_mask = mask
            if out_config.use_tma_aligned_col_major_sf:
                store_indices = V.vci(T.uint16(0), size=64)
                pair_index = V.vshrs(store_indices, 1, out_store_mask)
                byte_index = V.vand(store_indices, V.vbrc(T.uint16(1), size=64))
                offsets = V.vadd(V.vshls(pair_index, 2, out_store_mask), byte_index, out_store_mask)
                offsets = V.vadds(offsets, store_row * out_pack_factor, out_store_mask)
                V.vscatter(out_sf_value, out_sf_ub[0, 0], offsets, out_store_mask)
            else:
                V.vstore(out_sf_value, out_sf_ub[0, 0], out_store_mask)

    @T.macro
    def quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8, out_exp, row, chunk):
        load_mask = V.create_mask(256, size=256)
        in_exp_slots = V.vload(sf_exp_ub[row, chunk * 8], size=8, group=8, stride=1)
        in_exp = V.vcvt(
            V.vbrc(in_exp_slots, size=256, group=8),
            'uint8',
            saturate='NOSAT',
        )
        delta = V.vshls(V.vsub(in_exp, out_exp), 4, load_mask)
        index = V.vadd(fp4_codes_u8, delta, load_mask)
        quantized = V.vinterpret_cast(V.vselr(fp4_to_fp8_map, index), 'float8_e4m3fn')
        V.vstore(quantized, out_ub[row, chunk * 256])

    @T.macro
    def quant(sf_exp_ub, x_ub, out_ub, fp4_to_fp8_map_ub):
        with T.SimdVF():
            mask = V.vbrc(T.uint8(0x0F), size=256)
            load_mask = V.create_mask(256, size=256)
            fp4_to_fp8_map = V.vload(fp4_to_fp8_map_ub[0], size=256)
            if block_k == 256:
                out_exp = V.vcvt(V.vbrc(V.vload(sf_exp_ub[block_m, 0], size=8, group=8, stride=1), size=256, group=8), 'uint8', saturate='NOSAT')
                for row in T.serial(0, block_m, 2):
                    packed = V.vinterpret_cast(V.vload(x_ub[row, 0], size=512), 'uint8')
                    fp4_codes_u8_0, fp4_codes_u8_1 = V.vintlv(V.vand(packed, mask), V.vshrs(packed, 4, load_mask), load_mask)
                    quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_0, out_exp, row, 0)
                    quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_1, out_exp, row + 1, 0)
            else:
                for chunk in T.serial(0, block_k // 256, 2):
                    out_exp0 = V.vcvt(V.vbrc(V.vload(sf_exp_ub[block_m, chunk * 8], size=8, group=8, stride=1), size=256, group=8), 'uint8', saturate='NOSAT')
                    out_exp1 = V.vcvt(V.vbrc(V.vload(sf_exp_ub[block_m, (chunk + 1) * 8], size=8, group=8, stride=1), size=256, group=8), 'uint8', saturate='NOSAT')
                    for row in T.serial(block_m):
                        packed = V.vinterpret_cast(V.vload(x_ub[row, chunk * 256], size=512), 'uint8')
                        fp4_codes_u8_0, fp4_codes_u8_1 = V.vintlv(V.vand(packed, mask), V.vshrs(packed, 4, load_mask), load_mask)
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_0, out_exp0, row, chunk)
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_1, out_exp1, row, chunk + 1)

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
