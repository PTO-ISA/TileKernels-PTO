import math

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastInputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align, ceil_div


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_cast_back_kernel_vmi(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype,
    num_vec_cores: int,
):
    num_per_tokens, num_per_channels = in_config.sf_block
    pack_factor = get_packed_ue8m0_pack_factor() if in_config.use_packed_ue8m0 else 1
    assert (num_per_tokens, num_per_channels) in ((1, 32), (32, 1), (32, 32))
    assert not ((num_per_tokens, num_per_channels) == (32, 1) and in_config.use_tma_aligned_col_major_sf)

    min_k = 128 if in_config.dtype == T.float4_e2m1fn else 64
    block_k_limit = (128 if out_dtype == T.float32 else 512) if num_per_tokens == 1 else 256
    block_k = min(block_k_limit, max(min_k, ceil_div(hidden, min_k) * min_k))

    num_stages = 3 if num_per_tokens == 1 and out_dtype != T.float32 else 2
    sf_stages = num_stages if num_per_tokens == 1 else 1
    use_e2b_scale = (
        num_per_tokens == 1 and in_config.use_packed_ue8m0 and in_config.dtype == T.float4_e2m1fn and out_dtype != T.float32 and block_k % 256 == 0
    )
    ub_bytes_per_token = num_stages * block_k * (in_config.dtype.bits + out_dtype.bits) // 8 + sf_stages * 64  # 64 for worst case
    if num_per_tokens > 1:
        block_m_alignment = num_per_tokens
    elif use_e2b_scale:
        block_m_alignment = 128 // math.gcd(128, ceil_div(block_k, num_per_channels))
    else:
        block_m_alignment = 8
    block_m = get_max_ub_per_vector_core(use_simt=False) // ub_bytes_per_token // block_m_alignment * block_m_alignment

    num_sf_rows = ceil_div(block_m, num_per_tokens)
    num_sf_cols = ceil_div(block_k, num_per_channels)

    sf_raw_rows, sf_raw_cols = get_sf_shape((block_m, block_k), in_config)
    sf_raw_cols //= pack_factor
    sf_raw_dtype = T.uint16 if in_config.use_packed_ue8m0 else in_config.sf_dtype
    sf_raw_cols = align(sf_raw_cols, 32 // sf_raw_dtype.bytes)

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_input_shape = (sf_shape[0], sf_shape[1] // pack_factor) if in_config.use_packed_ue8m0 else sf_shape
    sf_stride = T.dynamic('sf_stride')
    num_token_tiles = T.ceildiv(num_tokens, block_m)
    num_hidden_tiles = T.ceildiv(hidden, block_k)

    use_lazy_packed_scale = in_config.use_packed_ue8m0 and not use_e2b_scale and num_per_channels == 32
    keep_unpacked_col_major = in_config.use_tma_aligned_col_major_sf and not in_config.use_packed_ue8m0
    sf_ub_shape = (num_sf_cols, num_sf_rows) if keep_unpacked_col_major else (num_sf_rows, num_sf_cols)
    if not in_config.use_packed_ue8m0:
        sf_ub_shape = (sf_ub_shape[0], align(sf_ub_shape[1], 32 // in_config.sf_dtype.bytes))
    num_sf_rows_padded = sf_ub_shape[1]

    @T.macro
    def load_scale_tile(x_sf_raw, sf_raw_ub, sf_ub, token_base, channel_base):
        if num_per_channels == 1:
            source = x_sf_raw[token_base // pack_factor, channel_base]
        elif in_config.use_tma_aligned_col_major_sf:
            source = x_sf_raw[channel_base // pack_factor, token_base]
        else:
            source_channel = channel_base // pack_factor if in_config.use_packed_ue8m0 else channel_base
            source = x_sf_raw[token_base, source_channel]
        if in_config.use_packed_ue8m0:
            T.copy(source, sf_raw_ub)
            if use_e2b_scale:
                assert num_sf_rows * num_sf_cols % 128 == 0
                with T.SimdVF():
                    mask = V.create_mask(128, size=128)
                    lanes = V.vinterpret_cast(V.vci(T.int16(0), size=128), 'uint16')
                    byte_mask = V.vbrc(T.uint16(0xFF), size=128)
                    for output_vector in T.serial(num_sf_rows * num_sf_cols // 128):
                        base = output_vector * 128
                        idx = V.vadds(lanes, base, mask)
                        rows = V.vshrs(idx, int(math.log2(num_sf_cols)), mask)
                        cols = V.vand(idx, V.vbrc(T.uint16(num_sf_cols - 1), size=128))
                        packed_channel = V.vshrs(cols, 1, mask)
                        byte_index = V.vand(cols, V.vbrc(T.uint16(1), size=128))
                        if in_config.use_tma_aligned_col_major_sf:
                            src_rows, src_cols = packed_channel, rows
                        else:
                            src_rows, src_cols = rows, packed_channel
                        src_idx = V.vadd(V.vmuls(src_rows, sf_raw_cols, mask), src_cols)
                        raw = V.vgather(sf_raw_ub[0, 0], src_idx, mask)
                        low_exp = V.vand(raw, byte_mask)
                        exp = V.vsel(V.vcmp(byte_index, V.vbrc(T.uint16(0), size=128), mask, 'eq'), low_exp, V.vshrs(raw, 8, mask))
                        scales = V.vinterpret_cast(V.vshls(exp, 7, mask), 'bfloat16')
                        V.vstore(scales, sf_ub[base // num_sf_cols, base % num_sf_cols])
            elif num_per_channels == 1:
                with T.SimdVF():
                    mask = V.create_mask(64, size=64)
                    byte_mask = V.vbrc(T.uint16(0xFF), size=64)
                    for group in T.serial(num_sf_rows):
                        packed_index = group + token_base % pack_factor
                        shift = packed_index % pack_factor * 8
                        for chunk in T.serial(num_sf_cols // 64):
                            col = chunk * 64
                            raw = V.vload(sf_raw_ub[packed_index // pack_factor, col], size=64)
                            exp = V.vand(V.vshrs(raw, shift, mask), byte_mask)
                            scales = V.vcvt(V.vinterpret_cast(V.vshls(exp, 7, mask), 'bfloat16'), 'float32')
                            V.vstore(scales, sf_ub[group, col])
        else:
            T.copy(source, sf_ub)

    @T.macro
    def load_scale(sf_ub, group, col, size, sf_shift=None, sf_exp_mask=None):
        if use_lazy_packed_scale:
            if in_config.use_tma_aligned_col_major_sf:
                packed_v = V.vload(sf_ub[col // 64, group], size=size * 2, stride=sf_raw_cols, group=size // 64)
            else:
                packed_v = V.vload(sf_ub[group, col // 64], size=size * 2, stride=1, group=size // 64)
            packed_u = V.vinterpret_cast(packed_v, 'uint32')
            return V.vinterpret_cast(V.vand(V.vshl(packed_u, sf_shift), sf_exp_mask), 'float32')
        if num_per_channels == 1:
            return V.vload(sf_ub[group, col], size=size)
        if keep_unpacked_col_major:
            return V.vload(sf_ub[col // 32, group], size=size, stride=num_sf_rows_padded, group=size // 32)
        return V.vload(sf_ub[group, col // 32], size=size, stride=1, group=size // 32)

    @T.macro
    def store_out(out_ub, row, col, out_f32):
        if out_dtype == T.float32:
            V.vstore(out_f32, out_ub[row, col])
        else:
            V.vstore(V.vcvt(out_f32, 'bfloat16'), out_ub[row, col])

    @T.macro
    def lazy_scale_args(size):
        return (
            V.vbrc(T.uint32(0x7F800000), size=size) if use_lazy_packed_scale else None,
            V.vsel(
                V.create_mask(32, size=size, group=size // 64),
                V.vbrc(T.uint32(23), size=size),
                V.vbrc(T.uint32(15), size=size),
            )
            if use_lazy_packed_scale and num_per_channels == 32
            else None,
        )

    @T.macro
    def transform_tile(x_ub, sf_ub, out_ub):
        with T.SimdVF():
            if use_e2b_scale:
                for row in T.serial(block_m):
                    for chunk in T.serial(block_k // 256):
                        col = chunk * 256
                        scale256 = V.vload(sf_ub[row, chunk * 8], size=256, stride=1, group=8)
                        x_256 = V.vcvt(V.vload(x_ub[row, col], size=256), 'bfloat16')
                        V.vstore(V.vmul(x_256, scale256), out_ub[row, col])
            else:
                scale_size = 128 if in_config.dtype == T.float4_e2m1fn else 64
                sf_exp_mask, sf_shift = lazy_scale_args(scale_size)
                for group in T.serial(num_sf_rows):
                    row_base = group * num_per_tokens
                    for chunk in T.serial(block_k // scale_size):
                        col = chunk * scale_size
                        scale = load_scale(sf_ub, group, col, scale_size, sf_shift, sf_exp_mask)
                        for row in T.serial(num_per_tokens):
                            raw = V.vload(x_ub[row_base + row, col], size=scale_size)
                            x_f32 = V.vcvt(raw, 'float32')
                            store_out(out_ub, row_base + row, col, V.vmul(x_f32, scale))

    @T.prim_func
    def cast_back_kernel_vmi(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[sf_input_shape, (sf_stride, 1), sf_raw_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_ub = T.alloc_shared(
                (sf_raw_rows, sf_raw_cols) if use_lazy_packed_scale else sf_ub_shape,
                T.uint16 if use_lazy_packed_scale else (T.bfloat16 if use_e2b_scale else T.float32),
            )
            sf_raw_ub = (
                T.alloc_shared((sf_raw_rows, sf_raw_cols), sf_raw_dtype) if in_config.use_packed_ue8m0 and not use_lazy_packed_scale else sf_ub
            )
            out_ub = T.alloc_shared((block_m, block_k), out_dtype)
            T.annotate_buffer_versions({x_ub: num_stages, sf_raw_ub: sf_stages, sf_ub: sf_stages, out_ub: num_stages})
            # Full last dimension: see per_token_cast - a smaller group size walks the
            # realigned PersistentFor domain column-major and loses DMA locality.
            for pid_token, pid_hidden in T.Persistent(
                [num_token_tiles, num_hidden_tiles], num_vec_cores, core_id, group_size=num_hidden_tiles, num_stages=num_stages
            ):
                token_base = pid_token * block_m // num_per_tokens
                channel_base = pid_hidden * block_k // num_per_channels
                T.copy(x[pid_token * block_m, pid_hidden * block_k], x_ub)
                load_scale_tile(x_sf, sf_raw_ub, sf_ub, token_base, channel_base)
                transform_tile(x_ub, sf_ub, out_ub)
                T.copy(out_ub, out[pid_token * block_m, pid_hidden * block_k])

    return cast_back_kernel_vmi
