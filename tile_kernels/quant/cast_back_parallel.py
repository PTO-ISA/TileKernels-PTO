"""Scalar-parallel dequantization with the ASC blocking and DMA schedule."""

import math

import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastInputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align, ceil_div


@tilelang.jit(target='pto')
def get_cast_back_kernel_parallel(
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
                with T.SimdVF(lanes=64):
                    for i, j in T.Parallel(num_sf_rows, num_sf_cols):
                        packed = sf_raw_ub[j // pack_factor, i] if in_config.use_tma_aligned_col_major_sf else sf_raw_ub[i, j // pack_factor]
                        exponent = T.cast((packed >> ((j % pack_factor) * 8)) & 255, T.uint16)
                        sf_ub[i, j] = T.reinterpret(exponent << 7, T.bfloat16)
            elif num_per_channels == 1:
                with T.SimdVF(lanes=64):
                    for i, j in T.Parallel(num_sf_rows, num_sf_cols):
                        packed_index = i + token_base % pack_factor
                        packed = T.cast(sf_raw_ub[packed_index // pack_factor, j], T.uint32)
                        exponent = (packed >> ((packed_index % pack_factor) * 8)) & T.uint32(255)
                        sf_ub[i, j] = T.reinterpret(exponent << 23, T.float32)
        else:
            T.copy(source, sf_ub)

    @T.macro
    def load_scale(sf_ub, group, col):
        if use_lazy_packed_scale:
            packed = sf_ub[col // 64, group] if in_config.use_tma_aligned_col_major_sf else sf_ub[group, col // 64]
            exponent = (T.cast(packed, T.uint32) >> ((col // 32 % pack_factor) * 8)) & T.uint32(255)
            return T.reinterpret(exponent << 23, T.float32)
        if num_per_channels == 1:
            return sf_ub[group, col]
        if keep_unpacked_col_major:
            return sf_ub[col // num_per_channels, group]
        return sf_ub[group, col // num_per_channels]

    @T.macro
    def transform_tile(x_ub, sf_ub, out_ub):
        with T.SimdVF(lanes=64):
            for i, j in T.Parallel(block_m, block_k):
                scale = load_scale(sf_ub, i // num_per_tokens, j)
                value = T.float32(x_ub[i, j]) * T.float32(scale)
                out_ub[i, j] = T.cast(value, out_dtype)

    @T.prim_func
    def cast_back_kernel_ascend(
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

    return cast_back_kernel_ascend
