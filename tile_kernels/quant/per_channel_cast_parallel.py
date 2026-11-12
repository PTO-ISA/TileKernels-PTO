"""Per-channel quantization with ASC scheduling and scalar-parallel VF math."""

import math

import tilelang

from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tile_kernels.quant.common import get_packed_ue8m0_pack_factor


@tilelang.jit(target='pto')
def get_per_channel_cast_kernel_parallel(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    num_per_tokens = out_config.sf_block[0]
    assert num_per_tokens == 32, 'Ascend cast supports num_per_tokens == 32 only'
    if in_config.with_sf:
        num_per_channels = in_config.sf_block[1]
        assert num_per_channels == 32, 'Ascend rescale supports num_per_channels == 32 only'
    else:
        assert in_config.dtype == T.bfloat16, 'per_channel_cast supports bf16 or e4m3 (rescale) input only'

    quant_max = 448.0
    assert out_config.clamp_min_value >= quant_max * (2 ** (-126)), 'Ascend scale exponent fast path requires clamp to normal number'

    use_bf16_compute = not in_config.with_sf or in_config.use_packed_ue8m0
    # Packed input with rounded output decode at 1024x256, the other bf16 paths pick 256x128.
    use_large_block = in_config.with_sf and in_config.use_packed_ue8m0 and out_config.round_sf
    vector_width = 256  # SIMD vector width in bytes
    vector_alignment = 32  # Minimum byte alignment for vector loads
    max_block_k = 1024 if use_large_block else 256
    block_k = math.gcd(max_block_k, hidden)
    block_m = min(128, 32768 // block_k)
    sf_rows_per_group = block_m // num_per_tokens

    assert block_m % num_per_tokens == 0, (
        f'Ascend per channel cast requires block_m % num_per_tokens == 0, got block_m={block_m}, num_per_tokens={num_per_tokens}'
    )
    assert block_k % 128 == 0, f'Ascend per channel cast requires block_k % 128 == 0, got block_k={block_k} (hidden={hidden})'
    num_k_blocks = hidden // block_k

    pack_factor = get_packed_ue8m0_pack_factor()
    assert pack_factor == 2
    if in_config.with_sf:
        sf_in_block_k = num_per_channels * pack_factor if in_config.use_packed_ue8m0 else num_per_channels
    num_blocks_per_group = pack_factor if out_config.use_packed_ue8m0 else 1

    num_stages = 2

    num_tokens = T.dynamic('num_tokens')
    sf_stride = T.dynamic('sf_stride')
    num_groups = T.ceildiv(num_tokens, block_m * num_blocks_per_group) * num_k_blocks
    num_waves = T.ceildiv(num_groups, num_vec_cores) * num_blocks_per_group
    sf_out_shape = get_sf_shape((num_tokens, hidden), out_config)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    packed_col_major_sf = in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0
    if packed_col_major_sf:
        x_sf_shape = (x_sf_shape[0], x_sf_shape[1] // pack_factor)
    in_sf_stride = T.dynamic('in_sf_stride') if in_config.with_sf else x_sf_shape[1]
    sf_in_dtype = T.int16 if packed_col_major_sf else in_config.sf_dtype
    quant_chunk_size = 128 if use_bf16_compute and not use_large_block else min(256, block_k)
    # Count 64-lane fp32 vectors; each pair combines into one 128-lane bf16 vector.
    num_vecs = quant_chunk_size // 64
    compute_vec_factor = 2 if use_bf16_compute else 1

    if in_config.with_sf:
        if in_config.use_tma_aligned_col_major_sf:
            # Col-major layout is read via per-lane gather, not wide loads, so no tail padding needed.
            sf_in_ub_shape = (block_k // sf_in_block_k, block_m)
        else:
            # Row stride stays vector_alignment-byte aligned for the aligned vector loads; vector_alignment // bytes = element floor.
            sf_in_cols = max(block_k // num_per_channels, vector_alignment // sf_in_dtype.bytes)
            # Pad the tail: a full vector_width (256B) load spills past the last row.
            sf_load_elements = vector_width // sf_in_dtype.bytes
            sf_in_rows = block_m + T.ceildiv(sf_load_elements, sf_in_cols) - 1
            sf_in_ub_shape = (sf_in_rows, sf_in_cols)
        dequant_ub_dtype = T.bfloat16 if use_bf16_compute else T.float32

    @T.macro
    def input_value(x_ub, sf_in_ub, row, col):
        value = T.float32(x_ub[row, col])
        if in_config.with_sf:
            group = col // num_per_channels
            if in_config.use_packed_ue8m0:
                if packed_col_major_sf:
                    packed = T.cast(sf_in_ub[group // pack_factor, row], T.uint16)
                    exponent = T.cast((packed >> ((group % pack_factor) * 8)) & 255, T.uint32)
                else:
                    exponent = T.cast(sf_in_ub[row, group], T.uint32)
                scale = T.reinterpret(exponent << 23, T.float32)
            else:
                scale = sf_in_ub[group, row] if in_config.use_tma_aligned_col_major_sf else sf_in_ub[row, group]
            value = value * scale
            if use_bf16_compute:
                value = T.float32(T.cast(value, T.bfloat16))
        return value

    @T.macro
    def compute_sf_and_inv(amax):
        clamped = T.max(amax, out_config.clamp_min_value)
        if not out_config.round_sf:
            return clamped / quant_max, quant_max / clamped
        raw_bits = T.reinterpret(clamped * (1.0 / quant_max), T.uint32)
        exponent = ((raw_bits - T.uint32(1)) >> 23) + T.uint32(1)
        inverse = T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
        if out_config.use_packed_ue8m0:
            return T.cast(exponent, T.uint8), inverse
        return T.reinterpret(exponent << 23, T.float32), inverse

    @T.macro
    def quantize_block(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, sf_row_base=0):
        inverse_ub = T.alloc_shared((block_k,), T.float32)
        for group in T.serial(sf_rows_per_group):
            with T.SimdVF(lanes=64):
                for j in T.Parallel(block_k):
                    amax = T.alloc_var(T.float32, init=T.abs(input_value(x_ub, sf_in_ub, group * num_per_tokens, j)))
                    for row in T.serial(1, num_per_tokens):
                        amax = T.max(amax, T.abs(input_value(x_ub, sf_in_ub, group * num_per_tokens + row, j)))
                    sf, inv = compute_sf_and_inv(amax)
                    sf_out_ub[sf_row_base + group, j] = sf
                    inverse_ub[j] = inv
            # M2 accepts one serial carry per Parallel unit. Keep reduction
            # and conversion in separate VFs so the inverse UB write is
            # ordered before its consumer, without changing any GM DMA.
            with T.SimdVF(lanes=64):
                for row, j in T.Parallel(num_per_tokens, block_k):
                    value = input_value(x_ub, sf_in_ub, group * num_per_tokens + row, j)
                    out_ub[group * num_per_tokens + row, j] = T.cast(value * inverse_ub[j], out_config.dtype)

    @T.macro
    def pack_sf_rows(unpacked_sf_ub, packed_sf_ub, unpacked_row, packed_row):
        with T.SimdVF(lanes=64):
            for j in T.Parallel(block_k * pack_factor):
                packed_sf_ub[packed_row, j] = unpacked_sf_ub[unpacked_row + j % pack_factor, j // pack_factor]

    @T.prim_func
    def per_channel_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_out_shape, (sf_stride, 1), out_config.sf_dtype],
        x_sf_invs: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), sf_in_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            versions = {x_ub: num_stages, out_ub: num_stages}

            if in_config.with_sf:
                sf_in_ub = T.alloc_shared(sf_in_ub_shape, sf_in_dtype)
                dequant_ub = T.alloc_shared((num_per_tokens, quant_chunk_size), dequant_ub_dtype)
                versions[sf_in_ub] = num_stages

            sf_out_ub_dtype = T.uint8 if out_config.use_packed_ue8m0 else T.float32
            aligned_block_k = align(block_k, vector_width // sf_out_ub_dtype.bytes)
            sf_out_ub = T.alloc_shared((sf_rows_per_group * num_blocks_per_group, aligned_block_k), sf_out_ub_dtype)
            versions[sf_out_ub] = 1 if out_config.use_packed_ue8m0 else num_stages
            if out_config.use_packed_ue8m0:
                packed_sf_ub = T.alloc_shared((sf_rows_per_group, block_k * pack_factor), T.uint8)
                versions[packed_sf_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for wave_id in T.Persistent(
                [num_waves],
                1,
                0,
                group_size=1,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [buf for buf, n in versions.items() if n != 1]},
            ):
                # Map `wave_id` into (`group_id`, `block_id_in_group`)
                group_id = (wave_id // num_blocks_per_group) * num_vec_cores + core_id
                block_id_in_group = wave_id % num_blocks_per_group

                if group_id < num_groups:
                    # Map (`group_id`, `block_id_in_group`) to (`m_block_id`, `k_block_id`)
                    m_group_id = group_id // num_k_blocks
                    k_block_id = group_id % num_k_blocks
                    m_block_id = m_group_id * num_blocks_per_group + block_id_in_group

                    col = k_block_id * block_k
                    if num_blocks_per_group == 1 or m_block_id * block_m < num_tokens:
                        T.copy(x[m_block_id * block_m, col], x_ub, l2_cache_ctrl='NOTALLOC_KEEP')
                        if in_config.with_sf:
                            if in_config.use_tma_aligned_col_major_sf:
                                T.copy(
                                    x_sf_invs[col // sf_in_block_k, m_block_id * block_m],
                                    sf_in_ub[: block_k // sf_in_block_k, :block_m],
                                )
                            else:
                                T.copy(
                                    x_sf_invs[m_block_id * block_m, col // num_per_channels],
                                    sf_in_ub[:block_m, : block_k // num_per_channels],
                                )
                        quantize_block(
                            x_ub,
                            out_ub,
                            sf_out_ub,
                            sf_in_ub if in_config.with_sf else None,
                            dequant_ub if in_config.with_sf else None,
                            sf_row_base=block_id_in_group * sf_rows_per_group,
                        )
                        T.copy(out_ub, out[m_block_id * block_m, col])

                    if block_id_in_group == num_blocks_per_group - 1:
                        sf_row = m_group_id * sf_rows_per_group
                        # Pack two stage output sf together if using ue8m0 scale factor output
                        if out_config.use_packed_ue8m0:
                            for packed_row in T.serial(sf_rows_per_group):
                                if m_group_id * (block_m * num_blocks_per_group) + 2 * packed_row * num_per_tokens < num_tokens:
                                    pack_sf_rows(sf_out_ub, packed_sf_ub, 2 * packed_row, packed_row)
                            valid_rows = T.alloc_var(T.int32, init=T.min(sf_rows_per_group, sf_out_shape[0] - sf_row))
                            T.copy(packed_sf_ub[0:valid_rows, :], out_sf[sf_row, col * pack_factor])
                        else:
                            T.copy(sf_out_ub, out_sf[sf_row, col])

    return per_channel_cast_kernel
