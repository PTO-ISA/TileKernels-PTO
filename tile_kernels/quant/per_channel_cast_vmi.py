import math

import tilelang

from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V
from tile_kernels.quant.common import get_packed_ue8m0_pack_factor


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_per_channel_cast_kernel_vmi(
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
    def compute_sf_and_inv(amax, clamp_v, qmax_v, inv_q, u1, u254, mask, lanes):
        clamped_amax = V.vmax(amax, clamp_v)
        if not out_config.round_sf:
            sf = V.vdiv(clamped_amax, qmax_v, mask, precision='exact')
            sf_inv = V.vdiv(qmax_v, clamped_amax, mask, precision='exact')
            return sf, sf_inv
        raw_sf = V.vmul(clamped_amax, inv_q)
        raw_sf_bits = V.vinterpret_cast(raw_sf, 'uint32')
        sf_exponent = V.vadds(V.vshrs(V.vsub(raw_sf_bits, u1), 23, mask), T.uint32(1), mask)
        sf_inv_exponent = V.vsub(u254, sf_exponent)
        sf_inv = V.vinterpret_cast(V.vshls(sf_inv_exponent, 23, mask), 'float32')
        if out_config.use_packed_ue8m0:
            return V.vcvt(sf_exponent, 'uint8', saturate='SAT'), sf_inv
        sf = V.vinterpret_cast(V.vshls(sf_exponent, 23, mask), 'float32')
        return sf, sf_inv

    @T.macro
    def quantize_block(x_ub, out_ub, sf_out_ub, sf_in_ub, dequant_ub, sf_row_base=0):
        with T.SimdVF():
            # VMI owns the physical layout; stage only what makes logical row vectors uniform across input SF layouts.
            mask64 = V.create_mask(64, size=64)
            mask128 = V.create_mask(128, size=128)
            f32_lanes = 128 if use_bf16_compute else 64
            f32_mask = mask128 if use_bf16_compute else mask64
            f32 = V.vreg(f32_lanes, T.float32)
            amax = V.alloc_local((num_vecs // compute_vec_factor,), f32)
            sf_inv = V.alloc_local((num_vecs // compute_vec_factor,), f32)
            clamp_v = V.vbrc(T.float32(out_config.clamp_min_value), size=f32_lanes)
            qmax_v = V.vbrc(T.float32(quant_max), size=f32_lanes)
            inv_q = V.vbrc(T.float32(1.0 / quant_max), size=f32_lanes)
            u1 = V.vbrc(T.uint32(1), size=f32_lanes)
            u254 = V.vbrc(T.uint32(254), size=f32_lanes)

            if use_bf16_compute:
                bf16_abs_mask = V.vbrc(T.uint16(0x7FFF), size=128)
                if in_config.with_sf:
                    sf_lane = V.vshrs(V.vci(T.uint16(0), size=128), 5, mask128)
                    if in_config.use_tma_aligned_col_major_sf:
                        sf_flat_ub = T.view(sf_in_ub, (block_k // sf_in_block_k * block_m * pack_factor,), T.uint8)
                        sf_offsets = V.vadd(
                            V.vmuls(V.vshrs(sf_lane, 1, mask128), block_m * pack_factor, mask128),
                            V.vand(sf_lane, V.vbrc(T.uint16(1), size=128)),
                        )
                        sf_mask = mask128
                        sf_col_stride = block_m * pack_factor
                        sf_row_stride = pack_factor
                    else:
                        sf_flat_ub = T.view(sf_in_ub, (sf_in_rows * sf_in_cols,), T.uint8)
                        sf_offsets = sf_lane
                        sf_mask = mask128
                        sf_col_stride = 1
                        sf_row_stride = sf_in_cols
            else:
                sf_idx = V.vshrs(V.vci(T.uint32(0), size=64), 5, mask64)

            for sf_row in T.serial(sf_rows_per_group):
                row_base = sf_row * num_per_tokens
                # Dequantize and accumulate amax
                for chunk in T.serial(block_k // quant_chunk_size):
                    chunk_base = chunk * quant_chunk_size
                    for vec in T.unroll(num_vecs // compute_vec_factor, explicit=True):
                        vec_col = vec * 64 * compute_vec_factor
                        if use_bf16_compute:
                            # bf16 path: accumulate amax at 128 lanes, then widen to f32.
                            acc_bf16 = V.alloc_var(T.bfloat16, size=128)
                            acc_bf16 = V.vbrc(T.bfloat16(0.0), size=128)
                            if in_config.with_sf:
                                if in_config.use_tma_aligned_col_major_sf:
                                    sf_col = chunk * (quant_chunk_size // 64) + vec * 2
                                else:
                                    sf_col = chunk * (quant_chunk_size // 32) + vec * 4
                            for row in T.serial(num_per_tokens):
                                if in_config.with_sf:
                                    f_pair = V.alloc_local((2,), V.vreg(64, T.float32))
                                    f_pair[0] = V.vcvt(V.vload(x_ub[row_base + row, chunk_base + vec_col], size=64), 'float32')
                                    f_pair[1] = V.vcvt(V.vload(x_ub[row_base + row, chunk_base + vec_col + 64], size=64), 'float32')
                                    _, raw_bf16 = V.vdintlv(
                                        V.vinterpret_cast(f_pair[0], 'bfloat16'),
                                        V.vinterpret_cast(f_pair[1], 'bfloat16'),
                                        mask128,
                                    )
                                    exp_u16 = V.vgather(
                                        sf_flat_ub[(row_base + row) * sf_row_stride + sf_col * sf_col_stride],
                                        sf_offsets, sf_mask)
                                    scale_bf16 = V.vinterpret_cast(V.vshls(exp_u16, 7, mask128), 'bfloat16')
                                    values_bf16 = V.vmul(raw_bf16, scale_bf16)
                                    V.vstore(values_bf16, dequant_ub[row, vec_col])
                                else:
                                    values_bf16 = V.vload(x_ub[row_base + row, chunk_base + vec_col], size=128)
                                acc_bf16 = V.vmax(
                                    acc_bf16,
                                    V.vinterpret_cast(
                                        V.vand(V.vinterpret_cast(values_bf16, 'uint16'), bf16_abs_mask), 'bfloat16'))
                            amax[vec] = V.vcvt(acc_bf16, 'float32')
                        else:
                            # f32 path: unpack e4m3, apply the input channel scale,
                            # accumulate absolute amax, and stage for the output pass.
                            acc_f32 = V.alloc_var(T.float32, size=64)
                            acc_f32 = V.vbrc(T.float32(0.0), size=64)
                            if in_config.use_tma_aligned_col_major_sf:
                                sf_offsets = V.vadds(V.vmuls(sf_idx, block_m, mask64), (chunk * 8 + vec * 2) * block_m, mask64)
                            else:
                                sf_offsets = V.vadds(sf_idx, vec * 2, mask64)
                            for row in T.serial(num_per_tokens):
                                raw_f32 = V.vcvt(V.vload(x_ub[row_base + row, chunk_base + vec_col], size=64), 'float32')
                                if in_config.use_tma_aligned_col_major_sf:
                                    input_scale = V.vgather(sf_in_ub[0, 0], V.vadds(sf_offsets, row_base + row, mask64), mask64)
                                else:
                                    input_scale = V.vgather(sf_in_ub[row_base + row, 0], sf_offsets, mask64)
                                dequant_f32 = V.vmul(raw_f32, input_scale)
                                V.vstore(dequant_f32, dequant_ub[row, vec_col])
                                acc_f32 = V.vmax(acc_f32, V.vabs(dequant_f32))
                            amax[vec] = acc_f32
                    # Mem bar on dequant_ub
                    S.mem_bar('VST_VLD')
                    for vec in T.unroll(num_vecs // compute_vec_factor, explicit=True):
                        # Write sf_out_ub
                        sf, sf_inv[vec] = compute_sf_and_inv(amax[vec], clamp_v, qmax_v, inv_q, u1, u254, f32_mask, f32_lanes)
                        V.vstore(sf, sf_out_ub[sf_row_base + sf_row, chunk_base + vec * 64 * compute_vec_factor])

                        # Calculate and write out_ub
                        vec_col = vec * 64 * compute_vec_factor
                        for row in T.serial(num_per_tokens):
                            if use_bf16_compute:
                                value_ub, value_row, value_col = (dequant_ub, row, 0) if in_config.with_sf else (x_ub, row_base + row, chunk_base)
                                dequant_f32 = V.vcvt(V.vload(value_ub[value_row, value_col + vec_col], size=128), 'float32')
                            else:
                                dequant_f32 = V.vload(dequant_ub[row, vec_col], size=64)
                            scaled = V.vmul(dequant_f32, sf_inv[vec])
                            V.vstore(
                                V.vcvt(scaled, 'float8_e4m3fn', rounding='R', saturate='SAT'),
                                out_ub[row_base + row, chunk_base + vec_col])

    @T.macro
    def pack_sf_rows(unpacked_sf_ub, packed_sf_ub, unpacked_row, packed_row):
        with T.SimdVF():
            mask = V.create_mask(vector_width, size=vector_width)
            for col in T.serial(0, block_k, vector_width):
                packed_0, packed_1 = V.vintlv(
                    V.vload(unpacked_sf_ub[unpacked_row, col], size=vector_width),
                    V.vload(unpacked_sf_ub[unpacked_row + 1, col], size=vector_width), mask)
                V.vstore(packed_0, packed_sf_ub[packed_row, col * 2])
                if block_k - col >= vector_width:
                    V.vstore(packed_1, packed_sf_ub[packed_row, col * 2 + vector_width])

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
