import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape
from tile_kernels.utils import align, ceil_div


@tilelang.jit(target='pto')
def get_norm_forward_kernel_parallel(
    num_tokens: int,
    hidden: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
    with_out_bf16: bool = False,
    mega_moe_block_m: int = 0,
):
    assert hidden > 0
    assert x_dtype in (T.bfloat16, T.float32)
    assert not out_config.with_sf or out_config.sf_block == (1, 32)
    padded_hidden = align(hidden, 128)
    row_bytes = padded_hidden * x_dtype.bytes
    vec_size = 64
    load_size = 256 // x_dtype.bytes
    num_parts = load_size // vec_size
    num_accumulators = min(4, padded_hidden // vec_size)
    reduction_cols = num_accumulators * vec_size

    # SF layout only affects stores. All formats share normalization and quantization.
    with_mega_sf = mega_moe_block_m > 0
    mega_sf_tile_rows = 128
    mega_sf_lane_rows = 32
    mega_sf_cols = mega_sf_tile_rows // mega_sf_lane_rows
    transpose_sf = out_config.with_sf and out_config.use_tma_aligned_col_major_sf
    num_groups = hidden // 32 if out_config.with_sf else 0
    # Mega SF packs four exponent bytes per word; scatter processes 64 words at a time.
    sf_size = align(max(1, num_groups), vec_size * T.uint32.bytes if with_mega_sf else vec_size)
    # Transpose SF as raw bits: pairs of exponent bytes or individual FP32 scales.
    sf_word_dtype = T.uint16 if out_config.use_packed_ue8m0 else T.uint32
    sf_pack = sf_word_dtype.bytes // out_config.sf_dtype.bytes
    sf_lanes = 256 // sf_word_dtype.bytes
    sf_store_dist = f'NORM_B{sf_word_dtype.bits}'
    num_sf_words = ceil_div(num_groups, sf_pack)
    mega_dma_stride = 32 // T.uint32.bytes

    # Group short RSTD stores and bound each core's work.
    rows_per_core = max(min(num_tokens, 4), num_tokens // num_vec_cores)
    rows_per_block = min(128, 1 << (rows_per_core.bit_length() - 1))
    if transpose_sf:
        rows_per_block = min(sf_lanes // 4, rows_per_block)
    # Target 32 KiB of input per tile, shared by x and residual.
    tile_capacity = max(1, 32768 // (1 + with_residual) // row_bytes)
    rows_per_tile = min(vec_size, rows_per_block, 1 << (tile_capacity.bit_length() - 1))
    # Shrink the tile until all buffers fit in UB, including optional SF outputs.
    while True:
        rows_per_block = min(rows_per_block, (64 if out_config.with_sf else 32) * rows_per_tile)
        sf_rows = max(rows_per_block, 32 // sf_word_dtype.bytes) if transpose_sf else rows_per_tile
        rstd_out_size = max(vec_size, rows_per_block)

        fixed_bytes = (padded_hidden + 1) * 4 + row_bytes * with_weight + (vec_size + rstd_out_size) * T.float32.bytes
        if out_config.with_sf:
            fixed_bytes += rows_per_tile * sf_size * T.float32.bytes
        if transpose_sf:
            fixed_bytes += 2 * sf_rows * sf_size * out_config.sf_dtype.bytes + vec_size * T.float32.bytes
        tile_bytes = rows_per_tile * row_bytes
        stage_bytes = (2 + with_residual) * tile_bytes
        if out_config.with_sf:
            stage_bytes += rows_per_tile * padded_hidden * out_config.dtype.bytes
            if not transpose_sf:
                stage_bytes += rows_per_tile * sf_size * out_config.sf_dtype.bytes
        if with_out_bf16 and x_dtype != T.bfloat16:
            stage_bytes += rows_per_tile * padded_hidden * T.bfloat16.bytes
        if with_mega_sf:
            fixed_bytes += rows_per_tile * sf_size * T.uint8.bytes
            stage_bytes += rows_per_tile * (sf_size // T.uint32.bytes) * 32
        available_bytes = get_max_ub_per_vector_core(use_simt=False) - fixed_bytes
        num_stages = min(4, available_bytes // stage_bytes)
        reused_stages = min(4, available_bytes // (stage_bytes - tile_bytes))
        reuse_input = num_stages < 3 and reused_stages > num_stages
        if reuse_input:
            num_stages = reused_stages
        if num_stages > 0:
            break
        assert rows_per_tile > 1, 'norm tile exceeds the available UB'
        rows_per_tile //= 2

    # Batches with the same tiling share compiled code, independent of row strides.
    num_tokens = T.dynamic('num_tokens')
    batch = T.dynamic('batch')
    x_stride = T.dynamic('x_stride')
    x_batch_stride = T.dynamic('x_batch_stride')
    out_stride = T.dynamic('out_stride')
    out_batch_stride = T.dynamic('out_batch_stride')
    residual_stride = T.dynamic('residual_stride')
    residual_batch_stride = T.dynamic('residual_batch_stride')
    residual_out_stride = T.dynamic('residual_out_stride')
    residual_out_batch_stride = T.dynamic('residual_out_batch_stride')

    out_bf16_stride = T.dynamic('out_bf16_stride')
    out_bf16_batch_stride = T.dynamic('out_bf16_batch_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    out_sf_batch_stride = T.dynamic('out_sf_batch_stride')
    mega_moe_sf_rows = T.dynamic('mega_moe_sf_rows')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config) if out_config.with_sf else (1, 1)

    @T.macro
    def quantize_tile(source, quantized, auxiliary, scratch, scales, scale_row_base, mega_packed, mega_dma):
        with T.SimdVF(lanes=128):
            for row in T.serial(rows_per_tile):
                for group in T.Parallel(num_groups):
                    maximum = T.alloc_var(T.float32, init=0.0)
                    for k in T.serial(32):
                        maximum = T.max(maximum, T.abs(T.float32(source[row * padded_hidden + group * 32 + k])))
                    maximum = T.max(maximum, out_config.clamp_min_value)
                    if out_config.round_sf:
                        bits = T.reinterpret(maximum * (1.0 / float(T.max_value(out_config.dtype))), T.uint32)
                        exponent = ((bits - T.uint32(1)) >> 23) + T.uint32(1)
                        inverse = T.reinterpret((T.uint32(254) - exponent) << 23, T.float32)
                        scale = T.reinterpret(exponent << 23, T.float32)
                    else:
                        scale = maximum / float(T.max_value(out_config.dtype))
                        inverse = float(T.max_value(out_config.dtype)) / maximum
                    scratch[row, group] = inverse
                    if out_config.use_packed_ue8m0:
                        scales[scale_row_base + row, group] = T.cast(exponent, T.uint8)
                    else:
                        scales[scale_row_base + row, group] = scale
                    if with_mega_sf:
                        mega_packed[row, group] = T.cast(T.reinterpret(scale, T.uint32) >> 23, T.uint8)
                for j in T.Parallel(padded_hidden):
                    value = T.float32(source[row * padded_hidden + j])
                    if with_out_bf16 and x_dtype != T.bfloat16:
                        auxiliary[row * padded_hidden + j] = T.cast(value, T.bfloat16)
                    quantized[row, j] = T.cast(value * scratch[row, j // 32], out_config.dtype)
            if with_mega_sf:
                packed_words = T.Tensor((rows_per_tile, sf_size // T.uint32.bytes), T.uint32, mega_packed.data)
                for row in T.serial(rows_per_tile):
                    for group in T.serial(num_groups // T.uint32.bytes):
                        for point in T.Parallel(1):
                            mega_dma[row, group, point] = packed_words[row, group + point]

    @T.macro
    def store_tile_scales(scales, out_sf, mega_sf, mega_dma, batch_id, token_start):
        if not transpose_sf:
            T.copy(scales[:, :num_groups], out_sf[batch_id, token_start : token_start + rows_per_tile, :num_groups])
        if with_mega_sf:
            for row in T.serial(rows_per_tile):
                token = token_start + row
                if token < num_tokens:
                    i = token % mega_moe_block_m
                    mega_row = (
                        token // mega_moe_block_m * align(mega_moe_block_m, mega_sf_tile_rows)
                        + i // mega_sf_tile_rows * mega_sf_tile_rows
                        + i % mega_sf_lane_rows * mega_sf_cols
                        + i % mega_sf_tile_rows // mega_sf_lane_rows
                    )
                    T.copy(mega_dma[row, : num_groups // T.uint32.bytes, :1], mega_sf[:, mega_row : mega_row + 1])

    @T.prim_func
    def norm_forward_kernel(
        x: T.StridedTensor[(batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype],
        out: T.StridedTensor[(batch, num_tokens, hidden), (out_batch_stride, out_stride, 1), out_config.dtype],
        out_sf: T.StridedTensor[(batch, *sf_shape), (out_sf_batch_stride, out_sf_stride, 1), out_config.sf_dtype],
        out_bf16: T.StridedTensor[(batch, num_tokens, hidden), (out_bf16_batch_stride, out_bf16_stride, 1), T.bfloat16],
        mega_moe_sf: T.StridedTensor[(mega_moe_sf_rows, hidden // 128, 4), (4, mega_moe_sf_rows * 4, 1), T.uint8],
        weight: T.Tensor[(hidden,), x_dtype],
        rstd: T.Tensor[(batch, num_tokens), T.float32],
        residual: T.StridedTensor[(batch, num_tokens, hidden), (residual_batch_stride, residual_stride, 1), x_dtype],
        residual_out: T.StridedTensor[(batch, num_tokens, hidden), (residual_out_batch_stride, residual_out_stride, 1), x_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            out_ub = x_ub if reuse_input else T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            weight_ub = T.alloc_shared((padded_hidden,), x_dtype)
            residual_ub = T.alloc_shared((rows_per_tile * padded_hidden,), x_dtype)
            out_bf16_ub = out_ub if x_dtype == T.bfloat16 else T.alloc_shared((rows_per_tile * padded_hidden,), T.bfloat16)
            rstd_ub = T.alloc_shared((vec_size,), T.float32)
            rstd_out_ub = T.alloc_shared((rstd_out_size,), T.float32)
            quantized_ub = T.alloc_shared((rows_per_tile, padded_hidden), out_config.dtype)
            sf_scratch_ub = T.alloc_shared((rows_per_tile, sf_size), T.float32)
            scales_ub = T.alloc_shared((sf_rows, sf_size), out_config.sf_dtype)
            mega_packed_ub = T.alloc_shared((rows_per_tile, sf_size), T.uint8)
            mega_dma_ub = T.alloc_shared((rows_per_tile, sf_size // T.uint32.bytes, mega_dma_stride), T.uint32)
            mega_sf_words = T.Tensor((hidden // 128, mega_moe_sf_rows), T.uint32, mega_moe_sf.data)
            if transpose_sf:
                transposed_sf_ub = T.alloc_shared((sf_size // sf_pack, sf_rows), sf_word_dtype)
                transpose_indices_ub = T.alloc_shared((sf_lanes,), sf_word_dtype)
                scale_words = T.Tensor((sf_rows, sf_size // sf_pack), sf_word_dtype, scales_ub.data)
                out_sf_words = T.StridedTensor(
                    (batch, num_sf_words, num_tokens),
                    (out_sf_batch_stride // sf_pack, out_sf_stride // sf_pack, 1),
                    sf_word_dtype,
                    data=out_sf.data,
                )
            versions = {
                x_ub: num_stages,
                out_ub: num_stages,
                residual_ub: num_stages if with_residual else 1,
                quantized_ub: num_stages if out_config.with_sf else 1,
                mega_dma_ub: num_stages if with_mega_sf else 1,
                scales_ub: num_stages if out_config.with_sf and not transpose_sf else 1,
                rstd_ub: 1,
                sf_scratch_ub: 1,
                mega_packed_ub: 1,
            }
            if with_out_bf16:
                versions[out_bf16_ub] = num_stages
            T.annotate_buffer_versions(versions)
            if transpose_sf:
                with T.SimdVF(lanes=64):
                    T.evaluate(0)
            if with_weight:
                T.copy(weight, weight_ub[:hidden])
            num_token_blocks = T.ceildiv(num_tokens, rows_per_block)
            # Full last dimension keeps the traversal row-major (batch outer, token inner).
            for batch_id, pid_token in T.Persistent(
                [batch, num_token_blocks], num_vec_cores, core_id, group_size=num_token_blocks, num_stages=num_stages
            ):
                token_base = pid_token * rows_per_block
                for tile_index in T.serial(rows_per_block // rows_per_tile):
                    row_base = tile_index * rows_per_tile
                    token_start = token_base + row_base
                    if token_start < num_tokens:
                        T.copy(
                            x[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            T.reshape(x_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                            l2_cache_ctrl='NOTALLOC_KEEP',
                        )
                        if with_residual:
                            T.copy(
                                residual[batch_id, token_start : token_start + rows_per_tile, :hidden],
                                T.reshape(residual_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                l2_cache_ctrl='NOTALLOC_KEEP',
                            )
                        with T.SimdVF(lanes=64):
                            squared = T.alloc_shared((padded_hidden,), T.float32)
                            total = T.alloc_shared((1,), T.float32)
                            for row in T.serial(rows_per_tile):
                                for j in T.Parallel(padded_hidden):
                                    value = T.if_then_else(j < hidden, T.float32(x_ub[row * padded_hidden + j]), 0.0)
                                    if with_residual:
                                        value = T.float32(
                                            T.cast(value + T.if_then_else(j < hidden, T.float32(residual_ub[row * padded_hidden + j]), 0.0), x_dtype)
                                        )
                                        residual_ub[row * padded_hidden + j] = T.cast(value, x_dtype)
                                    squared[j] = value * value
                                T.reduce_sum(squared, total, dim=0)
                                for point in T.Parallel(1):
                                    inverse = 1.0 / T.sqrt(total[point] * (1.0 / hidden) + eps)
                                    rstd_ub[row + point] = inverse
                                    rstd_out_ub[row_base + row + point] = inverse
                                for j in T.Parallel(padded_hidden):
                                    if with_residual:
                                        value = T.float32(residual_ub[row * padded_hidden + j])
                                    else:
                                        value = T.float32(x_ub[row * padded_hidden + j])
                                    y = value * rstd_ub[row]
                                    if with_weight:
                                        y = y * (T.float32(weight_ub[j]) * out_scale)
                                    else:
                                        y = y * out_scale
                                    out_ub[row * padded_hidden + j] = T.cast(y, x_dtype)
                        if out_config.with_sf:
                            quantize_tile(
                                out_ub,
                                quantized_ub,
                                out_bf16_ub,
                                sf_scratch_ub,
                                scales_ub,
                                row_base if transpose_sf else 0,
                                mega_packed_ub,
                                mega_dma_ub,
                            )
                        T.copy(
                            T.reshape(quantized_ub if out_config.with_sf else out_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                            out[batch_id, token_start : token_start + rows_per_tile, :hidden],
                        )
                        if out_config.with_sf:
                            store_tile_scales(scales_ub, out_sf, mega_sf_words, mega_dma_ub, batch_id, token_start)
                        if with_out_bf16:
                            T.copy(
                                T.reshape(out_bf16_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                out_bf16[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            )
                        if with_residual:
                            T.copy(
                                T.reshape(residual_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                                residual_out[batch_id, token_start : token_start + rows_per_tile, :hidden],
                            )
                valid_rows = T.min(rows_per_block, num_tokens - token_base)
                if transpose_sf:
                    # Gather scale words so each column becomes a contiguous DMA.
                    with T.SimdVF(lanes=64):
                        for pack, row in T.Parallel(num_sf_words, sf_rows):
                            transposed_sf_ub[pack, row] = scale_words[row, pack]
                    T.copy(
                        transposed_sf_ub[:num_sf_words, :valid_rows],
                        out_sf_words[batch_id, :, token_base : token_base + valid_rows],
                    )
                T.copy(rstd_out_ub[:valid_rows], rstd[batch_id, token_base])

    return norm_forward_kernel
