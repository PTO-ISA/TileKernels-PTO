import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape
from tile_kernels.quant.rstd_vmi import make_rstd_impl
from tile_kernels.utils import align, ceil_div


@T.macro
def load_f32(buffer, col, size):
    """Widen `size` contiguous elements into one FP32 vector."""
    if buffer.dtype == T.bfloat16:
        return V.vcvt(V.vload(buffer[col], size=size), 'float32')
    return V.vload(buffer[col], size=size)


@T.macro
def get_sf_and_inv_simd(amax, out_config: CastOutputConfig):
    """SIMD counterpart of get_sf_and_inv; packed SF is returned in uint32 lanes."""
    mask64 = V.create_mask(64, size=64)
    clamped = V.vmaxs(amax, out_config.clamp_min_value, mask64)
    max_value = float(T.max_value(out_config.dtype))
    if not out_config.round_sf:
        maximum = V.vbrc(T.float32(max_value), size=64)
        return V.vdiv(clamped, maximum, mask64, precision='exact'), V.vdiv(maximum, clamped, mask64, precision='exact')
    bits = V.vinterpret_cast(V.vmuls(clamped, 1.0 / max_value, mask64), 'uint32')
    exponent = V.vadds(V.vshrs(V.vsub(bits, V.vbrc(T.uint32(1), size=64), mask64), 23, mask64), T.uint32(1), mask64)
    inverse = V.vinterpret_cast(V.vshls(V.vsub(V.vbrc(T.uint32(254), size=64), exponent, mask64), 23, mask64), 'float32')
    if out_config.use_packed_ue8m0:
        return exponent, inverse
    return V.vinterpret_cast(V.vshls(exponent, 23, mask64), 'float32'), inverse


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_norm_forward_kernel_vmi(
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

        fixed_bytes = row_bytes * with_weight + (vec_size + rstd_out_size) * T.float32.bytes
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
    # Packed FP4 params cannot carry symbolic outer strides (PTO adapter), and a
    # batched contiguous FP4 output has batch stride num_tokens * hidden: keep
    # num_tokens as the compile-time int for FP4 (one kernel per token count).
    if out_config.dtype != T.float4_e2m1fn:
        num_tokens = T.dynamic('num_tokens')
    batch = T.dynamic('batch')
    x_stride = T.dynamic('x_stride')
    x_batch_stride = T.dynamic('x_batch_stride')
    out_stride = T.dynamic('out_stride')
    out_batch_stride = T.dynamic('out_batch_stride')
    if out_config.dtype == T.float4_e2m1fn:
        # The PTO adapter rejects any symbolic outer stride on packed FP4
        # buffers, so dynamic-stride outputs are impossible for FP4; declare
        # the output contiguous (per_token_cast_vmi/cast_back_vmi do the same).
        out_annotation = T.Tensor[(batch, num_tokens, hidden), out_config.dtype]
    else:
        out_annotation = T.StridedTensor[(batch, num_tokens, hidden), (out_batch_stride, out_stride, 1), out_config.dtype]
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
    def clear_padding(buffer):
        # No own T.SimdVF(): the only call sites are inside the kernel body's
        # SimdVF block, and PTO forbids nested vecscopes (ASC tolerated them).
        full = V.create_mask(load_size, size=load_size)
        if buffer.dtype == T.bfloat16:
            zero = V.vbrc(T.bfloat16(0.0), size=load_size)
            base = V.vci(T.int16(0), size=load_size)
        else:
            zero = V.vbrc(T.float32(0.0), size=load_size)
            base = V.vci(T.int32(0), size=load_size)
        for vector in T.serial(T.uint16(hidden // load_size), T.uint16(padded_hidden // load_size)):
            indices = V.vadds(base, vector * load_size, full)
            mask = V.vcmps(indices, hidden, full, 'ge')
            for row in T.serial(T.uint16(rows_per_tile)):
                V.vstore(zero, buffer[row * padded_hidden + vector * load_size], mask)

    compute_rstd = make_rstd_impl(hidden, eps)

    @T.macro
    def reduce_row(x_ub, residual_ub, row_start):
        # ASC widens each 128-element register into two 64-lane halves (vcvt
        # part=0/1); VMI widens it into one 128-lane vector, so one accumulator
        # covers both halves.
        mask128 = V.create_mask(128, size=128)
        num_chunks = reduction_cols // 128
        sums = V.alloc_local((num_chunks,), V.vreg(128, T.float32))
        for i in T.unroll(num_chunks, explicit=True):
            sums[i] = V.vbrc(T.float32(0.0), size=128)
        for group in T.serial(T.uint16(T.ceildiv(padded_hidden, reduction_cols))):
            for chunk in T.unroll(num_chunks, explicit=True):
                col = group * reduction_cols + chunk * 128
                if col < padded_hidden:
                    x = V.vload(x_ub[row_start + col], size=128)
                    if with_residual:
                        x = V.vadd(x, V.vload(residual_ub[row_start + col], size=128), mask128)
                        V.vstore(x, residual_ub[row_start + col], mask128)
                    if x_ub.dtype == T.bfloat16:
                        x = V.vcvt(x, 'float32')
                    sums[chunk] = V.vmula(sums[chunk], x, x, mask128)
        for i in T.unroll(1, num_chunks, explicit=True):
            sums[0] = V.vadd(sums[0], sums[i], mask128)
        return V.vcadd(sums[0], mask128, reassoc=True)

    @T.macro
    def apply_tile(x_ub, weight_ub, out_ub, rstd_source):
        mask128 = V.create_mask(128, size=128)
        for pair in T.serial(T.uint16(padded_hidden // 128)):
            col = pair * 128
            if with_weight:
                w = load_f32(weight_ub, col, 128)
                if out_scale != 1.0:
                    w = V.vmuls(w, out_scale, mask128)
            for row in T.serial(T.uint16(rows_per_tile)):
                offset = row * padded_hidden + col
                # The broadcast source must be a vector load: a bare scalar read
                # of rstd_source[row] bypasses the VST_VLD ordering (and the
                # version rotation), so the first user raced the rstd overwrite
                # and read the raw square sum. vload stays on the ordered pipe.
                inv_rms = V.vbrc(V.vload(rstd_source[row], size=1), size=128, group=1)
                y = V.vmul(load_f32(x_ub, offset, 128), inv_rms, mask128)
                if with_weight:
                    y = V.vmul(y, w, mask128)
                elif out_scale != 1.0:
                    y = V.vmuls(y, out_scale, mask128)
                if out_ub.dtype == T.bfloat16:
                    V.vstore(V.vcvt(y, 'bfloat16', rounding='R'), out_ub[offset], mask128)
                else:
                    V.vstore(y, out_ub[offset], mask128)

    @T.macro
    def quantize_tile(source, quantized, auxiliary, scratch, scales, scale_row_base, mega_packed, mega_dma):
        with T.SimdVF():
            mask4 = V.create_mask(4, size=4)
            mask64 = V.create_mask(64, size=64)
            mask128 = V.create_mask(128, size=128)
            for row in T.serial(T.uint16(rows_per_tile)):
                for vector in T.serial(T.uint16(T.ceildiv(num_groups, 4))):
                    col = row * padded_hidden + vector * 128
                    value = V.vabs(load_f32(source, col, 128), mask128)
                    # One FP32 vector contains four per-32 scaling groups.
                    V.vstore(V.vcmax(value, mask128, group=4), scratch[row, vector * 4], mask4)
            T.simd.mem_bar('VST_VLD')
            for row in T.serial(T.uint16(rows_per_tile)):
                for group_tile in T.serial(T.uint16(sf_size // vec_size)):
                    group = group_tile * vec_size
                    # Amax is dead after this load; reuse its storage for inverse scales.
                    sf, inv = get_sf_and_inv_simd(V.vload(scratch[row, group], size=64), out_config)
                    V.vstore(inv, scratch[row, group])
                    if out_config.use_packed_ue8m0:
                        V.vstore(V.vcvt(sf, 'uint8'), scales[scale_row_base + row, group], mask64)
                    else:
                        V.vstore(sf, scales[scale_row_base + row, group])
                    if with_mega_sf:
                        exponent = sf if out_config.use_packed_ue8m0 else V.vshrs(V.vinterpret_cast(sf, 'uint32'), 23, mask64)
                        V.vstore(V.vcvt(exponent, 'uint8'), mega_packed[row, group], mask64)
            T.simd.mem_bar('VST_VLD')
            if with_mega_sf:
                packed_words = T.Tensor((rows_per_tile, sf_size // T.uint32.bytes), T.uint32, mega_packed.data)
                offsets = V.vmuls(V.vinterpret_cast(V.vci(T.int32(0), size=64), 'uint32'), mega_dma_stride, mask64)
                for row in T.serial(T.uint16(rows_per_tile)):
                    for chunk in T.serial(T.uint16(sf_size // (vec_size * T.uint32.bytes))):
                        V.vscatter(V.vload(packed_words[row, chunk * vec_size], size=64), mega_dma[row, chunk * vec_size, 0], offsets, mask64)
            for row in T.serial(T.uint16(rows_per_tile)):
                for vector in T.serial(T.uint16(padded_hidden // 128)):
                    col = vector * 128
                    group = vector * 4
                    # One grouped broadcast covers the vector's four per-32 groups.
                    inv = V.vbrc(V.vload(scratch[row, group], size=4), size=128, group=4)
                    value = load_f32(source, row * padded_hidden + col, 128)
                    if with_out_bf16 and x_dtype != T.bfloat16:
                        V.vstore(V.vcvt(value, 'bfloat16', rounding='R'), auxiliary[row * padded_hidden + col], mask128)
                    scaled = V.vmul(value, inv, mask128)
                    if out_config.dtype == T.float4_e2m1fn:
                        # VMI converts to packed FP4 only from BF16. Round the
                        # product to odd (sticky bit = "dropped bits were
                        # nonzero") with vunzip so the FP4 rounding sees the
                        # exact FP32 bits.
                        lo, hi = V.vunzip(V.vinterpret_cast(scaled, 'uint32'), 'uint16')
                        packed = V.vinterpret_cast(V.vor(hi, V.vmins(lo, T.uint16(1), mask128), mask128), 'bfloat16')
                        V.vstore(V.vcvt(packed, 'float4_e2m1fn', rounding='R'), quantized[row, col], mask64)
                    else:
                        V.vstore(V.vcvt(scaled, 'float8_e4m3fn', rounding='R', saturate='SAT'), quantized[row, col], mask128)

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
                    # Byte-granular form of ASC's ui32->ui8 copy (the call site
                    # passes mega_dma_bytes / mega_sf_bytes, both ui8 views).
                    T.copy(
                        mega_dma[row, : num_groups // T.uint32.bytes, : T.uint32.bytes],
                        mega_sf[:, mega_row * T.uint32.bytes : (mega_row + 1) * T.uint32.bytes],
                    )

    @T.prim_func
    def norm_forward_kernel_vmi(
        x: T.StridedTensor[(batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype],
        out: out_annotation,
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
            # Byte views on both sides of the mega SF store: ASC copies ui32 DMA
            # lanes into a ui32 view of the ui8 GM table, but PTO's mte requires
            # matching element widths. Same word layout, expressed in bytes.
            mega_dma_bytes = T.view(
                mega_dma_ub, (rows_per_tile, sf_size // T.uint32.bytes, mega_dma_stride * T.uint32.bytes), T.uint8
            )
            mega_sf_bytes = T.StridedTensor(
                (hidden // 128, mega_moe_sf_rows * T.uint32.bytes),
                (mega_moe_sf_rows * T.uint32.bytes, 1),
                T.uint8,
                data=mega_moe_sf.data,
            )
            if transpose_sf:
                transposed_sf_ub = T.alloc_shared((sf_size // sf_pack, sf_rows), sf_word_dtype)
                transpose_indices_ub = T.alloc_shared((sf_lanes,), sf_word_dtype)
                scale_words = T.Tensor((sf_rows, sf_size // sf_pack), sf_word_dtype, scales_ub.data)
                # PTO's mte copies require matching element widths on both sides,
                # and a GM-side word view still lowers as the underlying ui8
                # pointer, so express the transposed copy in bytes (ui8<->ui8).
                transposed_sf_bytes = T.view(transposed_sf_ub, (sf_size // sf_pack, sf_rows * sf_pack), T.uint8)
                out_sf_bytes = T.StridedTensor(
                    (batch, num_sf_words, num_tokens * sf_pack),
                    (out_sf_batch_stride, out_sf_stride, 1),
                    T.uint8,
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
                with T.SimdVF():
                    mask_lanes = V.create_mask(sf_lanes, size=sf_lanes)
                    lane = V.vinterpret_cast(
                        V.vci(T.int16(0) if sf_word_dtype.bits == 16 else T.int32(0), size=sf_lanes),
                        f'{sf_word_dtype}',
                    )
                    row = V.vand(lane, V.vbrc(sf_word_dtype(sf_rows - 1), size=sf_lanes), mask_lanes)
                    pack = V.vshrs(lane, sf_rows.bit_length() - 1, mask_lanes)
                    V.vstore(V.vadd(V.vmuls(row, sf_size // sf_pack, mask_lanes), pack), transpose_indices_ub[0])
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
                        )
                        if with_residual:
                            T.copy(
                                residual[batch_id, token_start : token_start + rows_per_tile, :hidden],
                                T.reshape(residual_ub, (rows_per_tile, padded_hidden))[:, :hidden],
                            )
                        with T.SimdVF():
                            if hidden != padded_hidden:
                                clear_padding(x_ub)
                                if with_residual:
                                    clear_padding(residual_ub)
                                T.simd.mem_bar('VST_VLD')
                            for row in T.serial(T.uint16(rows_per_tile)):
                                # vcadd reduces to a one-lane vector, so this store is
                                # ASC's ONEPT single-element write of the row sum.
                                square_sum = reduce_row(x_ub, residual_ub, row * padded_hidden)
                                V.vstore(square_sum, rstd_ub[row])
                            T.simd.mem_bar('VST_VLD')
                            rstd_values = compute_rstd(V.vload(rstd_ub[0], size=64))
                            V.vstore(rstd_values, rstd_ub[0])
                            if rows_per_tile * T.float32.bytes >= 32:
                                V.vstore(rstd_values, rstd_out_ub[row_base], V.create_mask(rows_per_tile, size=64))
                            else:
                                # Short groups: a row_base-strided masked store is not
                                # provably 32-byte aligned (the ptoas masked-store
                                # contract), so write the rows with a scatter, whose
                                # per-lane addresses carry no alignment requirement.
                                V.vscatter(
                                    rstd_values,
                                    rstd_out_ub[0],
                                    V.vinterpret_cast(V.vci(T.int32(row_base), size=64), 'uint32'),
                                    V.create_mask(rows_per_tile, size=64),
                                )
                            T.simd.mem_bar('VST_VLD')
                            apply_tile(residual_ub if with_residual else x_ub, weight_ub, out_ub, rstd_ub)
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
                            store_tile_scales(scales_ub, out_sf, mega_sf_bytes, mega_dma_bytes, batch_id, token_start)
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
                    with T.SimdVF():
                        mask_lanes = V.create_mask(sf_lanes, size=sf_lanes)
                        indices = V.vload(transpose_indices_ub[0], size=sf_lanes)
                        for vector in T.unroll(T.ceildiv(num_sf_words * sf_rows, sf_lanes), explicit=True):
                            pack = vector * (sf_lanes // sf_rows)
                            values = V.vgather(scale_words[0, pack], indices, mask_lanes)
                            V.vstore(values, transposed_sf_ub[pack, 0])
                    # Byte-granular form of ASC's word copy (see view declarations).
                    T.copy(
                        transposed_sf_bytes[:num_sf_words, : valid_rows * sf_pack],
                        out_sf_bytes[batch_id, :, token_base * sf_pack : (token_base + valid_rows) * sf_pack],
                    )
                T.copy(rstd_out_ub[:valid_rows], rstd[batch_id, token_base])

    return norm_forward_kernel_vmi
