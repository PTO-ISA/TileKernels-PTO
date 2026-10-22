from typing import Any

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.config import get_max_ub_per_vector_core, get_num_vec_cores

SIMD_LANES = 64


@tilelang.jit(target='pto')
def get_rope_kernel_vmi(
    dtype: T.dtype,
    positions_dtype: T.dtype,
    nheads: int,
    rotary_dim: int,
    has_position_tensor: bool,
    interleaved: bool,
    conjugate: bool,
    cos_sin_stride: int,
    x_stride_0: int,
    x_stride_1: int,
) -> Any:
    assert dtype in (T.bfloat16, T.float32)
    assert positions_dtype in (T.int32, T.int64)
    assert rotary_dim in (64, 128)

    half_dim = rotary_dim // 2
    num_vec_cores = get_num_vec_cores()
    dtype_size = dtype.bytes
    bytes_per_row = nheads * rotary_dim * dtype_size + rotary_dim * 4
    max_ub_size = get_max_ub_per_vector_core(use_simt=False)
    num_stages = min(3, max_ub_size // bytes_per_row)
    ub_block_tokens = min(SIMD_LANES, max_ub_size // (num_stages * bytes_per_row))
    block_tokens = 1 << (ub_block_tokens.bit_length() - 1)
    can_bulk_copy = x_stride_0 == nheads * x_stride_1 or x_stride_1 == rotary_dim

    batch = T.dynamic('batch')
    seqlen = T.dynamic('seqlen')
    seqlen_ro = T.dynamic('seqlen_ro')
    x_stride_b = T.dynamic('x_stride_b')
    positions_stride_b = T.dynamic('positions_stride_b')

    @T.macro
    def rotate(
        first: T.float32x64,
        second: T.float32x64,
        cosine: T.float32x64,
        sine: T.float32x64,
        neg_sine: T.float32x64,
    ) -> tuple[T.float32x64, T.float32x64]:
        mask = V.create_mask(SIMD_LANES, size=SIMD_LANES)
        out_first = V.vmula(V.vmul(second, sine), first, cosine, mask)
        out_second = V.vmula(V.vmul(second, cosine), first, neg_sine, mask)
        return out_first, out_second

    @T.macro
    def load2(
        buffer: T.Buffer,
        row: T.int32,
        first_head: T.int32,
        first_column: T.int32,
        second_head: T.int32,
        second_column: T.int32,
    ) -> tuple[T.float32x64, T.float32x64]:
        first = V.vload(buffer[row, first_head, first_column], size=SIMD_LANES)
        second = V.vload(buffer[row, second_head, second_column], size=SIMD_LANES)
        if dtype == T.bfloat16:
            first = V.vcvt(first, 'float32')
            second = V.vcvt(second, 'float32')
        return first, second

    @T.macro
    def store2(
        buffer: T.Buffer,
        row: T.int32,
        first_head: T.int32,
        first_column: T.int32,
        second_head: T.int32,
        second_column: T.int32,
        first: T.float32x64,
        second: T.float32x64,
        has_second: bool,
    ) -> None:
        if dtype == T.bfloat16:
            first = V.vcvt(first, 'bfloat16')
            if has_second:
                second = V.vcvt(second, 'bfloat16')
        V.vstore(first, buffer[row, first_head, first_column])
        if has_second:
            V.vstore(second, buffer[row, second_head, second_column])

    @T.prim_func
    def _rope_kernel(
        x: T.StridedTensor((batch, seqlen, nheads, rotary_dim), (x_stride_b, x_stride_0, x_stride_1, 1), dtype),  # type: ignore
        cos_sin_cache: T.StridedTensor((seqlen_ro, rotary_dim), (cos_sin_stride, 1), T.float32),  # type: ignore
        positions: T.StridedTensor((batch, seqlen), (positions_stride_b, 1), positions_dtype),  # type: ignore
        seqlen_offsets: T.int32,
    ) -> None:
        T.assume(seqlen > 0)
        with T.Kernel(num_vec_cores) as core:
            x_ub = T.alloc_shared((block_tokens, nheads, rotary_dim), dtype)
            cos_sin_ub = T.alloc_shared((block_tokens, rotary_dim), T.float32)
            T.annotate_buffer_versions({x_ub: num_stages, cos_sin_ub: num_stages})

            for batch_id, token_block in T.Persistent(
                [batch, T.ceildiv(seqlen, block_tokens)], num_vec_cores, core, group_size=1, num_stages=num_stages
            ):
                token_start = token_block * block_tokens
                rows = T.min(block_tokens, seqlen - token_start)
                if can_bulk_copy:
                    T.copy(x[batch_id, token_start, 0, 0], x_ub[:rows, :, :])
                else:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            T.copy(x[batch_id, token_start + row, 0, 0], x_ub[row, :, :])
                if has_position_tensor:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            position = positions[batch_id, token_start + row] + seqlen_offsets
                            T.copy(cos_sin_cache[position, 0], cos_sin_ub[row, :])
                else:
                    position = token_start + seqlen_offsets
                    T.copy(cos_sin_cache[position, 0], cos_sin_ub[:rows, :])

                with T.SimdVF():
                    mask = V.create_mask(SIMD_LANES, size=SIMD_LANES)
                    if rotary_dim == SIMD_LANES and interleaved:
                        cos_indices = V.vand(V.vci(T.uint32(0), size=SIMD_LANES), V.vbrc(T.uint32(31), size=SIMD_LANES),)
                        sin_indices = V.vor(cos_indices, V.vbrc(T.uint32(32), size=SIMD_LANES))
                    for row in T.serial(block_tokens):
                        if rotary_dim == SIMD_LANES:
                            cos_sin_values = V.vload(cos_sin_ub[row, 0], size=SIMD_LANES)
                            if interleaved:
                                cosine = V.vgather(cos_sin_ub[row, 0], cos_indices, mask)
                                sine = V.vgather(cos_sin_ub[row, 0], sin_indices, mask)
                            else:
                                cosine, sine = V.vintlv(cos_sin_values, cos_sin_values, mask)
                        else:
                            cosine = V.vload(cos_sin_ub[row, 0], size=SIMD_LANES)
                            sine = V.vload(cos_sin_ub[row, half_dim], size=SIMD_LANES)
                        sine, neg_sine = (sine, V.vneg(sine)) if conjugate else (V.vneg(sine), sine)

                        if rotary_dim == SIMD_LANES:
                            for pair in range(nheads // 2):
                                head = pair * 2
                                first, second = load2(x_ub, row, head, 0, head + 1, 0)
                                first, second = V.vdintlv(first, second, mask) if interleaved else V.vintlv(first, second, mask)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                first, second = V.vintlv(first, second, mask) if interleaved else V.vdintlv(first, second, mask)
                                store2(x_ub, row, head, 0, head + 1, 0, first, second, True)
                            if nheads % 2:
                                odd_head = nheads - 1
                                first, second = load2(x_ub, row, odd_head, 0, odd_head, 0)
                                first, second = V.vdintlv(first, second, mask) if interleaved else V.vintlv(first, second, mask)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                first, second = V.vintlv(first, second, mask) if interleaved else V.vdintlv(first, second, mask)
                                store2(x_ub, row, odd_head, 0, odd_head, 0, first, second, False)
                        else:
                            for head in range(nheads):
                                first, second = load2(x_ub, row, head, 0, head, half_dim)
                                if interleaved:
                                    first, second = V.vdintlv(first, second, mask)
                                first, second = rotate(first, second, cosine, sine, neg_sine)
                                if interleaved:
                                    first, second = V.vintlv(first, second, mask)
                                store2(x_ub, row, head, 0, head, half_dim, first, second, True)
                if can_bulk_copy:
                    T.copy(x_ub[:rows, :, :], x[batch_id, token_start, 0, 0])
                else:
                    for row in T.serial(block_tokens):
                        if row < rows:
                            T.copy(x_ub[row, :, :], x[batch_id, token_start + row, 0, 0])

    return _rope_kernel
