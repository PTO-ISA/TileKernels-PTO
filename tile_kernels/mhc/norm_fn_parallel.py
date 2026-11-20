import tilelang
from tilelang.ascend import language as T

_COMPACT_REDUCE_TOKEN_BLOCK = 16
_COMPACT_REDUCE_SPLIT_CHUNK = 32
_COMPACT_REDUCE_MHC_MULT3 = 24
_COMPACT_REDUCE_VREGS = 6


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_fwd_parallel(
    fn,
    normw,
    out_fn,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn: T.Tensor[(mhc_mult3, hidden), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)

            with T.SimdVF(lanes=128):
                for row, col in T.Parallel(mhc_mult3, hidden_block):
                    out_fn_ub[row, col] = fn_ub[row, col] * normw_ub[col]

            T.copy(out_fn_ub, out_fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_bwd_parallel(
    fn,
    normw,
    out_fn_grad,
    fn_grad,
    normw_grad,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw_grad: T.Tensor[(hidden,), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_grad_ub = T.alloc_shared((hidden_block,), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_grad_ub: num_stages,
                fn_grad_ub: num_stages,
                normw_grad_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)
            T.copy(out_fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], out_fn_grad_ub)
            T.copy(fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_grad_ub)
            T.copy(normw_grad[hidden_offset : hidden_offset + hidden_block], normw_grad_ub)

            with T.SimdVF(lanes=128):
                # One pass over the gradient data per column: the serial row
                # shell keeps the single-read of out_fn_grad shared between the
                # fn_grad update and the per-column normw_grad accumulation.
                for col in T.Parallel(hidden_block):
                    for row in T.serial(mhc_mult3):
                        fn_grad_ub[row, col] = fn_grad_ub[row, col] + out_fn_grad_ub[row, col] * normw_ub[col]
                        normw_grad_ub[col] = normw_grad_ub[col] + out_fn_grad_ub[row, col] * fn_ub[row, col]

            T.copy(fn_grad_ub, fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])
            T.copy(normw_grad_ub, normw_grad[hidden_offset : hidden_offset + hidden_block])


def _compact_reduce_split_chunk(n_splits: int) -> int:
    for split_chunk in (_COMPACT_REDUCE_SPLIT_CHUNK, 16, 8, 4, 2):
        if n_splits % split_chunk == 0:
            return split_chunk
    return 1


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_fwd_parallel(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm fwd currently supports one RMS group'
    assert mhc_mult3 == _COMPACT_REDUCE_MHC_MULT3, 'Ascend compact reduce assumes mhc_mult3=24'

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    token_block = _COMPACT_REDUCE_TOKEN_BLOCK
    split_chunk = _compact_reduce_split_chunk(n_splits)
    compact_vregs = _COMPACT_REDUCE_VREGS
    token_elems = _COMPACT_REDUCE_TOKEN_BLOCK * _COMPACT_REDUCE_MHC_MULT3
    sq_pad64 = 64
    num_stages = 3
    n_token_blocks = T.ceildiv(num_tokens, token_block)
    n_split_chunks = n_splits // split_chunk
    assert n_splits % split_chunk == 0, 'compact reduce requires n_splits divisible by selected split_chunk'

    with T.Kernel(num_vec_cores) as core_id:
        mul_ub = T.alloc_shared((split_chunk, token_block, mhc_mult3), T.float32)
        sq_ub = T.alloc_shared((split_chunk, sq_pad64), T.float32)
        acc_ub = T.alloc_shared((compact_vregs, 64), T.float32)
        sq_acc_ub = T.alloc_shared(64, T.float32)
        mul_out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        sq_out_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                mul_ub: num_stages,
                sq_ub: num_stages,
                acc_ub: 1,
                sq_acc_ub: 1,
                mul_out_ub: 1,
                out_ub: 1,
                sq_out_ub: 1,
            }
        )
        # Zero-cost flat views over the same UB storage: the SimdVF compute
        # addresses the 16x24 token block as one contiguous 384-element run,
        # so full lanes are used without the VMI-era 6x64 vreg bookkeeping.
        acc_flat = T.Tensor((token_elems,), T.float32, data=acc_ub.data, scope='shared.dyn')
        mul_flat = T.Tensor((split_chunk, token_elems), T.float32, data=mul_ub.data, scope='shared.dyn')
        mul_out_flat = T.Tensor((token_elems,), T.float32, data=mul_out_ub.data, scope='shared.dyn')

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            # Ragged tail blocks are handled by the same vectorized path:
            # T.copy clamps out-of-bound GM reads/writes and the padded
            # tokens' stale UB lanes never reach GM, so no separate scalar
            # tail path is needed.
            with T.SimdVF(lanes=64):
                T.clear(acc_flat)
                T.clear(sq_acc_ub)

            for split_chunk_idx in T.serial(n_split_chunks):
                split_base = split_chunk_idx * split_chunk
                T.copy(
                    out_mul_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0, :mhc_mult3],
                    mul_ub,
                )
                T.copy(
                    sqrsum_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0],
                    sq_ub[:, :token_block],
                )

                with T.SimdVF(lanes=64):
                    for elem in T.Parallel(token_elems):
                        for local_split in T.serial(split_chunk):
                            acc_flat[elem] = acc_flat[elem] + mul_flat[local_split, elem]
                    for token_idx in T.Parallel(token_block):
                        for local_split in T.serial(split_chunk):
                            sq_acc_ub[token_idx] = sq_acc_ub[token_idx] + sq_ub[local_split, token_idx]

            with T.SimdVF(lanes=64):
                for lane in T.Parallel(64):
                    sq_out_ub[lane] = sq_acc_ub[lane]
                # Per-token rsqrt scale, staged in the first token_block words
                # of mul_out storage; the final flat overwrite below replaces
                # the staging values with the reduced out_mul payload.
                for token_idx in T.Parallel(token_block):
                    mul_out_flat[token_idx] = T.float32(1.0) / T.sqrt(
                        sq_acc_ub[token_idx] * T.float32(1.0 / rms_group_size) + T.float32(rms_eps)
                    )
                for token_idx, elem in T.Parallel(token_block, _COMPACT_REDUCE_MHC_MULT3):
                    out_ub[token_idx, elem] = (
                        acc_flat[token_idx * _COMPACT_REDUCE_MHC_MULT3 + elem] * mul_out_flat[token_idx]
                    )
                for elem in T.Parallel(token_elems):
                    mul_out_flat[elem] = acc_flat[elem]

            T.copy(mul_out_ub, out_mul[token_base : token_base + token_block, 0, :])
            T.copy(out_ub, out[token_base : token_base + token_block, :])
            T.copy(sq_out_ub[:token_block], sqrsum[token_base : token_base + token_block, 0])


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_bwd_parallel(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm bwd currently supports one RMS group'
    assert mhc_mult3 <= 32, f'Ascend reduce/rmsnorm bwd SimtVF maps mhc_mult3 to one 32-lane vector, got {mhc_mult3}'

    out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32]

    token_block = 16
    mhc_mult3_pad32 = 32
    num_stages = 6
    n_token_blocks = T.ceildiv(num_tokens, token_block)

    with T.Kernel(num_vec_cores) as core_id:
        out_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        sqrsum_ub = T.alloc_shared(token_block, T.float32)
        sqrsum_grad_ub = T.alloc_shared((token_block, 8), T.float32)
        sqrsum_grad_compact_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                out_grad_ub: num_stages,
                out_mul_ub: num_stages,
                out_mul_grad_ub: num_stages,
                sqrsum_ub: num_stages,
                sqrsum_grad_ub: num_stages,
                sqrsum_grad_compact_ub: num_stages,
            }
        )
        # Zero-cost views over existing UB storage for the SimdVF compute:
        # grad_mul_view exposes the logical token x mhc_mult3 block of the
        # 32-padded out_mul_grad rows, row_sum_view is the per-token reduce
        # destination inside the compact output words, and rms_stage reuses
        # the (otherwise scratch) sqrsum_grad_ub words to hold per-token rms.
        grad_mul_view = T.StridedTensor(
            (token_block, mhc_mult3), (mhc_mult3_pad32, 1), T.float32, data=out_mul_grad_ub.data, scope='shared.dyn'
        )
        row_sum_view = T.Tensor((token_block,), T.float32, data=sqrsum_grad_compact_ub.data, scope='shared.dyn')
        rms_stage = T.Tensor((token_block,), T.float32, data=sqrsum_grad_ub.data, scope='shared.dyn')

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            if token_base + token_block <= num_tokens:
                T.copy(out_grad[token_base : token_base + token_block, :mhc_mult3], out_grad_ub[:, :mhc_mult3])
                T.copy(out_mul[token_base : token_base + token_block, 0, :mhc_mult3], out_mul_ub[:, :mhc_mult3])
                T.copy(sqrsum[token_base : token_base + token_block, 0], sqrsum_ub[:token_block])

                with T.SimdVF(lanes=64):
                    # Stage grad*out_mul in the out_mul_grad rows, reduce each
                    # token row to its rms_grad sum, then overwrite the rows
                    # with the real out_mul_grad payload.
                    for token_idx, elem in T.Parallel(token_block, mhc_mult3):
                        grad_mul_view[token_idx, elem] = out_grad_ub[token_idx, elem] * out_mul_ub[token_idx, elem]
                    T.reduce_sum(grad_mul_view, row_sum_view, dim=1)
                    for token_idx in T.Parallel(token_block):
                        rms_stage[token_idx] = T.float32(1.0) / T.sqrt(
                            sqrsum_ub[token_idx] * T.float32(1.0 / rms_group_size) + T.float32(rms_eps)
                        )
                    for token_idx, elem in T.Parallel(token_block, mhc_mult3):
                        grad_mul_view[token_idx, elem] = out_grad_ub[token_idx, elem] * rms_stage[token_idx]
                    # Match the fwd rsqrt chain bitwise: sqrsum_grad =
                    # (rms_grad * rms) / (sqrsum + eps*K) * -0.5.
                    for token_idx in T.Parallel(token_block):
                        row_sum_view[token_idx] = (
                            row_sum_view[token_idx]
                            * (
                                T.float32(1.0)
                                / T.sqrt(sqrsum_ub[token_idx] * T.float32(1.0 / rms_group_size) + T.float32(rms_eps))
                            )
                            / (sqrsum_ub[token_idx] + T.float32(rms_eps * rms_group_size))
                            * T.float32(-0.5)
                        )

                T.copy(out_mul_grad_ub[:, :mhc_mult3], out_mul_grad[token_base : token_base + token_block, 0, :mhc_mult3])
                T.copy(sqrsum_grad_compact_ub[:token_block], sqrsum_grad[token_base : token_base + token_block, 0])
            else:
                for token_offset in T.serial(token_block):
                    pid = token_base + token_offset
                    if pid < num_tokens:
                        with T.SimtVF(threads=32):
                            sqrsum_frag = T.alloc_fragment(1, T.float32)
                            sqrsum_frag[0] = sqrsum[pid, 0]
                            # Match the vectorized main path's
                            # 1/sqrt(sq * (1/group) + eps) chain bitwise.
                            inv_rms_group_size = T.float32(1.0 / rms_group_size)
                            rms = T.float32(1.0) / T.sqrt(sqrsum_frag[0] * inv_rms_group_size + rms_eps)

                            rms_grad_reducer = T.alloc_reducer(1, T.float32, op='sum')
                            rms_grad_frag = T.alloc_fragment(1, T.float32)
                            T.reducer_init(rms_grad_reducer)
                            for j in T.Parallel(32):
                                if j < mhc_mult3:
                                    out_mul_grad[pid, 0, j] = out_grad[pid, j] * rms
                                    T.reducer_update(rms_grad_reducer[0], out_grad[pid, j] * out_mul[pid, 0, j])
                            T.finalize_reducer(rms_grad_reducer, rms_grad_frag)

                            for i in T.Parallel(1):
                                sqrsum_grad[pid, 0] = rms_grad_frag[i] * rms / (sqrsum_frag[i] + rms_eps * rms_group_size) / -2
