"""VMI port of the mhc norm_fn kernels (discussion #1616).

Each kernel is the ASC original with only the ``T.SimdVF()`` bodies rewritten
in logical VMI vectors; the outer schedule (T.Kernel / T.Persistent / T.copy /
UB layout / num_stages) and the host contract are kept verbatim from
``norm_fn_asc.py`` so ASC and VMI stay directly comparable.
"""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

_COMPACT_REDUCE_TOKEN_BLOCK = 16
_COMPACT_REDUCE_SPLIT_CHUNK = 32
_COMPACT_REDUCE_MHC_MULT3 = 24
_COMPACT_REDUCE_VREGS = 6


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_fwd_vmi(
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

            with T.SimdVF():
                # out_fn = fn * normw over full 64-lane fp32 vectors, so every
                # elementwise op runs unmasked.
                for vec_pair in range(hidden_block // 128):
                    col0 = vec_pair * 128
                    col1 = col0 + 64
                    normw0 = V.vload(normw_ub[col0], size=64)
                    normw1 = V.vload(normw_ub[col1], size=64)
                    for row in range(mhc_mult3):
                        fn0 = V.vload(fn_ub[row, col0], size=64)
                        fn1 = V.vload(fn_ub[row, col1], size=64)
                        V.vstore(V.vmul(fn0, normw0), out_fn_ub[row, col0])
                        V.vstore(V.vmul(fn1, normw1), out_fn_ub[row, col1])

            T.copy(out_fn_ub, out_fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_bwd_vmi(
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

            with T.SimdVF():
                # fn_grad += out_fn_grad * normw; normw_grad accumulates
                # out_fn_grad * fn across rows.  Full 64-lane fp32 vectors.
                for vec_pair in range(hidden_block // 128):
                    col0 = vec_pair * 128
                    col1 = col0 + 64
                    normw0 = V.vload(normw_ub[col0], size=64)
                    normw1 = V.vload(normw_ub[col1], size=64)
                    normw_grad0 = V.alloc_var(T.float32, size=64)
                    normw_grad1 = V.alloc_var(T.float32, size=64)
                    normw_grad0 = V.vload(normw_grad_ub[col0], size=64)
                    normw_grad1 = V.vload(normw_grad_ub[col1], size=64)

                    for row in range(mhc_mult3):
                        out_grad0 = V.vload(out_fn_grad_ub[row, col0], size=64)
                        out_grad1 = V.vload(out_fn_grad_ub[row, col1], size=64)
                        fn0 = V.vload(fn_ub[row, col0], size=64)
                        fn1 = V.vload(fn_ub[row, col1], size=64)
                        fn_grad0 = V.vload(fn_grad_ub[row, col0], size=64)
                        fn_grad1 = V.vload(fn_grad_ub[row, col1], size=64)

                        fn_grad_out0 = V.vadd(fn_grad0, V.vmul(out_grad0, normw0))
                        fn_grad_out1 = V.vadd(fn_grad1, V.vmul(out_grad1, normw1))
                        normw_grad0 = V.vadd(normw_grad0, V.vmul(out_grad0, fn0))
                        normw_grad1 = V.vadd(normw_grad1, V.vmul(out_grad1, fn1))

                        V.vstore(fn_grad_out0, fn_grad_ub[row, col0])
                        V.vstore(fn_grad_out1, fn_grad_ub[row, col1])

                    V.vstore(normw_grad0, normw_grad_ub[col0])
                    V.vstore(normw_grad1, normw_grad_ub[col1])

            T.copy(fn_grad_ub, fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])
            T.copy(normw_grad_ub, normw_grad[hidden_offset : hidden_offset + hidden_block])


def _compact_reduce_split_chunk(n_splits: int) -> int:
    for split_chunk in (_COMPACT_REDUCE_SPLIT_CHUNK, 16, 8, 4, 2):
        if n_splits % split_chunk == 0:
            return split_chunk
    return 1


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_fwd_vmi(
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

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            # Ragged tail blocks are handled by the same vectorized path:
            # T.copy clamps out-of-bound GM reads/writes, and the per-token
            # lane masks keep the padded tokens' stale UB lanes isolated,
            # so no separate scalar tail path is needed.
            with T.SimdVF():
                zero = V.vbrc(T.float32(0.0), size=64)
                for vec_idx in T.unroll(compact_vregs, explicit=True):
                    V.vstore(zero, acc_ub[vec_idx, 0])
                V.vstore(zero, sq_acc_ub[0])
                T.simd.mem_bar('VST_VLD')

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

                with T.SimdVF():
                    # Token 16 sqrsum lanes live in lanes 0..15 of the
                    # accumulator; only those lanes take part in the add.
                    mask_token = V.create_mask(16, size=64)
                    acc = V.alloc_local((compact_vregs,), V.vreg(64, T.float32))
                    acc_sq = V.alloc_var(T.float32, size=64)
                    for vec_idx in T.unroll(compact_vregs, explicit=True):
                        acc[vec_idx] = V.vload(acc_ub[vec_idx, 0], size=64)
                    acc_sq = V.vload(sq_acc_ub[0], size=64)

                    for local_split in range(split_chunk):
                        for vec_idx in T.unroll(compact_vregs, explicit=True):
                            acc[vec_idx] = V.vadd(
                                acc[vec_idx],
                                V.vload(mul_ub[local_split, (vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3], size=64),
                            )
                        acc_sq = V.vadd(acc_sq, V.vload(sq_ub[local_split, 0], size=64), mask_token)

                    for vec_idx in T.unroll(compact_vregs, explicit=True):
                        V.vstore(acc[vec_idx], acc_ub[vec_idx, 0])
                    V.vstore(acc_sq, sq_acc_ub[0])
                    T.simd.mem_bar('VST_VLD')

            with T.SimdVF():
                # One 64-lane vector spans 24/64 = 2 2/3 tokens, so each lane
                # picks up the rsqrt scale of the token it belongs to via a
                # lane-range mask instead of a physical pattern.
                mask_full = V.create_mask(64, size=64)
                inv_rms_group_size = V.vbrc(T.float32(1.0 / rms_group_size), size=64)
                one = V.vbrc(T.float32(1.0), size=64)
                zero = V.vbrc(T.float32(0.0), size=64)
                lane_idx = V.vci(T.int32(0), size=64)
                final_sq_acc = V.vload(sq_acc_ub[0], size=64)
                V.vstore(final_sq_acc, sq_out_ub[0])
                T.simd.mem_bar('VST_VLD')

                for vec_idx in T.unroll(compact_vregs, explicit=True):
                    acc_vec = V.vload(acc_ub[vec_idx, 0], size=64)
                    rms_vec = V.alloc_var(T.float32, size=64)
                    rms_vec = zero
                    elem_start = vec_idx * 64
                    elem_end = elem_start + 64
                    token_start = elem_start // _COMPACT_REDUCE_MHC_MULT3
                    token_end = (elem_end - 1) // _COMPACT_REDUCE_MHC_MULT3
                    for token_idx in range(token_start, token_end + 1):
                        sq_val = V.vbrc(V.vload(sq_out_ub[token_idx], size=1), size=64)
                        rms_val = V.vdiv(
                            one,
                            V.vsqrt(V.vadd(V.vmul(sq_val, inv_rms_group_size), V.vbrc(T.float32(rms_eps), size=64))),
                        )
                        lane_start = token_idx * _COMPACT_REDUCE_MHC_MULT3 - elem_start
                        lane_end = lane_start + _COMPACT_REDUCE_MHC_MULT3
                        token_mask = V.vand(
                            V.vcmps(lane_idx, T.int32(lane_start), mask_full, 'ge'),
                            V.vcmps(lane_idx, T.int32(lane_end), mask_full, 'lt'),
                        )
                        rms_vec = V.vsel(token_mask, rms_val, rms_vec)
                    V.vstore(acc_vec, mul_out_ub[(vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3])
                    V.vstore(
                        V.vmul(acc_vec, rms_vec), out_ub[(vec_idx * 64) // mhc_mult3, (vec_idx * 64) % mhc_mult3]
                    )

            T.copy(mul_out_ub, out_mul[token_base : token_base + token_block, 0, :])
            T.copy(out_ub, out[token_base : token_base + token_block, :])
            T.copy(sq_out_ub[:token_block], sqrsum[token_base : token_base + token_block, 0])


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_bwd_vmi(
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

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            if token_base + token_block <= num_tokens:
                T.copy(out_grad[token_base : token_base + token_block, :mhc_mult3], out_grad_ub[:, :mhc_mult3])
                T.copy(out_mul[token_base : token_base + token_block, 0, :mhc_mult3], out_mul_ub[:, :mhc_mult3])
                T.copy(sqrsum[token_base : token_base + token_block, 0], sqrsum_ub[:token_block])

                with T.SimdVF():
                    # Each 64-lane vector holds two tokens (mhc_mult3 <= 32
                    # lanes each, padded to the 32-lane row boundary).
                    mask_full = V.create_mask(64, size=64)
                    mask_lane0 = V.create_mask(1, size=64)
                    mask_first_token = V.create_mask(32, size=64)
                    lane_idx = V.vci(T.int32(0), size=64)
                    first_token_mask = V.vcmps(lane_idx, T.int32(mhc_mult3), mask_full, 'lt')
                    second_token_mask = V.vand(
                        V.vcmps(lane_idx, T.int32(32), mask_full, 'ge'),
                        V.vcmps(lane_idx, T.int32(32 + mhc_mult3), mask_full, 'lt'),
                    )
                    inv_rms_group_size = V.vbrc(T.float32(1.0 / rms_group_size), size=64)
                    one = V.vbrc(T.float32(1.0), size=64)
                    neg_half = V.vbrc(T.float32(-0.5), size=64)

                    for pair_idx in T.unroll(token_block // 2, explicit=True):
                        grad_pair = V.vload(out_grad_ub[pair_idx * 2, 0], size=64)
                        mul_pair = V.vload(out_mul_ub[pair_idx * 2, 0], size=64)
                        sq_even = V.vbrc(V.vload(sqrsum_ub[pair_idx * 2], size=1), size=64)
                        sq_odd = V.vbrc(V.vload(sqrsum_ub[pair_idx * 2 + 1], size=1), size=64)
                        rms_even = V.vdiv(
                            one,
                            V.vsqrt(V.vadd(V.vmul(sq_even, inv_rms_group_size), V.vbrc(T.float32(rms_eps), size=64))),
                        )
                        rms_odd = V.vdiv(
                            one,
                            V.vsqrt(V.vadd(V.vmul(sq_odd, inv_rms_group_size), V.vbrc(T.float32(rms_eps), size=64))),
                        )
                        rms_pair = V.vsel(mask_first_token, rms_even, rms_odd)
                        grad_mul = V.vmul(grad_pair, mul_pair)
                        out_mul_grad_pair = V.vmul(grad_pair, rms_pair)
                        V.vstore(out_mul_grad_pair, out_mul_grad_ub[pair_idx * 2, 0])

                        # ASC vcadd pairwise-reduces the active lanes and
                        # broadcasts the sum; VMI returns a 1-lane vreg, so
                        # broadcast it back (reassoc=True keeps the tree
                        # reduction the pairwise ASC form implies).
                        rms_grad_even = V.vbrc(V.vcadd(grad_mul, first_token_mask, reassoc=True), size=64)
                        rms_grad_odd = V.vbrc(V.vcadd(grad_mul, second_token_mask, reassoc=True), size=64)
                        denom_even = V.vadd(sq_even, V.vbrc(T.float32(rms_eps * rms_group_size), size=64))
                        denom_odd = V.vadd(sq_odd, V.vbrc(T.float32(rms_eps * rms_group_size), size=64))
                        sq_grad_even = V.vmul(V.vdiv(V.vmul(rms_grad_even, rms_even), denom_even), neg_half)
                        sq_grad_odd = V.vmul(V.vdiv(V.vmul(rms_grad_odd, rms_odd), denom_odd), neg_half)
                        V.vstore(sq_grad_even, sqrsum_grad_ub[pair_idx * 2, 0], mask=mask_lane0)
                        V.vstore(sq_grad_odd, sqrsum_grad_ub[pair_idx * 2 + 1, 0], mask=mask_lane0)

                    T.simd.mem_bar('VST_VLD')
                    sq_grad_compact = V.alloc_var(T.float32, size=64)
                    sq_grad_compact = V.vbrc(T.float32(0.0), size=64)
                    for token_offset in T.unroll(token_block, explicit=True):
                        token_sq_grad = V.vbrc(V.vload(sqrsum_grad_ub[token_offset, 0], size=1), size=64)
                        token_mask = V.vcmps(lane_idx, T.int32(token_offset), mask_full, 'eq')
                        sq_grad_compact = V.vsel(token_mask, token_sq_grad, sq_grad_compact)
                    V.vstore(sq_grad_compact, sqrsum_grad_compact_ub[0])

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
