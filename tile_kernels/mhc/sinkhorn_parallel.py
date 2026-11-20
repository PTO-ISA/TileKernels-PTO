import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def mhc_sinkhorn_fwd_parallel(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """T.Parallel forward: logical per-token 4x4 Sinkhorn normalization.

    The VMI port packed 4 tokens per 64-lane vector with segmented
    vcadd/vcmax row folds and vselr transposes; the scalar form keeps the
    matrices in UB, folds rows with direct reduces over a (tb*4, 4) row view,
    broadcasts each row value with four affine scatter stores (rep), and runs
    column passes as row passes on an explicitly materialized transpose.  The
    fold/rep/transpose scratch is new UB storage: the VMI working set lived
    entirely in vector registers, which the Parallel path cannot express yet.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 4
    tb = token_block_size
    mhc = 4
    mhc2 = 16

    with T.Kernel(num_vec_cores) as core_id:
        comb_in_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        comb_out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        fold_ub = T.alloc_shared((token_block_size * 4,), T.float32)
        rep_ub = T.alloc_shared((token_block_size, 16), T.float32)
        trans_ub = T.alloc_shared((token_block_size, 16), T.float32)
        T.annotate_buffer_versions(
            {comb_in_ub: num_stages, comb_out_ub: num_stages, fold_ub: 1, rep_ub: 1, trans_ub: 1}
        )
        mat_rows = T.Tensor((tb * mhc, mhc), T.float32, data=comb_out_ub.data, scope='shared.dyn')
        mat_flat = T.Tensor((tb * mhc2,), T.float32, data=comb_out_ub.data, scope='shared.dyn')
        in_flat = T.Tensor((tb * mhc2,), T.float32, data=comb_in_ub.data, scope='shared.dyn')
        trans_rows = T.Tensor((tb * mhc, mhc), T.float32, data=trans_ub.data, scope='shared.dyn')
        trans_flat = T.Tensor((tb * mhc2,), T.float32, data=trans_ub.data, scope='shared.dyn')
        rep_flat = T.Tensor((tb * mhc2,), T.float32, data=rep_ub.data, scope='shared.dyn')

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(comb_res_mix[token_offset, 0, 0], comb_in_ub)

            with T.SimdVF(lanes=64):
                # Step 1: row softmax + eps.
                for t, i in T.Parallel(tb, mhc2):
                    mat_flat[t * mhc2 + i] = in_flat[t * mhc2 + i]
                T.reduce_max(mat_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    mat_flat[t * mhc2 + i] = T.exp(mat_flat[t * mhc2 + i] - rep_ub[t, i])
                T.reduce_sum(mat_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    mat_flat[t * mhc2 + i] = mat_flat[t * mhc2 + i] / rep_ub[t, i] + T.float32(eps)
                # Step 2: first column normalize (row pass on the transpose).
                for t in T.Parallel(tb):
                    trans_flat[t * mhc2 + 0] = mat_flat[t * mhc2 + 0]
                    trans_flat[t * mhc2 + 1] = mat_flat[t * mhc2 + 4]
                    trans_flat[t * mhc2 + 2] = mat_flat[t * mhc2 + 8]
                    trans_flat[t * mhc2 + 3] = mat_flat[t * mhc2 + 12]
                    trans_flat[t * mhc2 + 4] = mat_flat[t * mhc2 + 1]
                    trans_flat[t * mhc2 + 5] = mat_flat[t * mhc2 + 5]
                    trans_flat[t * mhc2 + 6] = mat_flat[t * mhc2 + 9]
                    trans_flat[t * mhc2 + 7] = mat_flat[t * mhc2 + 13]
                    trans_flat[t * mhc2 + 8] = mat_flat[t * mhc2 + 2]
                    trans_flat[t * mhc2 + 9] = mat_flat[t * mhc2 + 6]
                    trans_flat[t * mhc2 + 10] = mat_flat[t * mhc2 + 10]
                    trans_flat[t * mhc2 + 11] = mat_flat[t * mhc2 + 14]
                    trans_flat[t * mhc2 + 12] = mat_flat[t * mhc2 + 3]
                    trans_flat[t * mhc2 + 13] = mat_flat[t * mhc2 + 7]
                    trans_flat[t * mhc2 + 14] = mat_flat[t * mhc2 + 11]
                    trans_flat[t * mhc2 + 15] = mat_flat[t * mhc2 + 15]
                T.reduce_sum(trans_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    trans_flat[t * mhc2 + i] = trans_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                for t in T.Parallel(tb):
                    mat_flat[t * mhc2 + 0] = trans_flat[t * mhc2 + 0]
                    mat_flat[t * mhc2 + 1] = trans_flat[t * mhc2 + 4]
                    mat_flat[t * mhc2 + 2] = trans_flat[t * mhc2 + 8]
                    mat_flat[t * mhc2 + 3] = trans_flat[t * mhc2 + 12]
                    mat_flat[t * mhc2 + 4] = trans_flat[t * mhc2 + 1]
                    mat_flat[t * mhc2 + 5] = trans_flat[t * mhc2 + 5]
                    mat_flat[t * mhc2 + 6] = trans_flat[t * mhc2 + 9]
                    mat_flat[t * mhc2 + 7] = trans_flat[t * mhc2 + 13]
                    mat_flat[t * mhc2 + 8] = trans_flat[t * mhc2 + 2]
                    mat_flat[t * mhc2 + 9] = trans_flat[t * mhc2 + 6]
                    mat_flat[t * mhc2 + 10] = trans_flat[t * mhc2 + 10]
                    mat_flat[t * mhc2 + 11] = trans_flat[t * mhc2 + 14]
                    mat_flat[t * mhc2 + 12] = trans_flat[t * mhc2 + 3]
                    mat_flat[t * mhc2 + 13] = trans_flat[t * mhc2 + 7]
                    mat_flat[t * mhc2 + 14] = trans_flat[t * mhc2 + 11]
                    mat_flat[t * mhc2 + 15] = trans_flat[t * mhc2 + 15]
                # Step 3: repeat-1 row+col normalize rounds.
                for _step in range(repeat - 1):
                    T.reduce_sum(mat_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t, i in T.Parallel(tb, mhc2):
                        mat_flat[t * mhc2 + i] = mat_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                    for t in T.Parallel(tb):
                        trans_flat[t * mhc2 + 0] = mat_flat[t * mhc2 + 0]
                        trans_flat[t * mhc2 + 1] = mat_flat[t * mhc2 + 4]
                        trans_flat[t * mhc2 + 2] = mat_flat[t * mhc2 + 8]
                        trans_flat[t * mhc2 + 3] = mat_flat[t * mhc2 + 12]
                        trans_flat[t * mhc2 + 4] = mat_flat[t * mhc2 + 1]
                        trans_flat[t * mhc2 + 5] = mat_flat[t * mhc2 + 5]
                        trans_flat[t * mhc2 + 6] = mat_flat[t * mhc2 + 9]
                        trans_flat[t * mhc2 + 7] = mat_flat[t * mhc2 + 13]
                        trans_flat[t * mhc2 + 8] = mat_flat[t * mhc2 + 2]
                        trans_flat[t * mhc2 + 9] = mat_flat[t * mhc2 + 6]
                        trans_flat[t * mhc2 + 10] = mat_flat[t * mhc2 + 10]
                        trans_flat[t * mhc2 + 11] = mat_flat[t * mhc2 + 14]
                        trans_flat[t * mhc2 + 12] = mat_flat[t * mhc2 + 3]
                        trans_flat[t * mhc2 + 13] = mat_flat[t * mhc2 + 7]
                        trans_flat[t * mhc2 + 14] = mat_flat[t * mhc2 + 11]
                        trans_flat[t * mhc2 + 15] = mat_flat[t * mhc2 + 15]
                    T.reduce_sum(trans_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t, i in T.Parallel(tb, mhc2):
                        trans_flat[t * mhc2 + i] = trans_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                    for t in T.Parallel(tb):
                        mat_flat[t * mhc2 + 0] = trans_flat[t * mhc2 + 0]
                        mat_flat[t * mhc2 + 1] = trans_flat[t * mhc2 + 4]
                        mat_flat[t * mhc2 + 2] = trans_flat[t * mhc2 + 8]
                        mat_flat[t * mhc2 + 3] = trans_flat[t * mhc2 + 12]
                        mat_flat[t * mhc2 + 4] = trans_flat[t * mhc2 + 1]
                        mat_flat[t * mhc2 + 5] = trans_flat[t * mhc2 + 5]
                        mat_flat[t * mhc2 + 6] = trans_flat[t * mhc2 + 9]
                        mat_flat[t * mhc2 + 7] = trans_flat[t * mhc2 + 13]
                        mat_flat[t * mhc2 + 8] = trans_flat[t * mhc2 + 2]
                        mat_flat[t * mhc2 + 9] = trans_flat[t * mhc2 + 6]
                        mat_flat[t * mhc2 + 10] = trans_flat[t * mhc2 + 10]
                        mat_flat[t * mhc2 + 11] = trans_flat[t * mhc2 + 14]
                        mat_flat[t * mhc2 + 12] = trans_flat[t * mhc2 + 3]
                        mat_flat[t * mhc2 + 13] = trans_flat[t * mhc2 + 7]
                        mat_flat[t * mhc2 + 14] = trans_flat[t * mhc2 + 11]
                        mat_flat[t * mhc2 + 15] = trans_flat[t * mhc2 + 15]

            T.copy(comb_out_ub, comb_res_mix_out[token_offset, 0, 0])


@tilelang.jit(target='pto')
def mhc_sinkhorn_bwd_parallel(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """T.Parallel backward: forward recompute (saving per-step states and
    broadcast sums to UB, same storage semantics as the VMI port) followed by
    the reverse sweep.  The gradient iterates in grad_ub in place; row folds
    are direct reduces, column folds run on the explicit transpose; the final
    result lands in out_ub.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 3
    tb = token_block_size
    mhc = 4
    mhc2 = 16

    with T.Kernel(num_vec_cores) as core_id:
        grad_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        x_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        xs_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        fold_ub = T.alloc_shared((token_block_size * 4,), T.float32)
        rep_ub = T.alloc_shared((token_block_size, 16), T.float32)
        trans_ub = T.alloc_shared((token_block_size, 16), T.float32)
        temp_ub = T.alloc_shared((token_block_size, 16), T.float32)
        T.annotate_buffer_versions(
            {
                grad_ub: num_stages,
                x_ub: num_stages,
                out_ub: num_stages,
                fold_ub: 1,
                rep_ub: 1,
                trans_ub: 1,
                temp_ub: 1,
            }
        )
        cur_rows = T.Tensor((tb * mhc, mhc), T.float32, data=out_ub.data, scope='shared.dyn')
        cur_flat = T.Tensor((tb * mhc2,), T.float32, data=out_ub.data, scope='shared.dyn')
        x_flat = T.Tensor((tb * mhc2,), T.float32, data=x_ub.data, scope='shared.dyn')
        g_flat = T.Tensor((tb * mhc2,), T.float32, data=grad_ub.data, scope='shared.dyn')
        xs_flat = T.Tensor((repeat * 2 * tb * mhc2,), T.float32, data=xs_ub.data, scope='shared.dyn')
        sums_flat = T.Tensor((repeat * 2 * tb * mhc2,), T.float32, data=sums_ub.data, scope='shared.dyn')
        trans_rows = T.Tensor((tb * mhc, mhc), T.float32, data=trans_ub.data, scope='shared.dyn')
        trans_flat = T.Tensor((tb * mhc2,), T.float32, data=trans_ub.data, scope='shared.dyn')
        rep_flat = T.Tensor((tb * mhc2,), T.float32, data=rep_ub.data, scope='shared.dyn')
        temp_rows = T.Tensor((tb * mhc, mhc), T.float32, data=temp_ub.data, scope='shared.dyn')
        temp_flat = T.Tensor((tb * mhc2,), T.float32, data=temp_ub.data, scope='shared.dyn')

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(grad_output[token_offset, 0, 0], grad_ub)
            T.copy(x[token_offset, 0, 0], x_ub)

            with T.SimdVF(lanes=64):
                # ================= Phase 1: forward recompute =================
                for t, i in T.Parallel(tb, mhc2):
                    cur_flat[t * mhc2 + i] = x_flat[t * mhc2 + i]
                T.reduce_max(cur_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    cur_flat[t * mhc2 + i] = T.exp(cur_flat[t * mhc2 + i] - rep_ub[t, i])
                T.reduce_sum(cur_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    xs_flat[0 * tb * mhc2 + t * mhc2 + i] = cur_flat[t * mhc2 + i] / rep_ub[t, i]
                for t, i in T.Parallel(tb, mhc2):
                    cur_flat[t * mhc2 + i] = cur_flat[t * mhc2 + i] / rep_ub[t, i] + T.float32(eps)
                for t, i in T.Parallel(tb, mhc2):
                    xs_flat[1 * tb * mhc2 + t * mhc2 + i] = cur_flat[t * mhc2 + i]
                # First column normalize; sums[1] = col sums, original layout.
                for t in T.Parallel(tb):
                    trans_flat[t * mhc2 + 0] = cur_flat[t * mhc2 + 0]
                    trans_flat[t * mhc2 + 1] = cur_flat[t * mhc2 + 4]
                    trans_flat[t * mhc2 + 2] = cur_flat[t * mhc2 + 8]
                    trans_flat[t * mhc2 + 3] = cur_flat[t * mhc2 + 12]
                    trans_flat[t * mhc2 + 4] = cur_flat[t * mhc2 + 1]
                    trans_flat[t * mhc2 + 5] = cur_flat[t * mhc2 + 5]
                    trans_flat[t * mhc2 + 6] = cur_flat[t * mhc2 + 9]
                    trans_flat[t * mhc2 + 7] = cur_flat[t * mhc2 + 13]
                    trans_flat[t * mhc2 + 8] = cur_flat[t * mhc2 + 2]
                    trans_flat[t * mhc2 + 9] = cur_flat[t * mhc2 + 6]
                    trans_flat[t * mhc2 + 10] = cur_flat[t * mhc2 + 10]
                    trans_flat[t * mhc2 + 11] = cur_flat[t * mhc2 + 14]
                    trans_flat[t * mhc2 + 12] = cur_flat[t * mhc2 + 3]
                    trans_flat[t * mhc2 + 13] = cur_flat[t * mhc2 + 7]
                    trans_flat[t * mhc2 + 14] = cur_flat[t * mhc2 + 11]
                    trans_flat[t * mhc2 + 15] = cur_flat[t * mhc2 + 15]
                T.reduce_sum(trans_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t in T.Parallel(tb):
                    temp_flat[t * mhc2 + 0] = rep_flat[t * mhc2 + 0]
                    temp_flat[t * mhc2 + 1] = rep_flat[t * mhc2 + 4]
                    temp_flat[t * mhc2 + 2] = rep_flat[t * mhc2 + 8]
                    temp_flat[t * mhc2 + 3] = rep_flat[t * mhc2 + 12]
                    temp_flat[t * mhc2 + 4] = rep_flat[t * mhc2 + 1]
                    temp_flat[t * mhc2 + 5] = rep_flat[t * mhc2 + 5]
                    temp_flat[t * mhc2 + 6] = rep_flat[t * mhc2 + 9]
                    temp_flat[t * mhc2 + 7] = rep_flat[t * mhc2 + 13]
                    temp_flat[t * mhc2 + 8] = rep_flat[t * mhc2 + 2]
                    temp_flat[t * mhc2 + 9] = rep_flat[t * mhc2 + 6]
                    temp_flat[t * mhc2 + 10] = rep_flat[t * mhc2 + 10]
                    temp_flat[t * mhc2 + 11] = rep_flat[t * mhc2 + 14]
                    temp_flat[t * mhc2 + 12] = rep_flat[t * mhc2 + 3]
                    temp_flat[t * mhc2 + 13] = rep_flat[t * mhc2 + 7]
                    temp_flat[t * mhc2 + 14] = rep_flat[t * mhc2 + 11]
                    temp_flat[t * mhc2 + 15] = rep_flat[t * mhc2 + 15]
                for t, i in T.Parallel(tb, mhc2):
                    sums_flat[1 * tb * mhc2 + t * mhc2 + i] = temp_flat[t * mhc2 + i]
                for t, i in T.Parallel(tb, mhc2):
                    trans_flat[t * mhc2 + i] = trans_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                for t in T.Parallel(tb):
                    cur_flat[t * mhc2 + 0] = trans_flat[t * mhc2 + 0]
                    cur_flat[t * mhc2 + 1] = trans_flat[t * mhc2 + 4]
                    cur_flat[t * mhc2 + 2] = trans_flat[t * mhc2 + 8]
                    cur_flat[t * mhc2 + 3] = trans_flat[t * mhc2 + 12]
                    cur_flat[t * mhc2 + 4] = trans_flat[t * mhc2 + 1]
                    cur_flat[t * mhc2 + 5] = trans_flat[t * mhc2 + 5]
                    cur_flat[t * mhc2 + 6] = trans_flat[t * mhc2 + 9]
                    cur_flat[t * mhc2 + 7] = trans_flat[t * mhc2 + 13]
                    cur_flat[t * mhc2 + 8] = trans_flat[t * mhc2 + 2]
                    cur_flat[t * mhc2 + 9] = trans_flat[t * mhc2 + 6]
                    cur_flat[t * mhc2 + 10] = trans_flat[t * mhc2 + 10]
                    cur_flat[t * mhc2 + 11] = trans_flat[t * mhc2 + 14]
                    cur_flat[t * mhc2 + 12] = trans_flat[t * mhc2 + 3]
                    cur_flat[t * mhc2 + 13] = trans_flat[t * mhc2 + 7]
                    cur_flat[t * mhc2 + 14] = trans_flat[t * mhc2 + 11]
                    cur_flat[t * mhc2 + 15] = trans_flat[t * mhc2 + 15]
                # repeat-1 rounds; save pre-divide states and broadcast sums.
                for step in range(repeat - 1):
                    s_row = (step + 1) * 2
                    s_col = (step + 1) * 2 + 1
                    T.reduce_sum(cur_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t, i in T.Parallel(tb, mhc2):
                        sums_flat[s_row * tb * mhc2 + t * mhc2 + i] = rep_ub[t, i]
                    for t, i in T.Parallel(tb, mhc2):
                        xs_flat[s_row * tb * mhc2 + t * mhc2 + i] = cur_flat[t * mhc2 + i]
                    for t, i in T.Parallel(tb, mhc2):
                        cur_flat[t * mhc2 + i] = cur_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                    for t in T.Parallel(tb):
                        trans_flat[t * mhc2 + 0] = cur_flat[t * mhc2 + 0]
                        trans_flat[t * mhc2 + 1] = cur_flat[t * mhc2 + 4]
                        trans_flat[t * mhc2 + 2] = cur_flat[t * mhc2 + 8]
                        trans_flat[t * mhc2 + 3] = cur_flat[t * mhc2 + 12]
                        trans_flat[t * mhc2 + 4] = cur_flat[t * mhc2 + 1]
                        trans_flat[t * mhc2 + 5] = cur_flat[t * mhc2 + 5]
                        trans_flat[t * mhc2 + 6] = cur_flat[t * mhc2 + 9]
                        trans_flat[t * mhc2 + 7] = cur_flat[t * mhc2 + 13]
                        trans_flat[t * mhc2 + 8] = cur_flat[t * mhc2 + 2]
                        trans_flat[t * mhc2 + 9] = cur_flat[t * mhc2 + 6]
                        trans_flat[t * mhc2 + 10] = cur_flat[t * mhc2 + 10]
                        trans_flat[t * mhc2 + 11] = cur_flat[t * mhc2 + 14]
                        trans_flat[t * mhc2 + 12] = cur_flat[t * mhc2 + 3]
                        trans_flat[t * mhc2 + 13] = cur_flat[t * mhc2 + 7]
                        trans_flat[t * mhc2 + 14] = cur_flat[t * mhc2 + 11]
                        trans_flat[t * mhc2 + 15] = cur_flat[t * mhc2 + 15]
                    T.reduce_sum(trans_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t in T.Parallel(tb):
                        temp_flat[t * mhc2 + 0] = rep_flat[t * mhc2 + 0]
                        temp_flat[t * mhc2 + 1] = rep_flat[t * mhc2 + 4]
                        temp_flat[t * mhc2 + 2] = rep_flat[t * mhc2 + 8]
                        temp_flat[t * mhc2 + 3] = rep_flat[t * mhc2 + 12]
                        temp_flat[t * mhc2 + 4] = rep_flat[t * mhc2 + 1]
                        temp_flat[t * mhc2 + 5] = rep_flat[t * mhc2 + 5]
                        temp_flat[t * mhc2 + 6] = rep_flat[t * mhc2 + 9]
                        temp_flat[t * mhc2 + 7] = rep_flat[t * mhc2 + 13]
                        temp_flat[t * mhc2 + 8] = rep_flat[t * mhc2 + 2]
                        temp_flat[t * mhc2 + 9] = rep_flat[t * mhc2 + 6]
                        temp_flat[t * mhc2 + 10] = rep_flat[t * mhc2 + 10]
                        temp_flat[t * mhc2 + 11] = rep_flat[t * mhc2 + 14]
                        temp_flat[t * mhc2 + 12] = rep_flat[t * mhc2 + 3]
                        temp_flat[t * mhc2 + 13] = rep_flat[t * mhc2 + 7]
                        temp_flat[t * mhc2 + 14] = rep_flat[t * mhc2 + 11]
                        temp_flat[t * mhc2 + 15] = rep_flat[t * mhc2 + 15]
                    for t, i in T.Parallel(tb, mhc2):
                        sums_flat[s_col * tb * mhc2 + t * mhc2 + i] = temp_flat[t * mhc2 + i]
                    for t, i in T.Parallel(tb, mhc2):
                        xs_flat[s_col * tb * mhc2 + t * mhc2 + i] = cur_flat[t * mhc2 + i]
                    for t, i in T.Parallel(tb, mhc2):
                        trans_flat[t * mhc2 + i] = trans_flat[t * mhc2 + i] / (rep_ub[t, i] + T.float32(eps))
                    for t in T.Parallel(tb):
                        cur_flat[t * mhc2 + 0] = trans_flat[t * mhc2 + 0]
                        cur_flat[t * mhc2 + 1] = trans_flat[t * mhc2 + 4]
                        cur_flat[t * mhc2 + 2] = trans_flat[t * mhc2 + 8]
                        cur_flat[t * mhc2 + 3] = trans_flat[t * mhc2 + 12]
                        cur_flat[t * mhc2 + 4] = trans_flat[t * mhc2 + 1]
                        cur_flat[t * mhc2 + 5] = trans_flat[t * mhc2 + 5]
                        cur_flat[t * mhc2 + 6] = trans_flat[t * mhc2 + 9]
                        cur_flat[t * mhc2 + 7] = trans_flat[t * mhc2 + 13]
                        cur_flat[t * mhc2 + 8] = trans_flat[t * mhc2 + 2]
                        cur_flat[t * mhc2 + 9] = trans_flat[t * mhc2 + 6]
                        cur_flat[t * mhc2 + 10] = trans_flat[t * mhc2 + 10]
                        cur_flat[t * mhc2 + 11] = trans_flat[t * mhc2 + 14]
                        cur_flat[t * mhc2 + 12] = trans_flat[t * mhc2 + 3]
                        cur_flat[t * mhc2 + 13] = trans_flat[t * mhc2 + 7]
                        cur_flat[t * mhc2 + 14] = trans_flat[t * mhc2 + 11]
                        cur_flat[t * mhc2 + 15] = trans_flat[t * mhc2 + 15]

                # ================= Phase 2: reverse sweep =================
                # The 2*repeat-1 reverse steps alternate col/row backward
                # starting from the last col step.  The pairs are emitted
                # statically (a runtime parity branch would wrap the direct
                # reduces in a TIR If, which Verify rejects).
                for pair in range(repeat - 1):
                    s_idx = 2 * repeat - 1 - 2 * pair
                    # Column-normalize backward.
                    for t, i in T.Parallel(tb, mhc2):
                        temp_flat[t * mhc2 + i] = g_flat[t * mhc2 + i] * xs_flat[s_idx * tb * mhc2 + t * mhc2 + i]
                    for t in T.Parallel(tb):
                        trans_flat[t * mhc2 + 0] = temp_flat[t * mhc2 + 0]
                        trans_flat[t * mhc2 + 1] = temp_flat[t * mhc2 + 4]
                        trans_flat[t * mhc2 + 2] = temp_flat[t * mhc2 + 8]
                        trans_flat[t * mhc2 + 3] = temp_flat[t * mhc2 + 12]
                        trans_flat[t * mhc2 + 4] = temp_flat[t * mhc2 + 1]
                        trans_flat[t * mhc2 + 5] = temp_flat[t * mhc2 + 5]
                        trans_flat[t * mhc2 + 6] = temp_flat[t * mhc2 + 9]
                        trans_flat[t * mhc2 + 7] = temp_flat[t * mhc2 + 13]
                        trans_flat[t * mhc2 + 8] = temp_flat[t * mhc2 + 2]
                        trans_flat[t * mhc2 + 9] = temp_flat[t * mhc2 + 6]
                        trans_flat[t * mhc2 + 10] = temp_flat[t * mhc2 + 10]
                        trans_flat[t * mhc2 + 11] = temp_flat[t * mhc2 + 14]
                        trans_flat[t * mhc2 + 12] = temp_flat[t * mhc2 + 3]
                        trans_flat[t * mhc2 + 13] = temp_flat[t * mhc2 + 7]
                        trans_flat[t * mhc2 + 14] = temp_flat[t * mhc2 + 11]
                        trans_flat[t * mhc2 + 15] = temp_flat[t * mhc2 + 15]
                    T.reduce_sum(trans_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t in T.Parallel(tb):
                        temp_flat[t * mhc2 + 0] = rep_flat[t * mhc2 + 0]
                        temp_flat[t * mhc2 + 1] = rep_flat[t * mhc2 + 4]
                        temp_flat[t * mhc2 + 2] = rep_flat[t * mhc2 + 8]
                        temp_flat[t * mhc2 + 3] = rep_flat[t * mhc2 + 12]
                        temp_flat[t * mhc2 + 4] = rep_flat[t * mhc2 + 1]
                        temp_flat[t * mhc2 + 5] = rep_flat[t * mhc2 + 5]
                        temp_flat[t * mhc2 + 6] = rep_flat[t * mhc2 + 9]
                        temp_flat[t * mhc2 + 7] = rep_flat[t * mhc2 + 13]
                        temp_flat[t * mhc2 + 8] = rep_flat[t * mhc2 + 2]
                        temp_flat[t * mhc2 + 9] = rep_flat[t * mhc2 + 6]
                        temp_flat[t * mhc2 + 10] = rep_flat[t * mhc2 + 10]
                        temp_flat[t * mhc2 + 11] = rep_flat[t * mhc2 + 14]
                        temp_flat[t * mhc2 + 12] = rep_flat[t * mhc2 + 3]
                        temp_flat[t * mhc2 + 13] = rep_flat[t * mhc2 + 7]
                        temp_flat[t * mhc2 + 14] = rep_flat[t * mhc2 + 11]
                        temp_flat[t * mhc2 + 15] = rep_flat[t * mhc2 + 15]
                    for t, i in T.Parallel(tb, mhc2):
                        g_flat[t * mhc2 + i] = (g_flat[t * mhc2 + i] - temp_flat[t * mhc2 + i] / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))) / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))
                    s_idx = 2 * repeat - 2 - 2 * pair
                    # Row-normalize backward.
                    for t, i in T.Parallel(tb, mhc2):
                        temp_flat[t * mhc2 + i] = g_flat[t * mhc2 + i] * xs_flat[s_idx * tb * mhc2 + t * mhc2 + i]
                    T.reduce_sum(temp_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t, i in T.Parallel(tb, mhc2):
                        g_flat[t * mhc2 + i] = (g_flat[t * mhc2 + i] - rep_ub[t, i] / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))) / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))
                for _last in range(1):
                    s_idx = 1
                    # Final column-normalize backward.
                    for t, i in T.Parallel(tb, mhc2):
                        temp_flat[t * mhc2 + i] = g_flat[t * mhc2 + i] * xs_flat[s_idx * tb * mhc2 + t * mhc2 + i]
                    for t in T.Parallel(tb):
                        trans_flat[t * mhc2 + 0] = temp_flat[t * mhc2 + 0]
                        trans_flat[t * mhc2 + 1] = temp_flat[t * mhc2 + 4]
                        trans_flat[t * mhc2 + 2] = temp_flat[t * mhc2 + 8]
                        trans_flat[t * mhc2 + 3] = temp_flat[t * mhc2 + 12]
                        trans_flat[t * mhc2 + 4] = temp_flat[t * mhc2 + 1]
                        trans_flat[t * mhc2 + 5] = temp_flat[t * mhc2 + 5]
                        trans_flat[t * mhc2 + 6] = temp_flat[t * mhc2 + 9]
                        trans_flat[t * mhc2 + 7] = temp_flat[t * mhc2 + 13]
                        trans_flat[t * mhc2 + 8] = temp_flat[t * mhc2 + 2]
                        trans_flat[t * mhc2 + 9] = temp_flat[t * mhc2 + 6]
                        trans_flat[t * mhc2 + 10] = temp_flat[t * mhc2 + 10]
                        trans_flat[t * mhc2 + 11] = temp_flat[t * mhc2 + 14]
                        trans_flat[t * mhc2 + 12] = temp_flat[t * mhc2 + 3]
                        trans_flat[t * mhc2 + 13] = temp_flat[t * mhc2 + 7]
                        trans_flat[t * mhc2 + 14] = temp_flat[t * mhc2 + 11]
                        trans_flat[t * mhc2 + 15] = temp_flat[t * mhc2 + 15]
                    T.reduce_sum(trans_rows, fold_ub, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold_ub[k]
                        rep_flat[k * mhc + 1] = fold_ub[k]
                        rep_flat[k * mhc + 2] = fold_ub[k]
                        rep_flat[k * mhc + 3] = fold_ub[k]
                    for t in T.Parallel(tb):
                        temp_flat[t * mhc2 + 0] = rep_flat[t * mhc2 + 0]
                        temp_flat[t * mhc2 + 1] = rep_flat[t * mhc2 + 4]
                        temp_flat[t * mhc2 + 2] = rep_flat[t * mhc2 + 8]
                        temp_flat[t * mhc2 + 3] = rep_flat[t * mhc2 + 12]
                        temp_flat[t * mhc2 + 4] = rep_flat[t * mhc2 + 1]
                        temp_flat[t * mhc2 + 5] = rep_flat[t * mhc2 + 5]
                        temp_flat[t * mhc2 + 6] = rep_flat[t * mhc2 + 9]
                        temp_flat[t * mhc2 + 7] = rep_flat[t * mhc2 + 13]
                        temp_flat[t * mhc2 + 8] = rep_flat[t * mhc2 + 2]
                        temp_flat[t * mhc2 + 9] = rep_flat[t * mhc2 + 6]
                        temp_flat[t * mhc2 + 10] = rep_flat[t * mhc2 + 10]
                        temp_flat[t * mhc2 + 11] = rep_flat[t * mhc2 + 14]
                        temp_flat[t * mhc2 + 12] = rep_flat[t * mhc2 + 3]
                        temp_flat[t * mhc2 + 13] = rep_flat[t * mhc2 + 7]
                        temp_flat[t * mhc2 + 14] = rep_flat[t * mhc2 + 11]
                        temp_flat[t * mhc2 + 15] = rep_flat[t * mhc2 + 15]
                    for t, i in T.Parallel(tb, mhc2):
                        g_flat[t * mhc2 + i] = (g_flat[t * mhc2 + i] - temp_flat[t * mhc2 + i] / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))) / (sums_flat[s_idx * tb * mhc2 + t * mhc2 + i] + T.float32(eps))

                # Final step: g = (g - rowsum(g * xs[0])) * xs[0].
                for t, i in T.Parallel(tb, mhc2):
                    temp_flat[t * mhc2 + i] = g_flat[t * mhc2 + i] * xs_flat[0 * tb * mhc2 + t * mhc2 + i]
                T.reduce_sum(temp_rows, fold_ub, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold_ub[k]
                    rep_flat[k * mhc + 1] = fold_ub[k]
                    rep_flat[k * mhc + 2] = fold_ub[k]
                    rep_flat[k * mhc + 3] = fold_ub[k]
                for t, i in T.Parallel(tb, mhc2):
                    cur_flat[t * mhc2 + i] = (g_flat[t * mhc2 + i] - rep_ub[t, i]) * xs_flat[0 * tb * mhc2 + t * mhc2 + i]

            T.copy(out_ub, grad_input[token_offset, 0, 0])
