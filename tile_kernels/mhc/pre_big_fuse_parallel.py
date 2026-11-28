import tilelang
from tilelang.ascend import language as T


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit(target='pto')
def mhc_pre_big_fuse_fwd_parallel(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    h_blk: int,
    n_splits: int,
    token_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    """T.Parallel fused forward: reduce partials + RMSNorm, split mixes,
    Sinkhorn, pre-apply.

    The VMI port packed four 4x4 token matrices per 64-lane vector and used
    vgather/shift index math, segmented vcadd/vcmax with 4-of-8 masks and
    vselr transposes.  The scalar T.Parallel form writes the logical per-token
    math instead: the Sinkhorn rows live in a (tb*4, 4) view of the matrix
    storage, the row/column folds are 4-term scalar sums staged in disjoint
    columns of comb_in_ub, and the column normalization indexes columns
    directly (no transpose).  Outer scheduling is verbatim from the VMI port.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden = T.const('hidden')
    mhc = 4
    mhc2 = mhc * mhc
    mhc3 = mhc * 2 + mhc2
    assert mhc3 == 24
    assert n_splits == 1, 'Ascend pre_big_fuse expects the GEMM to reduce split-K internally (n_splits=1)'
    assert h_blk % _VEC == 0 and hidden % h_blk == 0

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc3,), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_mix: T.Tensor[(num_tokens, mhc), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc2), T.float32]
    layer_input: T.Tensor[(num_tokens, hidden), T.bfloat16]

    tb = token_block
    num_hidden_tiles = hidden // h_blk
    # Same compile-time fp32 constant as the reference reduce kernel.
    inv_rms_group_size = T.float32(1.0 / (mhc * hidden))

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc3, 8), T.float32)
        mul_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        mixes_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        sq_ub = T.alloc_shared((tb,), T.float32)
        pre_ub = T.alloc_shared((tb, mhc), T.float32)
        post_ub = T.alloc_shared((tb, mhc), T.float32)
        comb_in_ub = T.alloc_shared((tb, mhc2), T.float32)
        comb_out_ub = T.alloc_shared((tb, mhc2), T.float32)
        x_ub = T.alloc_shared((tb, mhc, h_blk), T.bfloat16)
        layer_input_ub = T.alloc_shared((tb, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                mul_flat_ub: num_stages,
                mixes_flat_ub: num_stages,
                sq_ub: num_stages,
                pre_ub: num_stages,
                post_ub: num_stages,
                comb_in_ub: num_stages,
                comb_out_ub: num_stages,
                x_ub: num_stages,
                layer_input_ub: num_stages,
            }
        )
        # Zero-cost views over existing storage:
        #  * mat_rows / mat_flat: the Sinkhorn matrices in comb_out_ub seen as
        #    (tb*mhc, mhc) rows (reduce source) and flat words;
        #  * rms_stage: per-token rsqrt scale staged in the pre_ub words
        #    (overwritten by the real pre values later in the same region);
        #  * fold / rep / scratch_flat: Sinkhorn row-fold results, their
        #    broadcast expansion and the transpose scratch, all packed into
        #    mixes_flat_ub (free after Phase B consumed it) and comb_in_ub
        #    (free after the matrices moved to comb_out_ub).
        mat_rows = T.Tensor((tb * mhc, mhc), T.float32, data=comb_out_ub.data, scope='shared.dyn')
        mat_flat = T.Tensor((tb * mhc2,), T.float32, data=comb_out_ub.data, scope='shared.dyn')
        rms_stage = T.Tensor((tb,), T.float32, data=pre_ub.data, scope='shared.dyn')
        base_compact = T.Tensor((mhc3,), T.float32, data=mul_flat_ub.data, scope='shared.dyn')
        pre_flat = T.Tensor((tb * mhc,), T.float32, data=pre_ub.data, scope='shared.dyn')
        post_flat = T.Tensor((tb * mhc,), T.float32, data=post_ub.data, scope='shared.dyn')
        fold = T.Tensor((tb * mhc,), T.float32, data=mul_flat_ub.data, scope='shared.dyn')
        rep = T.Tensor((tb, mhc2), T.float32, data=mixes_flat_ub.data, scope='shared.dyn')
        rep_flat = T.Tensor((tb * mhc2,), T.float32, data=mixes_flat_ub.data, scope='shared.dyn')
        trans_rows = T.Tensor((tb * mhc, mhc), T.float32, data=comb_in_ub.data, scope='shared.dyn')
        trans_flat = T.Tensor((tb * mhc2,), T.float32, data=comb_in_ub.data, scope='shared.dyn')

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc3):
                base_ub[i, 0] = mhc_base[i]

        for pid in T.Persistent(
            [T.ceildiv(num_tokens, tb)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            token_base = pid * tb
            T.copy(gemm_out_mul[0:1, token_base : token_base + tb, 0:mhc3], T.view(mul_flat_ub, (tb, mhc3)))
            T.copy(gemm_out_sqrsum[0:1, token_base : token_base + tb], sq_ub)

            ##################################################################
            # _mhc_reduce_partials_and_rmsnorm_fwd + _mhc_pre_split_mixes_fwd
            # + _mhc_sinkhorn_fwd in a single SimdVF phase (schedule verbatim
            # from the VMI fused kernel).
            with T.SimdVF(lanes=64):
                ##############################################################
                # Phase A: mixes = gemm_out_mul * rms.  Per-token rsqrt scale
                # computed lane-varying over the token index, then applied as
                # a row broadcast (the VMI lane-range mask smearing is gone).
                for t in T.Parallel(tb):
                    rms_stage[t] = T.float32(1.0) / T.sqrt(
                        sq_ub[t] * inv_rms_group_size + T.float32(rms_eps)
                    )
                for t, j in T.Parallel(tb, mhc3):
                    mixes_flat_ub[t * mhc3 + j] = mul_flat_ub[t * mhc3 + j] * rms_stage[t]

                ##############################################################
                # Phase B: split mixes into pre/post/comb (sigmoid folds; the
                # VMI vgather + shift index math becomes plain indexing).
                # base_ub rows are 8-padded, so compact the scalars first with
                # one unconditional affine gather into the (free) mul_flat
                # words; the split unit is a full 24-column row so every lane
                # coefficient stays 1 (a narrower unit would re-map lanes and
                # push these accesses onto the indexed path).
                for j in T.Parallel(mhc3):
                    base_compact[j] = base_ub[j, 0]
                for t, j in T.Parallel(tb, mhc3):
                    if j < mhc:
                        mixes_flat_ub[t * mhc3 + j] = (
                            T.float32(1.0)
                            / (
                                T.exp(
                                    T.float32(0.0)
                                    - (mixes_flat_ub[t * mhc3 + j] * scale_ub[0, 0] + base_compact[j])
                                )
                                + T.float32(1.0)
                            )
                            + T.float32(mhc_pre_eps)
                        )
                    else:
                        if j < 2 * mhc:
                            mixes_flat_ub[t * mhc3 + j] = (
                                T.float32(1.0)
                                / (
                                    T.exp(
                                        T.float32(0.0)
                                        - (mixes_flat_ub[t * mhc3 + j] * scale_ub[1, 0] + base_compact[j])
                                    )
                                    + T.float32(1.0)
                                )
                                * T.float32(mhc_post_mult_value)
                            )
                        else:
                            comb_out_ub[t, j - 2 * mhc] = (
                                mixes_flat_ub[t * mhc3 + j] * scale_ub[2, 0] + base_compact[j]
                            )
                # Scatter the transformed pre/post columns into their narrow
                # 16B-pitch rows (a conditional masked store into those rows
                # cannot prove the 32-byte alignment the backend requires, so
                # the sigmoid folds above go in place into the 96B-pitch
                # mixes rows and move here with per-lane affine stores).
                for t in T.Parallel(tb):
                    pre_flat[t * mhc + 0] = mixes_flat_ub[t * mhc3 + 0]
                    pre_flat[t * mhc + 1] = mixes_flat_ub[t * mhc3 + 1]
                    pre_flat[t * mhc + 2] = mixes_flat_ub[t * mhc3 + 2]
                    pre_flat[t * mhc + 3] = mixes_flat_ub[t * mhc3 + 3]
                    post_flat[t * mhc + 0] = mixes_flat_ub[t * mhc3 + mhc + 0]
                    post_flat[t * mhc + 1] = mixes_flat_ub[t * mhc3 + mhc + 1]
                    post_flat[t * mhc + 2] = mixes_flat_ub[t * mhc3 + mhc + 2]
                    post_flat[t * mhc + 3] = mixes_flat_ub[t * mhc3 + mhc + 3]

                ##############################################################
                # _mhc_sinkhorn_fwd, logical per-token 4x4 form.  M lives in
                # comb_out_ub; row folds are direct reduces over the
                # (tb*mhc, mhc) row view into the 64-word fold scratch; the
                # per-row value is expanded to all four columns with four
                # affine scatter stores (rep); column passes run as row passes
                # on an explicitly materialized transpose in comb_in_ub (16
                # affine gather+scatter stores per direction).  Every 2D unit
                # is a full 16-column row, so all lane coefficients stay 1.
                T.reduce_max(mat_rows, fold, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold[k]
                    rep_flat[k * mhc + 1] = fold[k]
                    rep_flat[k * mhc + 2] = fold[k]
                    rep_flat[k * mhc + 3] = fold[k]
                for t, i in T.Parallel(tb, mhc2):
                    comb_out_ub[t, i] = T.exp(comb_out_ub[t, i] - rep[t, i])
                T.reduce_sum(mat_rows, fold, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold[k]
                    rep_flat[k * mhc + 1] = fold[k]
                    rep_flat[k * mhc + 2] = fold[k]
                    rep_flat[k * mhc + 3] = fold[k]
                for t, i in T.Parallel(tb, mhc2):
                    comb_out_ub[t, i] = comb_out_ub[t, i] / rep[t, i] + T.float32(mhc_sinkhorn_eps)
                # First column normalization (row pass on the transpose).
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
                T.reduce_sum(trans_rows, fold, dim=1)
                for k in T.Parallel(tb * mhc):
                    rep_flat[k * mhc + 0] = fold[k]
                    rep_flat[k * mhc + 1] = fold[k]
                    rep_flat[k * mhc + 2] = fold[k]
                    rep_flat[k * mhc + 3] = fold[k]
                for t, i in T.Parallel(tb, mhc2):
                    comb_in_ub[t, i] = comb_in_ub[t, i] / (rep[t, i] + T.float32(mhc_sinkhorn_eps))
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
                for _step in range(sinkhorn_repeat - 1):
                    T.reduce_sum(mat_rows, fold, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold[k]
                        rep_flat[k * mhc + 1] = fold[k]
                        rep_flat[k * mhc + 2] = fold[k]
                        rep_flat[k * mhc + 3] = fold[k]
                    for t, i in T.Parallel(tb, mhc2):
                        comb_out_ub[t, i] = comb_out_ub[t, i] / (rep[t, i] + T.float32(mhc_sinkhorn_eps))
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
                    T.reduce_sum(trans_rows, fold, dim=1)
                    for k in T.Parallel(tb * mhc):
                        rep_flat[k * mhc + 0] = fold[k]
                        rep_flat[k * mhc + 1] = fold[k]
                        rep_flat[k * mhc + 2] = fold[k]
                        rep_flat[k * mhc + 3] = fold[k]
                    for t, i in T.Parallel(tb, mhc2):
                        comb_in_ub[t, i] = comb_in_ub[t, i] / (rep[t, i] + T.float32(mhc_sinkhorn_eps))
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

            T.copy(post_ub[:, 0:mhc], post_mix[token_base : token_base + tb, 0:mhc])
            T.copy(comb_out_ub, comb_mix[token_base : token_base + tb, 0:mhc2])

            ##################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(num_hidden_tiles, num_stages=2):
                T.copy(
                    residual[
                        token_base : token_base + tb,
                        0:mhc,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                    x_ub,
                )
                with T.SimdVF(lanes=64):
                    for t, h in T.Parallel(tb, h_blk):
                        layer_input_ub[t, h] = T.Cast(
                            'bfloat16',
                            sum(
                                (
                                    pre_ub[t, m] * T.Cast('float32', x_ub[t, m, h])
                                    for m in range(1, mhc)
                                ),
                                pre_ub[t, 0] * T.Cast('float32', x_ub[t, 0, h]),
                            ),
                        )

                T.copy(
                    layer_input_ub,
                    layer_input[
                        token_base : token_base + tb,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                )
