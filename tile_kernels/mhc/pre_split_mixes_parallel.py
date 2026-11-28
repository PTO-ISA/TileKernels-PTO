import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_fwd_parallel(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    num_stages: int,
    dtype: T.dtype,
    num_vec_cores: int,
):
    """Generic fallback: verbatim SimtVF implementation from the VMI port
    (not a SimdVF region, outside the #1747 rewrite scope)."""
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype]

    with T.Kernel(num_vec_cores) as core_id:
        for pid in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, core_id, num_stages=num_stages):
            input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), dtype)
            pre_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), dtype)
            post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), dtype)
            comb_res_mix_ub = T.alloc_shared((token_block_size, mhc_mult2), dtype)
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_ub)

            with T.SimtVF(threads=256):
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        pre_layer_mix_ub[i1, j] = T.sigmoid(input_mixes_ub[i1, j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        post_layer_mix_ub[i1, j] = (
                            T.sigmoid(input_mixes_ub[i1, j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j] = input_mixes_ub[i1, j + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2]

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult]
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult * 2] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult * 2]
                        )

                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        comb_res_mix_ub[i1, j + mhc_mult * 3] = (
                            input_mixes_ub[i1, j + mhc_mult * 2 + mhc_mult * 3] * mhc_scale[2] + mhc_base[j + mhc_mult * 2 + mhc_mult * 3]
                        )

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_fwd_simd_mhc4_parallel(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    num_stages: int,
    num_vec_cores: int,
):
    """T.Parallel rewrite of the SimdVF mhc=4 forward.

    The VMI port packed 16 tokens x 4 mixes (or 4 x 16) per 64-lane vector
    with vgather + shift index math and vsel-composed base vectors; the
    scalar form runs one full-24-column conditional unit per token row: the
    sigmoid folds go in place into the input rows (96B pitch, provably
    aligned for the masked store), then the pre/post columns move into their
    narrow 16B-pitch outputs with per-lane affine scatters.
    """
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[mhc_mult3, T.float32]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), T.float32]

    tbs = token_block_size

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc_mult3, 8), T.float32)

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc_mult3):
                base_ub[i, 0] = mhc_base[i]

        input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_res_mix_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        # Sigmoid staging: writing the folds back into the pipelined input
        # slot would make the region a writer of a multi-versioned producer
        # buffer (a data race under the persistent pipeline); a dedicated
        # single-version scratch keeps the protocol clean.
        sig_ub = T.alloc_shared((token_block_size, 8), T.float32)
        T.annotate_buffer_versions(
            {
                input_mixes_ub: num_stages,
                pre_layer_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                comb_res_mix_ub: num_stages,
            }
        )
        # Zero-cost views: compacted base scalars staged in the (free until
        # the move pass) pre_layer_mix words; flat row views for the affine
        # scatter moves.
        base_compact = T.Tensor((mhc_mult3,), T.float32, data=pre_layer_mix_ub.data, scope='shared.dyn')
        mix_flat = T.Tensor((tbs * mhc_mult3,), T.float32, data=input_mixes_ub.data, scope='shared.dyn')
        sig_flat = T.Tensor((tbs * 8,), T.float32, data=sig_ub.data, scope='shared.dyn')
        pre_flat = T.Tensor((tbs * mhc_mult,), T.float32, data=pre_layer_mix_ub.data, scope='shared.dyn')
        post_flat = T.Tensor((tbs * mhc_mult,), T.float32, data=post_layer_mix_ub.data, scope='shared.dyn')

        for pid in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, core_id, num_stages=num_stages):
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_ub)

            with T.SimdVF(lanes=64):
                # Compact the 8-padded base scalars (one unconditional affine
                # gather) into the pre rows, read before they are overwritten.
                for j in T.Parallel(mhc_mult3):
                    base_compact[j] = base_ub[j, 0]
                # Full-24-column conditional units. The token dimension is
                # split into 16-row sub-units: a (64, 24) unit lets the layout
                # pick the 64-extent token axis as the lane axis (the column
                # coefficient becomes 24 and the conditional store degrades to
                # an unsupported conditional scatter); 16-row units keep the
                # 24-column axis vectorized with lane coefficient 1.
                for tq in range(4):
                    for t1q, j in T.Parallel(tbs // 4, mhc_mult3):
                        if j < mhc_mult:
                            sig_flat[(tq * (tbs // 4) + t1q) * 8 + j] = (
                                T.float32(1.0)
                                / (
                                    T.exp(
                                        T.float32(0.0)
                                        - (mix_flat[(tq * (tbs // 4) + t1q) * mhc_mult3 + j] * scale_ub[0, 0] + base_compact[j])
                                    )
                                    + T.float32(1.0)
                                )
                                + T.float32(mhc_pre_eps)
                            )
                        else:
                            if j < 2 * mhc_mult:
                                sig_flat[(tq * (tbs // 4) + t1q) * 8 + j] = (
                                    T.float32(1.0)
                                    / (
                                        T.exp(
                                            T.float32(0.0)
                                            - (mix_flat[(tq * (tbs // 4) + t1q) * mhc_mult3 + j] * scale_ub[1, 0] + base_compact[j])
                                        )
                                        + T.float32(1.0)
                                    )
                                    * T.float32(mhc_post_mult_value)
                                )
                # Comb columns: unconditional full-16-column units (an
                # occasionally-reordered masked store on the last token row of
                # a block raced the copy-out; a plain full-row vstore does
                # not).
                for tq in range(4):
                    for t1q, i2 in T.Parallel(tbs // 4, mhc_mult2):
                        comb_res_mix_ub[tq * (tbs // 4) + t1q, i2] = (
                            mix_flat[(tq * (tbs // 4) + t1q) * mhc_mult3 + 2 * mhc_mult + i2]
                            * scale_ub[2, 0]
                            + base_compact[2 * mhc_mult + i2]
                        )
                # Move the transformed pre/post columns into their 16B-pitch
                # rows (affine per-lane scatter stores; a conditional masked
                # store into those rows cannot prove 32-byte alignment).
                for t1 in T.Parallel(tbs):
                    pre_flat[t1 * mhc_mult + 0] = sig_flat[t1 * 8 + 0]
                    pre_flat[t1 * mhc_mult + 1] = sig_flat[t1 * 8 + 1]
                    pre_flat[t1 * mhc_mult + 2] = sig_flat[t1 * 8 + 2]
                    pre_flat[t1 * mhc_mult + 3] = sig_flat[t1 * 8 + 3]
                    post_flat[t1 * mhc_mult + 0] = sig_flat[t1 * 8 + 4]
                    post_flat[t1 * mhc_mult + 1] = sig_flat[t1 * 8 + 5]
                    post_flat[t1 * mhc_mult + 2] = sig_flat[t1 * 8 + 6]
                    post_flat[t1 * mhc_mult + 3] = sig_flat[t1 * 8 + 7]

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_bwd_parallel(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_post_mult_value: float,
    token_block_size: int,
    dtype: T.dtype,
    num_vec_cores: int,
):
    """Generic fallback: verbatim SimtVF implementation from the VMI port."""
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')

    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), dtype]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 3), dtype]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult3), dtype]

    with T.Kernel(num_vec_cores) as pid:
        with T.SimtVF(threads=256):
            scale_grad_0 = T.alloc_reducer(1, dtype, op='sum')
            scale_grad_1 = T.alloc_reducer(1, dtype, op='sum')
            scale_grad_2 = T.alloc_reducer(1, dtype, op='sum')
            base_grad_reducer = T.alloc_reducer(mhc_mult3, dtype, op='sum')
            scale_grad_0_frag = T.alloc_fragment(1, dtype)
            scale_grad_1_frag = T.alloc_fragment(1, dtype)
            scale_grad_2_frag = T.alloc_fragment(1, dtype)
            base_grad_frag = T.alloc_fragment(mhc_mult3, dtype)
            T.reducer_init(scale_grad_0)
            T.reducer_init(scale_grad_1)
            T.reducer_init(scale_grad_2)
            T.reducer_init(base_grad_reducer)

            for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1):
                grad_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)

                for i1, j in T.Parallel(token_block_size, mhc_mult3):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        inp = input_mixes[i, j]
                        if j < mhc_mult:
                            pre_act = T.sigmoid(inp * mhc_scale[0] + mhc_base[j])
                            grad_frag[i1, j] = pre_layer_mix_grad[i, j] * pre_act * (1 - pre_act)
                        if j >= mhc_mult and j < mhc_mult * 2:
                            post_act = post_layer_mix[i, j - mhc_mult]
                            grad_frag[i1, j] = post_layer_mix_grad[i, j - mhc_mult] * post_act * (1 - post_act / mhc_post_mult_value)
                        if j >= mhc_mult * 2:
                            grad_frag[i1, j] = comb_res_mix_grad[i, j - mhc_mult * 2]

                for i1, j in T.Parallel(token_block_size, mhc_mult3):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        inp = input_mixes[i, j]
                        if j < mhc_mult:
                            T.reducer_update(scale_grad_0[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                        if j >= mhc_mult and j < mhc_mult * 2:
                            T.reducer_update(scale_grad_1[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[1]
                        if j >= mhc_mult * 2:
                            T.reducer_update(scale_grad_2[0], grad_frag[i1, j] * inp)
                            T.reducer_update(base_grad_reducer[j], grad_frag[i1, j])
                            input_mixes_grad[i, j] = grad_frag[i1, j] * mhc_scale[2]

            T.finalize_reducer(scale_grad_0, scale_grad_0_frag)
            T.finalize_reducer(scale_grad_1, scale_grad_1_frag)
            T.finalize_reducer(scale_grad_2, scale_grad_2_frag)
            T.finalize_reducer(base_grad_reducer, base_grad_frag)
            mhc_scale_grad_partial[pid, 0] = scale_grad_0_frag[0]
            mhc_scale_grad_partial[pid, 1] = scale_grad_1_frag[0]
            mhc_scale_grad_partial[pid, 2] = scale_grad_2_frag[0]
            T.copy(base_grad_frag, mhc_base_grad_partial[pid, :])


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_bwd_mhc4_parallel(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_post_mult_value: float,
    token_block_size: int,
    num_stages: int,
    num_vec_cores: int,
):
    """T.Parallel rewrite of the SimdVF mhc=4 backward.

    Logical per-(token, column) form: one conditional unit computes the raw
    gradient g per column segment into input_mixes_grad_ub, dim=0 accumulate
    reduces fold g into the 24 per-column base partials and g*inp (built in
    place in input_mixes_ub, whose values have no later reader) into the 24
    per-column scale partials, then a final conditional unit scales g by its
    segment's mhc_scale.  Padding tokens of the ragged tail write zeros so
    the accumulate reduces stay clean.  The per-core scalar partials are
    summed from the 24-word accumulators in the trailing SimtVF block (the
    VMI port's segmented-mask vcadd finale).
    """
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = 4
    mhc_mult2 = 16
    mhc_mult3 = 24

    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), T.float32]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[mhc_mult3, T.float32]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 3), T.float32]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult3), T.float32]

    tbs = token_block_size

    with T.Kernel(num_vec_cores) as pid:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
        input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        input_mixes_grad_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_res_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        # 24 per-column fp32 partials for the base and scale reductions,
        # accumulated across persistent token blocks via clear=False reduces.
        scale_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        base_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        # Scalar reduction results use separate 32B-aligned rows.
        scale_grad_out_ub = T.alloc_shared((3, 8), T.float32)
        base_grad_out_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
        # Product staging for the scale reduction: writing g*inp back into the
        # pipelined input slot would turn the region into a writer of a
        # multi-versioned producer buffer; a dedicated single-version scratch
        # keeps the pipeline protocol clean (the VMI working set lived in
        # vector registers, which the Parallel path cannot express yet).
        prod_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        T.annotate_buffer_versions(
            {
                input_mixes_ub: num_stages,
                input_mixes_grad_ub: num_stages,
                pre_layer_mix_grad_ub: num_stages,
                post_layer_mix_grad_ub: num_stages,
                post_layer_mix_ub: num_stages,
                comb_res_mix_grad_ub: num_stages,
            }
        )
        base_compact = T.Tensor((mhc_mult3,), T.float32, data=base_grad_out_ub.data, scope='shared.dyn')
        base_part = T.Tensor((mhc_mult3,), T.float32, data=base_grad_acc_ub.data, scope='shared.dyn')
        scale_part = T.Tensor((mhc_mult3,), T.float32, data=scale_grad_acc_ub.data, scope='shared.dyn')

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc_mult3):
                base_ub[i, 0] = mhc_base[i]

        with T.SimdVF(lanes=64):
            # Clear the whole accumulator rows (the 24-word views have a tail
            # the fill/clear contract rejects; the owners are lane-multiples).
            T.clear(base_grad_acc_ub)
            T.clear(scale_grad_acc_ub)
        with T.SimdVF(lanes=64):
            # Compact the 8-padded base scalars once (the staging rows are
            # only consumed at the very end, written by the final reduce).
            for j in T.Parallel(mhc_mult3):
                base_compact[j] = base_ub[j, 0]
        with T.SimdVF():
            # The accumulators live across persistent blocks and the first
            # block's reduces must not overtake the clears under the
            # persistent pipeline (the VMI port kept the same barrier here).
            T.simd.mem_bar('VST_VLD')

        for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1, num_stages=num_stages):
            T.copy(input_mixes[t * token_block_size, 0], input_mixes_ub)
            T.copy(pre_layer_mix_grad[t * token_block_size, 0], pre_layer_mix_grad_ub)
            T.copy(post_layer_mix_grad[t * token_block_size, 0], post_layer_mix_grad_ub)
            T.copy(post_layer_mix[t * token_block_size, 0], post_layer_mix_ub)
            T.copy(comb_res_mix_grad[t * token_block_size, 0], comb_res_mix_grad_ub)
            # Bind the ragged-tail bound outside the region: the dynamic
            # shape symbol itself is not in the region's defined-variable set.
            valid_tokens: T.int32 = num_tokens - t * token_block_size

            with T.SimdVF(lanes=64):
                # Raw per-column gradient g; padding tokens write zero so the
                # accumulate reduces see no stale lanes. 16-row sub-units keep
                # the 24-column axis on the lanes (see the fwd note).
                for tq in range(4):
                    for t1q, j in T.Parallel(tbs // 4, mhc_mult3):
                        if tq * (tbs // 4) + t1q < valid_tokens:
                            if j < mhc_mult:
                                input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = (
                                    pre_layer_mix_grad_ub[tq * (tbs // 4) + t1q, j]
                                    * (
                                        T.float32(1.0)
                                        / (
                                            T.exp(
                                                T.float32(0.0)
                                                - (
                                                    input_mixes_ub[tq * (tbs // 4) + t1q, j] * scale_ub[0, 0]
                                                    + base_compact[j]
                                                )
                                            )
                                            + T.float32(1.0)
                                        )
                                    )
                                    * (
                                        T.float32(1.0)
                                        - T.float32(1.0)
                                        / (
                                            T.exp(
                                                T.float32(0.0)
                                                - (
                                                    input_mixes_ub[tq * (tbs // 4) + t1q, j] * scale_ub[0, 0]
                                                    + base_compact[j]
                                                )
                                            )
                                            + T.float32(1.0)
                                        )
                                    )
                                )
                            else:
                                if j < 2 * mhc_mult:
                                    input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = (
                                        post_layer_mix_grad_ub[tq * (tbs // 4) + t1q, j - mhc_mult]
                                        * post_layer_mix_ub[tq * (tbs // 4) + t1q, j - mhc_mult]
                                        * (
                                            T.float32(1.0)
                                            - post_layer_mix_ub[tq * (tbs // 4) + t1q, j - mhc_mult]
                                            / T.float32(mhc_post_mult_value)
                                        )
                                    )
                                else:
                                    input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = comb_res_mix_grad_ub[tq * (tbs // 4) + t1q, j - 2 * mhc_mult]
                        else:
                            input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = T.float32(0.0)
                # base partials: fold g over tokens into the 24 columns.
                T.reduce_sum(input_mixes_grad_ub, base_part, dim=0, clear=False)
                # g * inp into the dedicated scratch, then fold into the
                # 24 per-column scale partials.
                for tq in range(4):
                    for t1q, j in T.Parallel(tbs // 4, mhc_mult3):
                        prod_ub[tq * (tbs // 4) + t1q, j] = (
                            input_mixes_grad_ub[tq * (tbs // 4) + t1q, j]
                            * input_mixes_ub[tq * (tbs // 4) + t1q, j]
                        )
                T.reduce_sum(prod_ub, scale_part, dim=0, clear=False)
                # Final per-segment scaling of g.
                for tq in range(4):
                    for t1q, j in T.Parallel(tbs // 4, mhc_mult3):
                        if j < mhc_mult:
                            input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = (
                                input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] * scale_ub[0, 0]
                            )
                        else:
                            if j < 2 * mhc_mult:
                                input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = (
                                    input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] * scale_ub[1, 0]
                                )
                            else:
                                input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] = (
                                    input_mixes_grad_ub[tq * (tbs // 4) + t1q, j] * scale_ub[2, 0]
                                )

            T.copy(input_mixes_grad_ub, input_mixes_grad[t * token_block_size, 0])

        with T.SimtVF(threads=32):
            # Scalar finale: the 24 per-column partials collapse into the
            # three scale sums and copy out as the per-core base partials
            # (the VMI port's segmented-mask vcadd block).
            for i in T.Parallel(3):
                mhc_scale_grad_partial[pid, i] = T.float32(0.0)
            for i in T.Parallel(1):
                mhc_scale_grad_partial[pid, 0] = (
                    scale_part[0] + scale_part[1] + scale_part[2] + scale_part[3]
                )
                mhc_scale_grad_partial[pid, 1] = (
                    scale_part[4] + scale_part[5] + scale_part[6] + scale_part[7]
                )
                mhc_scale_grad_partial[pid, 2] = (
                    scale_part[8] + scale_part[9] + scale_part[10] + scale_part[11]
                    + scale_part[12] + scale_part[13] + scale_part[14] + scale_part[15]
                    + scale_part[16] + scale_part[17] + scale_part[18] + scale_part[19]
                    + scale_part[20] + scale_part[21] + scale_part[22] + scale_part[23]
                )
            for i in T.Parallel(mhc_mult3):
                mhc_base_grad_partial[pid, i] = base_part[i]
