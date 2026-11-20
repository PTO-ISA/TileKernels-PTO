import tilelang
from tilelang.ascend import language as T


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit(target='pto')
def mhc_post_fwd_parallel(comb_res_mix, residual, post_layer_mix, x, out, h_blk, num_vec_cores: int):
    """Ascend forward.

    ``out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                      + sum_mi comb_res_mix[n, mi, mo] * residual[n, mi, h]``
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), T.bfloat16]
    out: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        out_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                residual_ub: num_stages,
                x_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                out_ub: num_stages,
            }
        )

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])
            T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)
            T.copy(x[pid_n, pid_h * h_blk], x_ub)

            with T.SimdVF(lanes=64):
                # One scalar expression per output element: widen each bf16
                # operand to fp32, take the mix scalars as lane-uniform UB
                # reads, and fold x*post + sum_mi res*comb in the same order
                # as the VMI vmula chain. The python generator runs at trace
                # time (mhc is bound by then), building one store.
                for mo, h in T.Parallel(mhc, h_blk):
                    out_ub[mo, h] = T.Cast(
                        'bfloat16',
                        sum(
                            (
                                comb_res_mix_ub[mi, mo] * T.Cast('float32', residual_ub[mi, h])
                                for mi in range(mhc)
                            ),
                            post_layer_mix_ub[mo] * T.Cast('float32', x_ub[h]),
                        ),
                    )

            T.copy(out_ub, out[pid_n, 0, pid_h * h_blk])


@tilelang.jit(target='pto')
def mhc_post_bwd_parallel(d_o, comb_res_mix, residual, post_layer_mix, x, h_blk, num_stages, num_vec_cores: int):
    """Ascend backward.

    d_x[n, h] = sum_mo d_o[n, mo, h] * post_layer_mix[n, mo]
    d_post_layer_mix[n, mo] = sum_h d_o[n, mo, h] * x[n, h]
    d_residual[n, mi, h] = sum_mo d_o[n, mo, h] * comb_res_mix[n, mi, mo]
    d_comb_res_mix[n, mi, mo] = sum_h d_o[n, mo, h] * residual[n, mi, h]
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    d_o: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), T.bfloat16]

    d_comb_res_mix = T.empty((num_tokens, mhc, mhc), T.float32)
    d_residual = T.empty((num_tokens, mhc, hidden), T.bfloat16)
    d_post_layer_mix = T.empty((num_tokens, mhc), T.float32)
    d_x = T.empty((num_tokens, hidden), T.bfloat16)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        d_o_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        d_residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        d_post_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_comb_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_mix_acc = T.alloc_shared((mhc, _VEC), T.float32)
        comb_mix_acc = T.alloc_shared((mhc * mhc, _VEC), T.float32)
        T.annotate_buffer_versions(
            {
                d_o_ub: num_stages,
                x_ub: num_stages,
                residual_ub: num_stages,
                d_x_ub: num_stages,
                d_residual_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                d_post_mix_ub: 1,
                d_comb_mix_ub: 1,
                post_mix_acc: 1,
                comb_mix_acc: 1,
            }
        )
        # Zero-cost destination views for the final cross-lane reductions.
        d_post_total = T.Tensor((mhc,), T.float32, data=d_post_mix_ub.data, scope='shared.dyn')
        d_comb_total = T.Tensor((mhc * mhc,), T.float32, data=d_comb_mix_ub.data, scope='shared.dyn')

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])

            with T.SimdVF(lanes=64):
                T.clear(post_mix_acc)
                T.clear(comb_mix_acc)

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(d_o[pid_n, 0, pid_h * h_blk], d_o_ub)
                T.copy(x[pid_n, pid_h * h_blk], x_ub)
                T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)

                with T.SimdVF(lanes=64):
                    # d_x: fold the mo products in the same order as the VMI
                    # vmula chain (zero start).
                    for h in T.Parallel(h_blk):
                        d_x_ub[h] = T.Cast(
                            'bfloat16',
                            sum(
                                (
                                    T.Cast('float32', d_o_ub[mo, h]) * post_layer_mix_ub[mo]
                                    for mo in range(1, mhc)
                                ),
                                T.Cast('float32', d_o_ub[0, h]) * post_layer_mix_ub[0],
                            ),
                        )
                    # d_post accumulation: 64-lane accumulators, 4-way
                    # unrolled serial chunk shell (amortizes the accumulator's
                    # UB round trip; same per-chunk order as the VMI sweep).
                    for mo, lane in T.Parallel(mhc, _VEC):
                        for c4 in T.serial(num_chunks // 4):
                            post_mix_acc[mo, lane] = sum(
                                (
                                    T.Cast('float32', d_o_ub[mo, (4 * c4 + u) * _VEC + lane])
                                    * T.Cast('float32', x_ub[(4 * c4 + u) * _VEC + lane])
                                    for u in range(4)
                                ),
                                post_mix_acc[mo, lane],
                            )
                        for ct in range(num_chunks - num_chunks % 4, num_chunks):
                            post_mix_acc[mo, lane] = (
                                post_mix_acc[mo, lane]
                                + T.Cast('float32', d_o_ub[mo, ct * _VEC + lane])
                                * T.Cast('float32', x_ub[ct * _VEC + lane])
                            )
                    # d_residual rows.
                    for mi, h in T.Parallel(mhc, h_blk):
                        d_residual_ub[mi, h] = T.Cast(
                            'bfloat16',
                            sum(
                                (
                                    T.Cast('float32', d_o_ub[mo, h]) * comb_res_mix_ub[mi, mo]
                                    for mo in range(1, mhc)
                                ),
                                T.Cast('float32', d_o_ub[0, h]) * comb_res_mix_ub[mi, 0],
                            ),
                        )
                    # d_comb accumulation: the (mi, mo) pairs are flattened to
                    # one 2D unit; the kept coordinate recovers mi/mo with
                    # trace-constant mhc divmod (lane-uniform integer math).
                    for mo, lane in T.Parallel(mhc * mhc, _VEC):
                        for c4 in T.serial(num_chunks // 4):
                            comb_mix_acc[mo, lane] = sum(
                                (
                                    T.Cast('float32', d_o_ub[mo % mhc, (4 * c4 + u) * _VEC + lane])
                                    * T.Cast('float32', residual_ub[mo // mhc, (4 * c4 + u) * _VEC + lane])
                                    for u in range(4)
                                ),
                                comb_mix_acc[mo, lane],
                            )
                        for ct in range(num_chunks - num_chunks % 4, num_chunks):
                            comb_mix_acc[mo, lane] = (
                                comb_mix_acc[mo, lane]
                                + T.Cast('float32', d_o_ub[mo % mhc, ct * _VEC + lane])
                                * T.Cast('float32', residual_ub[mo // mhc, ct * _VEC + lane])
                            )

                T.copy(d_x_ub, d_x[pid_n, pid_h * h_blk])
                T.copy(d_residual_ub, d_residual[pid_n, 0, pid_h * h_blk])

            with T.SimdVF(lanes=64):
                # Cross-lane fold of each 64-lane accumulator row (the VMI
                # port used vcadd with reassoc=True; the direct reduce emits
                # the same pairwise tree).
                T.reduce_sum(post_mix_acc, d_post_total, dim=1)
                T.reduce_sum(comb_mix_acc, d_comb_total, dim=1)

            T.copy(d_post_mix_ub[0:mhc], d_post_layer_mix[pid_n, 0])
            T.copy(d_comb_mix_ub, d_comb_res_mix[pid_n, 0, 0])

    return d_comb_res_mix, d_residual, d_post_layer_mix, d_x
