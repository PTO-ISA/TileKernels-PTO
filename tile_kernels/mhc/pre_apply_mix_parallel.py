import tilelang
from tilelang.ascend import language as T


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit(target='pto')
def mhc_pre_apply_mix_fwd_parallel(x, mix, o, h_blk, num_vec_cores: int):
    """Ascend forward: ``o[n, h] = sum_mhc mix[n, mhc] * x[n, mhc, h]``."""
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    x: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    o: T.Tensor[(num_tokens, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        o_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, mix_ub: num_stages, o_ub: num_stages})

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])
            T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)

            with T.SimdVF(lanes=64):
                # One scalar expression per element: widen each bf16 operand to
                # fp32, take the mix scalars as lane-uniform UB reads, and sum
                # the mhc products (the VMI port used a vmula chain; plain
                # mul+add is the T.Parallel equivalent). The python loop runs
                # at trace time (mhc is bound by then), building one store.
                for h in T.Parallel(h_blk):
                    o_ub[h] = T.Cast(
                        'bfloat16',
                        sum(
                            (mix_ub[m] * T.Cast('float32', x_ub[m, h]) for m in range(1, mhc)),
                            mix_ub[0] * T.Cast('float32', x_ub[0, h]),
                        ),
                    )

            T.copy(o_ub, o[pid_n, pid_h * h_blk])


@tilelang.jit(target='pto')
def mhc_pre_apply_mix_bwd_parallel(o_grad, x, mix, x_grad, h_blk, num_stages, num_vec_cores: int):
    """Ascend backward.

    ``x_grad[n, m, h] += mix[n, m] * o_grad[n, h]``  (accumulated onto existing values)
    ``mix_grad[n, m] = sum_h o_grad[n, h] * x[n, m, h]``  (reduction over h)
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    o_grad: T.Tensor[(num_tokens, hidden), T.bfloat16]
    x: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    x_grad: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix_grad = T.empty((num_tokens, mhc), T.float32)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((h_blk,), T.bfloat16)
        x_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_grad_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        mix_grad_acc = T.alloc_shared((mhc, _VEC), T.float32)
        mix_grad_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        T.annotate_buffer_versions(
            {
                o_grad_ub: num_stages,
                x_ub: num_stages,
                x_grad_ub: num_stages,
                mix_ub: num_stages,
                mix_grad_ub: num_stages,
            }
        )
        # Per-mhc destination view for the final cross-lane reduction of the
        # 64-lane accumulators (zero-cost alias of the padded output words).
        mix_grad_total = T.Tensor((mhc,), T.float32, data=mix_grad_ub.data, scope='shared.dyn')

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])

            with T.SimdVF(lanes=64):
                T.clear(mix_grad_acc)

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(o_grad[pid_n, pid_h * h_blk], o_grad_ub)
                T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)
                T.copy(x_grad[pid_n, 0, pid_h * h_blk], x_grad_ub)

                with T.SimdVF(lanes=64):
                    # Single sweep per (m, lane), 8-way chunk unroll: the
                    # x_grad updates and the mix_grad accumulation share the
                    # widened o_grad/x reads, and folding eight products per
                    # accumulator round-trip amortizes the UB RMW that the
                    # VMI port kept in a vector register (no private reducer
                    # storage on the Parallel path yet).
                    for m, lane in T.Parallel(mhc, _VEC):
                        for c4 in T.serial(num_chunks // 8):
                            for u in range(8):
                                x_grad_ub[m, (8 * c4 + u) * _VEC + lane] = T.Cast(
                                    'bfloat16',
                                    T.Cast('float32', x_grad_ub[m, (8 * c4 + u) * _VEC + lane])
                                    + T.Cast('float32', o_grad_ub[(8 * c4 + u) * _VEC + lane]) * mix_ub[m],
                                )
                            mix_grad_acc[m, lane] = sum(
                                (
                                    T.Cast('float32', o_grad_ub[(8 * c4 + u) * _VEC + lane])
                                    * T.Cast('float32', x_ub[m, (8 * c4 + u) * _VEC + lane])
                                    for u in range(8)
                                ),
                                mix_grad_acc[m, lane],
                            )
                        for ct in range(num_chunks - num_chunks % 8, num_chunks):
                            x_grad_ub[m, ct * _VEC + lane] = T.Cast(
                                'bfloat16',
                                T.Cast('float32', x_grad_ub[m, ct * _VEC + lane])
                                + T.Cast('float32', o_grad_ub[ct * _VEC + lane]) * mix_ub[m],
                            )
                            mix_grad_acc[m, lane] = (
                                mix_grad_acc[m, lane]
                                + T.Cast('float32', o_grad_ub[ct * _VEC + lane])
                                * T.Cast('float32', x_ub[m, ct * _VEC + lane])
                            )

                T.copy(x_grad_ub, x_grad[pid_n, 0, pid_h * h_blk])

            with T.SimdVF(lanes=64):
                # Cross-lane fold of each 64-lane accumulator row (the VMI port
                # used vcadd with reassoc=True; the direct reduce emits the
                # same pairwise tree).
                T.reduce_sum(mix_grad_acc, mix_grad_total, dim=1)
            T.copy(mix_grad_ub[0:mhc], mix_grad[pid_n, 0])

    return mix_grad
