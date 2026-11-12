"""T.Parallel VF bodies; scheduling and DMA copied from pre_apply_mix_asc.py."""

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
    num_chunks = h_blk // _VEC
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
                for j in T.Parallel(h_blk):
                    acc = T.alloc_var(T.float32, init=T.float32(x_ub[0, j]) * mix_ub[0])
                    for m in T.serial(1, mhc):
                        acc += T.float32(x_ub[m, j]) * mix_ub[m]
                    o_ub[j] = T.cast(acc, T.bfloat16)

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

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])

            with T.SimdVF(lanes=64):
                for m, j in T.Parallel(mhc, _VEC):
                    mix_grad_acc[m, j] = 0.0

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(o_grad[pid_n, pid_h * h_blk], o_grad_ub)
                T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)
                T.copy(x_grad[pid_n, 0, pid_h * h_blk], x_grad_ub)

                with T.SimdVF(lanes=64):
                    for m, j in T.Parallel(mhc, h_blk):
                        x_grad_ub[m, j] = T.cast(T.float32(x_grad_ub[m, j]) + T.float32(o_grad_ub[j]) * mix_ub[m], T.bfloat16)
                    for m, j in T.Parallel(mhc, _VEC):
                        acc = T.alloc_var(T.float32, init=mix_grad_acc[m, j])
                        for c in T.serial(num_chunks):
                            acc += T.float32(o_grad_ub[c * _VEC + j]) * T.float32(x_ub[m, c * _VEC + j])
                        mix_grad_acc[m, j] = acc

                T.copy(x_grad_ub, x_grad[pid_n, 0, pid_h * h_blk])

            with T.SimdVF(lanes=64):
                T.reduce_sum(mix_grad_acc, mix_grad_ub[0:mhc], dim=1)
            T.copy(mix_grad_ub[0:mhc], mix_grad[pid_n, 0])

    return mix_grad
