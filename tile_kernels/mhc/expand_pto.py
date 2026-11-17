import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def expand_to_mhc_fwd_pto(x, o, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    x: T.Tensor[(num_tokens, h), T.bfloat16]
    o: T.Tensor[(num_tokens, mhc, h), T.bfloat16]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        o_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions({
            x_ub: (num_stages, 'iteration'),
            o_ub: (num_stages, 'iteration'),
        })

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(x[pid_n, pid_h * h_blk], x_ub)
            with T.SimdVF():
                for m, i in T.Parallel(mhc, h_blk):
                    o_ub[m, i] = x_ub[i]

            T.copy(o_ub, o[pid_n, 0, pid_h * h_blk])


@tilelang.jit(target='pto')
def expand_to_mhc_bwd_pto(o_grad, x_grad, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    o_grad: T.Tensor[(num_tokens, mhc, h), T.bfloat16]
    x_grad: T.Tensor[(num_tokens, h), T.bfloat16]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_grad_f32_ub = T.alloc_shared((h_blk,), T.float32)
        x_grad_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({
            o_grad_ub: (num_stages, 'iteration'),
            x_grad_f32_ub: (num_stages, 'iteration'),
            x_grad_ub: (num_stages, 'iteration'),
        })

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(o_grad[pid_n, 0, pid_h * h_blk], o_grad_ub)

            with T.SimdVF(lanes=128):
                T.reduce_sum(o_grad_ub, x_grad_f32_ub, dim=0, clear=True)

            with T.SimdVF(lanes=128):
                for i in T.Parallel(h_blk):
                    x_grad_ub[i] = T.cast(x_grad_f32_ub[i], T.bfloat16)

            T.copy(x_grad_ub, x_grad[pid_n, pid_h * h_blk])
