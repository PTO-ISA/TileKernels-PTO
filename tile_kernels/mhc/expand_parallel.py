"""PTO SIMD implementation, retaining the ASC persistent schedule and buffers."""

import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def expand_to_mhc_fwd_parallel(x, o, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    x: T.Tensor[(num_tokens, h), T.bfloat16]
    o: T.Tensor[(num_tokens, mhc, h), T.bfloat16]
    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        o_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, o_ub: num_stages})
        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(x[pid_n, pid_h * h_blk], x_ub)
            with T.SimdVF(lanes=128):
                for m, j in T.Parallel(mhc, h_blk):
                    o_ub[m, j] = x_ub[j]
            T.copy(o_ub, o[pid_n, 0, pid_h * h_blk])


@tilelang.jit(target='pto')
def expand_to_mhc_bwd_parallel(o_grad, x_grad, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    o_grad: T.Tensor[(num_tokens, mhc, h), T.bfloat16]
    x_grad: T.Tensor[(num_tokens, h), T.bfloat16]
    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_grad_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({o_grad_ub: num_stages, x_grad_ub: num_stages})
        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(o_grad[pid_n, 0, pid_h * h_blk], o_grad_ub)
            with T.SimdVF(lanes=64):
                for j in T.Parallel(h_blk):
                    acc = T.alloc_var(T.float32, init=T.float32(o_grad_ub[0, j]))
                    for m in T.serial(1, mhc):
                        acc += T.float32(o_grad_ub[m, j])
                    x_grad_ub[j] = T.cast(acc, T.bfloat16)
            T.copy(x_grad_ub, x_grad[pid_n, pid_h * h_blk])
