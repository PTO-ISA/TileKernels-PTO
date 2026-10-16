import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def expand_to_mhc_fwd_vmi(x, o, h_blk: int, num_stages: int, num_vec_cores: int):
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
            with T.SimdVF():
                num_chunks = h_blk // 128
                for c in T.serial(num_chunks):
                    col = c * 128
                    xb = V.vload(x_ub[col], size=128)
                    for m in T.unroll(mhc, explicit=True):
                        V.vstore(xb, o_ub[m, col])

            T.copy(o_ub, o[pid_n, 0, pid_h * h_blk])


@tilelang.jit(target='pto')
def expand_to_mhc_bwd_vmi(o_grad, x_grad, h_blk: int, num_stages: int, num_vec_cores: int):
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

            with T.SimdVF():
                num_chunks = h_blk // 64
                for c in T.serial(num_chunks):
                    col = c * 64
                    acc = V.alloc_var(T.float32, size=64)
                    acc = V.vcvt(V.vload(o_grad_ub[0, col], size=64), 'float32')
                    for m in T.unroll(mhc - 1, explicit=True):
                        og = V.vcvt(V.vload(o_grad_ub[m + 1, col], size=64), 'float32')
                        acc = V.vadd(acc, og)
                    V.vstore(V.vcvt(acc, 'bfloat16'), x_grad_ub[col])

            T.copy(x_grad_ub, x_grad[pid_n, pid_h * h_blk])
