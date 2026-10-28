"""T.Parallel VF bodies; scheduling and DMA copied from post_asc.py."""

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
    num_chunks = h_blk // _VEC
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
                for mo, j in T.Parallel(mhc, h_blk):
                    acc = T.alloc_var(T.float32, init=T.float32(x_ub[j]) * post_layer_mix_ub[mo])
                    for mi in T.serial(mhc):
                        acc += T.float32(residual_ub[mi, j]) * comb_res_mix_ub[mi, mo]
                    out_ub[mo, j] = T.cast(acc, T.bfloat16)

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

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])

            with T.SimdVF(lanes=64):
                for mo, j in T.Parallel(mhc, _VEC):
                    post_mix_acc[mo, j] = 0.0
                for idx, j in T.Parallel(mhc * mhc, _VEC):
                    comb_mix_acc[idx, j] = 0.0

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(d_o[pid_n, 0, pid_h * h_blk], d_o_ub)
                T.copy(x[pid_n, pid_h * h_blk], x_ub)
                T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)

                with T.SimdVF(lanes=64):
                    for j in T.Parallel(h_blk):
                        acc = T.alloc_var(T.float32, init=0.0)
                        for mo in T.serial(mhc):
                            acc += T.float32(d_o_ub[mo, j]) * post_layer_mix_ub[mo]
                        d_x_ub[j] = T.cast(acc, T.bfloat16)
                    for mi, j in T.Parallel(mhc, h_blk):
                        acc = T.alloc_var(T.float32, init=0.0)
                        for mo in T.serial(mhc):
                            acc += T.float32(d_o_ub[mo, j]) * comb_res_mix_ub[mi, mo]
                        d_residual_ub[mi, j] = T.cast(acc, T.bfloat16)
                    for mo, j in T.Parallel(mhc, _VEC):
                        acc = T.alloc_var(T.float32, init=post_mix_acc[mo, j])
                        for c in T.serial(num_chunks):
                            acc += T.float32(d_o_ub[mo, c * _VEC + j]) * T.float32(x_ub[c * _VEC + j])
                        post_mix_acc[mo, j] = acc
                    for idx, j in T.Parallel(mhc * mhc, _VEC):
                        acc = T.alloc_var(T.float32, init=comb_mix_acc[idx, j])
                        for c in T.serial(num_chunks):
                            acc += T.float32(d_o_ub[idx % mhc, c * _VEC + j]) * T.float32(residual_ub[idx // mhc, c * _VEC + j])
                        comb_mix_acc[idx, j] = acc

                T.copy(d_x_ub, d_x[pid_n, pid_h * h_blk])
                T.copy(d_residual_ub, d_residual[pid_n, 0, pid_h * h_blk])

            with T.SimdVF(lanes=64):
                T.reduce_sum(post_mix_acc, d_post_mix_ub[0:mhc], dim=1)
                T.reduce_sum(comb_mix_acc, T.reshape(d_comb_mix_ub, (mhc * mhc,)), dim=1)

            T.copy(d_post_mix_ub[0:mhc], d_post_layer_mix[pid_n, 0])
            T.copy(d_comb_mix_ub, d_comb_res_mix[pid_n, 0, 0])

    return d_comb_res_mix, d_residual, d_post_layer_mix, d_x
