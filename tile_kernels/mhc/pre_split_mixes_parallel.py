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
        T.annotate_buffer_versions(
            {
                input_mixes_ub: num_stages,
                pre_layer_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                comb_res_mix_ub: num_stages,
            }
        )

        for pid in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, core_id, num_stages=num_stages):
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_ub)

            with T.SimdVF(lanes=64):
                for flat in T.Parallel(token_block_size * mhc_mult):
                    row = flat // mhc_mult
                    j = flat % mhc_mult
                    pre = input_mixes_ub[row, j] * scale_ub[0, 0] + base_ub[j, 0]
                    post = input_mixes_ub[row, j + mhc_mult] * scale_ub[1, 0] + base_ub[j + mhc_mult, 0]
                    pre_layer_mix_ub[row, j] = 1.0 / (1.0 + T.exp(-pre)) + mhc_pre_eps
                    post_layer_mix_ub[row, j] = mhc_post_mult_value / (1.0 + T.exp(-post))
                for flat in T.Parallel(token_block_size * mhc_mult2):
                    row = flat // mhc_mult2
                    j = flat % mhc_mult2
                    comb_res_mix_ub[row, j] = input_mixes_ub[row, j + 2 * mhc_mult] * scale_ub[2, 0] + base_ub[j + 2 * mhc_mult, 0]

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

    with T.Kernel(num_vec_cores) as pid:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
        input_mixes_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        input_mixes_grad_ub = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
        pre_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        post_layer_mix_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        comb_res_mix_grad_ub = T.alloc_shared((token_block_size, mhc_mult2), T.float32)
        # Keep six vector-lane partials across persistent token blocks: three
        # scale reductions plus pre/post/comb base reductions.  A full-mask
        # Keep the original 256-byte accumulator row stride.
        scale_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        base_grad_acc_ub = T.alloc_shared((3, 64), T.float32)
        # Scalar reduction results use separate 32B-aligned rows.
        scale_grad_out_ub = T.alloc_shared((3, 8), T.float32)
        base_grad_out_ub = T.alloc_shared((mhc_mult3, 8), T.float32)
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

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc_mult3):
                base_ub[i, 0] = mhc_base[i]

        with T.SimdVF(lanes=64):
            for i, j in T.Parallel(3, 64):
                scale_grad_acc_ub[i, j] = 0.0
                base_grad_acc_ub[i, j] = 0.0

        for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1, num_stages=num_stages):
            T.copy(input_mixes[t * token_block_size, 0], input_mixes_ub)
            T.copy(pre_layer_mix_grad[t * token_block_size, 0], pre_layer_mix_grad_ub)
            T.copy(post_layer_mix_grad[t * token_block_size, 0], post_layer_mix_grad_ub)
            T.copy(post_layer_mix[t * token_block_size, 0], post_layer_mix_ub)
            T.copy(comb_res_mix_grad[t * token_block_size, 0], comb_res_mix_grad_ub)

            with T.SimdVF(lanes=64):
                local_grad = T.alloc_shared((token_block_size, mhc_mult3), T.float32)
                for row, j in T.Parallel(token_block_size, mhc_mult3):
                    value = input_mixes_ub[row, j]
                    if j < mhc_mult:
                        act = 1.0 / (1.0 + T.exp(-(value * scale_ub[0, 0] + base_ub[j, 0])))
                        grad = pre_layer_mix_grad_ub[row, j % mhc_mult] * act * (1.0 - act)
                        local_grad[row, j] = T.if_then_else(t * token_block_size + row < num_tokens, grad, 0.0)
                        input_mixes_grad_ub[row, j] = grad * scale_ub[0, 0]
                    elif j < 2 * mhc_mult:
                        act = post_layer_mix_ub[row, j % mhc_mult]
                        grad = post_layer_mix_grad_ub[row, j % mhc_mult] * act * (1.0 - act / mhc_post_mult_value)
                        local_grad[row, j] = T.if_then_else(t * token_block_size + row < num_tokens, grad, 0.0)
                        input_mixes_grad_ub[row, j] = grad * scale_ub[1, 0]
                    else:
                        grad = comb_res_mix_grad_ub[row, T.max(j - 2 * mhc_mult, 0)]
                        local_grad[row, j] = T.if_then_else(t * token_block_size + row < num_tokens, grad, 0.0)
                        input_mixes_grad_ub[row, j] = grad * scale_ub[2, 0]
                for j in T.Parallel(mhc_mult3):
                    base_acc = T.alloc_var(T.float32, init=base_grad_acc_ub[0, j])
                    scale_acc = T.alloc_var(T.float32, init=scale_grad_acc_ub[0, j])
                    for row in T.serial(token_block_size):
                        base_acc = base_acc + local_grad[row, j]
                        scale_acc = scale_acc + T.if_then_else(
                            t * token_block_size + row < num_tokens, local_grad[row, j] * input_mixes_ub[row, j], 0.0
                        )
                    base_grad_acc_ub[0, j] = base_acc
                    scale_grad_acc_ub[0, j] = scale_acc

            T.copy(input_mixes_grad_ub, input_mixes_grad[t * token_block_size, 0])

        with T.SimdVF(lanes=64):
            T.reduce_sum(scale_grad_acc_ub[0, 0:mhc_mult], scale_grad_out_ub[0, 0:1], dim=1)
            T.reduce_sum(scale_grad_acc_ub[0, mhc_mult : 2 * mhc_mult], scale_grad_out_ub[1, 0:1], dim=1)
            T.reduce_sum(scale_grad_acc_ub[0, 2 * mhc_mult : mhc_mult3], scale_grad_out_ub[2, 0:1], dim=1)
            for j in T.serial(mhc_mult3):
                for point in T.Parallel(1):
                    base_grad_out_ub[j, point] = base_grad_acc_ub[0, j + point]

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                mhc_scale_grad_partial[pid, i] = scale_grad_out_ub[i, 0]
            for i in T.Parallel(mhc_mult3):
                mhc_base_grad_partial[pid, i] = base_grad_out_ub[i, 0]
