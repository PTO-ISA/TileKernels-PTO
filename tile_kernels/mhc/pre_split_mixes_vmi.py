import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


def _u32_index(index_i32):
    return V.vinterpret_cast(index_i32, 'uint32')


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_fwd_vmi(
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
def mhc_pre_split_mixes_fwd_simd_mhc4_vmi(
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

            with T.SimdVF():
                mask_full = V.create_mask(64, size=64)
                idx = V.vci(T.int32(0), size=64)
                zeros = V.vbrc(T.float32(0), size=64)
                ones = V.vbrc(T.float32(1), size=64)
                col3 = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col15 = V.vand(idx, V.vbrc(T.int32(15), size=64))
                scale0 = V.vbrc(V.vload(scale_ub[0, 0], size=1), size=64)
                scale1 = V.vbrc(V.vload(scale_ub[1, 0], size=1), size=64)
                scale2 = V.vbrc(V.vload(scale_ub[2, 0], size=1), size=64)
                pre_base_vec = V.alloc_var(T.float32, size=64)
                post_base_vec = V.alloc_var(T.float32, size=64)
                comb_base_vec = V.alloc_var(T.float32, size=64)
                pre_base_vec = zeros
                post_base_vec = zeros
                comb_base_vec = zeros

                for base_id in range(4):
                    base_mask = V.vcmps(col3, T.int32(base_id), mask_full, 'eq')
                    pre_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id, 0], size=1), size=64), pre_base_vec)
                    post_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id + 4, 0], size=1), size=64), post_base_vec)

                for base_id in range(16):
                    base_mask = V.vcmps(col15, T.int32(base_id), mask_full, 'eq')
                    comb_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id + 8, 0], size=1), size=64), comb_base_vec)

                for vec_id in T.serial(4):
                    token = V.vadd(V.vshr(idx, V.vbrc(T.int32(2), size=64)), V.vbrc(T.int32(vec_id * 16), size=64))
                    token_offset = V.vadd(V.vshl(token, V.vbrc(T.int32(4), size=64)), V.vshl(token, V.vbrc(T.int32(3), size=64)))
                    pre_index = V.vadd(token_offset, col3)
                    post_index = V.vadd(pre_index, V.vbrc(T.int32(4), size=64))

                    pre_val = V.vadd(V.vmul(V.vgather(input_mixes_ub[0, 0], _u32_index(pre_index), mask_full), scale0), pre_base_vec)
                    pre_sig = V.vdiv(ones, V.vadd(V.vexpdif(zeros, pre_val, mask_full), ones))
                    V.vstore(V.vadd(pre_sig, V.vbrc(T.float32(mhc_pre_eps), size=64)), pre_layer_mix_ub[vec_id * 16, 0])

                    post_val = V.vadd(V.vmul(V.vgather(input_mixes_ub[0, 0], _u32_index(post_index), mask_full), scale1), post_base_vec)
                    post_sig = V.vdiv(ones, V.vadd(V.vexpdif(zeros, post_val, mask_full), ones))
                    V.vstore(V.vmul(post_sig, V.vbrc(T.float32(mhc_post_mult_value), size=64)), post_layer_mix_ub[vec_id * 16, 0])

                for vec_id in T.serial(16):
                    token = V.vadd(V.vshr(idx, V.vbrc(T.int32(4), size=64)), V.vbrc(T.int32(vec_id * 4), size=64))
                    token_offset = V.vadd(V.vshl(token, V.vbrc(T.int32(4), size=64)), V.vshl(token, V.vbrc(T.int32(3), size=64)))
                    comb_index = V.vadd(V.vadd(token_offset, V.vbrc(T.int32(8), size=64)), col15)
                    comb_val = V.vadd(
                        V.vmul(V.vgather(input_mixes_ub[0, 0], _u32_index(comb_index), mask_full), scale2),
                        comb_base_vec,
                    )
                    V.vstore(comb_val, comb_res_mix_ub[vec_id * 4, 0])

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0])
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])


@tilelang.jit(target='pto')
def mhc_pre_split_mixes_bwd_vmi(
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
def mhc_pre_split_mixes_bwd_mhc4_vmi(
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
        # VMI vstore writes 64 fp32 values, so every accumulator row is 256B.
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

        with T.SimdVF():
            zeros = V.vbrc(T.float32(0), size=64)
            # These UB buffers carry vector-lane partials across persistent
            # token blocks.  Initialize the buffers themselves: initializing
            # temporary registers would leave the first iteration reading stale
            # UB contents.
            for i in T.unroll(3, explicit=True):
                V.vstore(zeros, scale_grad_acc_ub[i, 0])
                V.vstore(zeros, base_grad_acc_ub[i, 0])
            T.simd.mem_bar('VST_VLD')

        for t in T.Persistent([T.ceildiv(num_tokens, token_block_size)], num_vec_cores, pid, group_size=1, num_stages=num_stages):
            T.copy(input_mixes[t * token_block_size, 0], input_mixes_ub)
            T.copy(pre_layer_mix_grad[t * token_block_size, 0], pre_layer_mix_grad_ub)
            T.copy(post_layer_mix_grad[t * token_block_size, 0], post_layer_mix_grad_ub)
            T.copy(post_layer_mix[t * token_block_size, 0], post_layer_mix_ub)
            T.copy(comb_res_mix_grad[t * token_block_size, 0], comb_res_mix_grad_ub)

            with T.SimdVF():
                mask_full = V.create_mask(64, size=64)
                idx = V.vci(T.int32(0), size=64)
                col3 = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col15 = V.vand(idx, V.vbrc(T.int32(15), size=64))
                zeros = V.vbrc(T.float32(0), size=64)
                ones = V.vbrc(T.float32(1), size=64)
                scale0 = V.vbrc(V.vload(scale_ub[0, 0], size=1), size=64)
                scale1 = V.vbrc(V.vload(scale_ub[1, 0], size=1), size=64)
                scale2 = V.vbrc(V.vload(scale_ub[2, 0], size=1), size=64)
                post_mult = V.vbrc(T.float32(mhc_post_mult_value), size=64)
                pre_base_vec = V.alloc_var(T.float32, size=64)
                post_base_vec = V.alloc_var(T.float32, size=64)
                comb_base_vec = V.alloc_var(T.float32, size=64)
                pre_base_vec = zeros
                post_base_vec = zeros
                comb_base_vec = zeros
                for base_id in range(4):
                    base_mask = V.vcmps(col3, T.int32(base_id), mask_full, 'eq')
                    pre_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id, 0], size=1), size=64), pre_base_vec)
                    post_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id + 4, 0], size=1), size=64), post_base_vec)
                for base_id in range(16):
                    base_mask = V.vcmps(col15, T.int32(base_id), mask_full, 'eq')
                    comb_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id + 8, 0], size=1), size=64), comb_base_vec)

                scale_acc = V.alloc_local((3,), V.vreg(64, T.float32))
                base_acc = V.alloc_local((3,), V.vreg(64, T.float32))
                for i in T.unroll(3, explicit=True):
                    scale_acc[i] = V.vload(scale_grad_acc_ub[i, 0], size=64)
                    base_acc[i] = V.vload(base_grad_acc_ub[i, 0], size=64)

                # The last persistent block may contain fewer than 64 tokens.
                # The hot full-block specialization uses the all-lane mask;
                # only the final block materializes per-vector tail masks.
                valid_tokens = T.min(token_block_size, num_tokens - t * token_block_size)
                for vec_id in T.serial(4):
                    token = V.vadd(V.vshr(idx, V.vbrc(T.int32(2), size=64)), V.vbrc(T.int32(vec_id * 16), size=64))
                    token4 = V.vshl(token, V.vbrc(T.int32(2), size=64))
                    token24 = V.vadd(V.vshl(token, V.vbrc(T.int32(4), size=64)), V.vshl(token, V.vbrc(T.int32(3), size=64)))
                    pre_index = V.vadd(token24, col3)
                    post_index = V.vadd(pre_index, V.vbrc(T.int32(4), size=64))
                    grad_index = V.vadd(token4, col3)
                    token_mask = V.vcmps(token, valid_tokens, mask_full, 'lt')

                    pre_inp = V.vgather(input_mixes_ub[0, 0], _u32_index(pre_index), token_mask)
                    pre_grad_out = V.vgather(pre_layer_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    pre_act = V.vdiv(
                        ones, V.vadd(V.vexpdif(zeros, V.vadd(V.vmul(pre_inp, scale0), pre_base_vec), mask_full), ones)
                    )
                    pre_grad = V.vmul(V.vmul(pre_grad_out, pre_act), V.vsub(ones, pre_act))
                    V.vscatter(V.vmul(pre_grad, scale0), input_mixes_grad_ub[0, 0], _u32_index(pre_index), token_mask)
                    scale_acc[0] = V.vadd(scale_acc[0], V.vmul(pre_grad, pre_inp))
                    base_acc[0] = V.vadd(base_acc[0], pre_grad)

                    post_inp = V.vgather(input_mixes_ub[0, 0], _u32_index(post_index), token_mask)
                    post_grad_out = V.vgather(post_layer_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    post_act = V.vgather(post_layer_mix_ub[0, 0], _u32_index(grad_index), token_mask)
                    post_grad = V.vmul(V.vmul(post_grad_out, post_act), V.vsub(ones, V.vdiv(post_act, post_mult)))
                    V.vscatter(V.vmul(post_grad, scale1), input_mixes_grad_ub[0, 0], _u32_index(post_index), token_mask)
                    scale_acc[1] = V.vadd(scale_acc[1], V.vmul(post_grad, post_inp))
                    base_acc[1] = V.vadd(base_acc[1], post_grad)

                for vec_id in T.serial(16):
                    token = V.vadd(V.vshr(idx, V.vbrc(T.int32(4), size=64)), V.vbrc(T.int32(vec_id * 4), size=64))
                    token16 = V.vshl(token, V.vbrc(T.int32(4), size=64))
                    token24 = V.vadd(token16, V.vshl(token, V.vbrc(T.int32(3), size=64)))
                    input_index = V.vadd(V.vadd(token24, V.vbrc(T.int32(8), size=64)), col15)
                    grad_index = V.vadd(token16, col15)
                    token_mask = V.vcmps(token, valid_tokens, mask_full, 'lt')
                    comb_inp = V.vgather(input_mixes_ub[0, 0], _u32_index(input_index), token_mask)
                    comb_grad = V.vgather(comb_res_mix_grad_ub[0, 0], _u32_index(grad_index), token_mask)
                    V.vscatter(V.vmul(comb_grad, scale2), input_mixes_grad_ub[0, 0], _u32_index(input_index), token_mask)
                    scale_acc[2] = V.vadd(scale_acc[2], V.vmul(comb_grad, comb_inp))
                    base_acc[2] = V.vadd(base_acc[2], comb_grad)

                for i in T.unroll(3, explicit=True):
                    V.vstore(scale_acc[i], scale_grad_acc_ub[i, 0])
                    V.vstore(base_acc[i], base_grad_acc_ub[i, 0])

            T.copy(input_mixes_grad_ub, input_mixes_grad[t * token_block_size, 0])

        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)
            idx = V.vci(T.int32(0), size=64)
            col3 = V.vand(idx, V.vbrc(T.int32(3), size=64))
            col15 = V.vand(idx, V.vbrc(T.int32(15), size=64))
            # ASC vcadd pairwise-reduces the active lanes and stores the
            # single result (ONEPT_B32); VMI vcadd returns a 1-lane vreg, so
            # store it directly as an unmasked one-lane (scalar) store.
            for i in T.unroll(3, explicit=True):
                total = V.vcadd(V.vload(scale_grad_acc_ub[i, 0], size=64), mask_full, reassoc=True)
                V.vstore(total, scale_grad_out_ub[i, 0])
            pre_base_acc = V.vload(base_grad_acc_ub[0, 0], size=64)
            post_base_acc = V.vload(base_grad_acc_ub[1, 0], size=64)
            comb_base_acc = V.vload(base_grad_acc_ub[2, 0], size=64)
            for base_id in T.unroll(4, explicit=True):
                base_mask = V.vcmps(col3, T.int32(base_id), mask_full, 'eq')
                V.vstore(V.vcadd(pre_base_acc, base_mask, reassoc=True), base_grad_out_ub[base_id, 0])
                V.vstore(V.vcadd(post_base_acc, base_mask, reassoc=True), base_grad_out_ub[base_id + 4, 0])
            for base_id in T.unroll(16, explicit=True):
                base_mask = V.vcmps(col15, T.int32(base_id), mask_full, 'eq')
                V.vstore(V.vcadd(comb_base_acc, base_mask, reassoc=True), base_grad_out_ub[base_id + 8, 0])

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                mhc_scale_grad_partial[pid, i] = scale_grad_out_ub[i, 0]
            for i in T.Parallel(mhc_mult3):
                mhc_base_grad_partial[pid, i] = base_grad_out_ub[i, 0]
