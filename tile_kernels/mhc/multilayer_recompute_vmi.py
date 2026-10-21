import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


_VEC = 64

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
}


@tilelang.jit(target='pto', pass_configs=_PASS_CONFIGS)
def mhc_multilayer_recompute_vmi(
    initial_residual,
    pre_mix_ptrs,
    layer_output_ptrs,
    post_mix_ptrs,
    comb_mix_ptrs,
    layer_input_ptrs,
    residual_ptrs,
    h_blk: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('n')
    mhc = T.const('mhc')
    hidden = T.const('h')
    num_layers = T.const('L')
    num_post_layers = T.const('L_post')

    initial_residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    pre_mix_ptrs: T.Tensor[(num_layers,), T.ptr]
    layer_output_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    post_mix_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    comb_mix_ptrs: T.Tensor[(num_post_layers,), T.ptr]
    layer_input_ptrs: T.Tensor[(num_layers,), T.ptr]
    residual_ptrs: T.Tensor[(num_post_layers,), T.ptr]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    mhc_pad = T.ceildiv(mhc, 8) * 8  # TODO remove manual padding

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((2, mhc, h_blk), T.bfloat16)
        layer_output_ub = T.alloc_shared((2, h_blk), T.bfloat16)
        layer_input_ub = T.alloc_shared((2, h_blk), T.bfloat16)
        pre_mix_ub = T.alloc_shared((2, mhc_pad), T.float32)
        post_mix_ub = T.alloc_shared((2, mhc_pad), T.float32)
        comb_mix_ub = T.alloc_shared((2, mhc, mhc), T.float32)

        T.ascend_set_flag('MTE3_MTE2', 0)

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=0):
            T.ascend_set_flag('V_MTE2', 0)
            T.ascend_set_flag('V_MTE2', 1)
            T.ascend_set_flag('MTE3_V', 0)
            T.ascend_set_flag('MTE3_V', 1)

            T.ascend_wait_flag('MTE3_MTE2', 0)
            T.copy(initial_residual[pid_n, 0, pid_h * h_blk], residual_ub[0, :, :])
            T.ascend_set_flag('MTE2_V', 2)
            T.ascend_wait_flag('MTE2_V', 2)

            for l in T.serial(num_post_layers):
                input_slot = l % 2
                output_slot = 1 - input_slot
                pre_mix_tensor = T.make_tensor(pre_mix_ptrs[l], (num_tokens, mhc), T.float32)
                post_mix_tensor = T.make_tensor(post_mix_ptrs[l], (num_tokens, mhc), T.float32)
                comb_mix_tensor = T.make_tensor(comb_mix_ptrs[l], (num_tokens, mhc, mhc), T.float32)
                layer_output_tensor = T.make_tensor(layer_output_ptrs[l], (num_tokens, hidden), T.bfloat16)
                layer_input_tensor = T.make_tensor(layer_input_ptrs[l], (num_tokens, hidden), T.bfloat16)
                residual_tensor = T.make_tensor(residual_ptrs[l], (num_tokens, mhc, hidden), T.bfloat16)

                T.ascend_wait_flag('V_MTE2', input_slot)
                T.copy(pre_mix_tensor[pid_n, 0], pre_mix_ub[input_slot, 0:mhc])
                T.copy(post_mix_tensor[pid_n, 0], post_mix_ub[input_slot, 0:mhc])
                T.copy(comb_mix_tensor[pid_n, 0, 0], comb_mix_ub[input_slot, :, :])
                T.copy(layer_output_tensor[pid_n, pid_h * h_blk], layer_output_ub[input_slot, :])
                T.ascend_set_flag('MTE2_V', input_slot)

                T.ascend_wait_flag('MTE2_V', input_slot)
                T.ascend_wait_flag('MTE3_V', output_slot)
                with T.SimdVF():
                    mask = V.create_mask(_VEC, size=_VEC)
                    f32 = V.vreg(_VEC, T.float32)
                    pre_mix = V.alloc_local((mhc,), f32)
                    post_mix = V.alloc_local((mhc,), f32)
                    comb_mix = V.alloc_local((mhc * mhc,), f32)
                    for mi in T.unroll(mhc, explicit=True):
                        pre_mix[mi] = V.vbrc(V.vload(pre_mix_ub[input_slot, mi], size=1), size=_VEC)
                        post_mix[mi] = V.vbrc(V.vload(post_mix_ub[input_slot, mi], size=1), size=_VEC)
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            comb_mix[mi * mhc + mo] = V.vbrc(V.vload(comb_mix_ub[input_slot, mi, mo], size=1), size=_VEC)
                    layer_input = V.alloc_local((1,), f32)
                    residual = V.alloc_local((mhc,), f32)
                    output = V.alloc_local((mhc,), f32)
                    for chunk_index in T.serial(num_chunks):
                        column_offset = chunk_index * _VEC
                        for mi in T.unroll(mhc, explicit=True):
                            residual[mi] = V.vcvt(V.vload(residual_ub[input_slot, mi, column_offset], size=_VEC), 'float32')
                        layer_input[0] = V.vmul(residual[0], pre_mix[0])
                        for mi in T.unroll(mhc - 1, explicit=True):
                            layer_input[0] = V.vmula(layer_input[0], residual[mi + 1], pre_mix[mi + 1], mask)
                        V.vstore(V.vcvt(layer_input[0], 'bfloat16'), layer_input_ub[input_slot, column_offset])
                        layer_output = V.vcvt(V.vload(layer_output_ub[input_slot, column_offset], size=_VEC), 'float32')
                        for mo in T.unroll(mhc, explicit=True):
                            output[mo] = V.vmul(layer_output, post_mix[mo])
                        for mi in T.unroll(mhc, explicit=True):
                            for mo in T.unroll(mhc, explicit=True):
                                output[mo] = V.vmula(output[mo], residual[mi], comb_mix[mi * mhc + mo], mask)
                        for mo in T.unroll(mhc, explicit=True):
                            V.vstore(V.vcvt(output[mo], 'bfloat16'), residual_ub[output_slot, mo, column_offset])
                T.ascend_set_flag('V_MTE2', input_slot)
                T.ascend_set_flag('V_MTE3', input_slot)

                T.ascend_wait_flag('V_MTE3', input_slot)
                T.copy(residual_ub[output_slot, :, :], residual_tensor[pid_n, 0, pid_h * h_blk])
                T.copy(layer_input_ub[input_slot, :], layer_input_tensor[pid_n, pid_h * h_blk])
                T.ascend_set_flag('MTE3_V', output_slot)

            T.ascend_set_flag('MTE3_MTE2', 0)

            if num_layers > num_post_layers:
                final_slot = num_post_layers % 2
                final_output_slot = 1 - final_slot
                pre_mix_tensor = T.make_tensor(pre_mix_ptrs[num_post_layers], (num_tokens, mhc), T.float32)
                layer_input_tensor = T.make_tensor(layer_input_ptrs[num_post_layers], (num_tokens, hidden), T.bfloat16)
                T.ascend_wait_flag('V_MTE2', final_slot)
                T.copy(pre_mix_tensor[pid_n, 0], pre_mix_ub[final_slot, 0:mhc])
                T.ascend_set_flag('MTE2_V', final_slot)
                T.ascend_wait_flag('MTE2_V', final_slot)
                T.ascend_wait_flag('MTE3_V', final_output_slot)
                with T.SimdVF():
                    mask = V.create_mask(_VEC, size=_VEC)
                    f32 = V.vreg(_VEC, T.float32)
                    pre_mix = V.alloc_local((mhc,), f32)
                    for mi in T.unroll(mhc, explicit=True):
                        pre_mix[mi] = V.vbrc(V.vload(pre_mix_ub[final_slot, mi], size=1), size=_VEC)
                    layer_input = V.alloc_local((1,), f32)
                    residual = V.alloc_local((mhc,), f32)
                    for chunk_index in T.serial(num_chunks):
                        column_offset = chunk_index * _VEC
                        for mi in T.unroll(mhc, explicit=True):
                            residual[mi] = V.vcvt(V.vload(residual_ub[final_slot, mi, column_offset], size=_VEC), 'float32')
                        layer_input[0] = V.vmul(residual[0], pre_mix[0])
                        for mi in T.unroll(mhc - 1, explicit=True):
                            layer_input[0] = V.vmula(layer_input[0], residual[mi + 1], pre_mix[mi + 1], mask)
                        V.vstore(V.vcvt(layer_input[0], 'bfloat16'), layer_input_ub[final_slot, column_offset])
                T.ascend_set_flag('V_MTE2', final_slot)
                T.ascend_set_flag('V_MTE3', final_slot)
                T.ascend_wait_flag('V_MTE3', final_slot)
                T.copy(layer_input_ub[final_slot, :], layer_input_tensor[pid_n, pid_h * h_blk])
                T.ascend_set_flag('MTE3_V', final_output_slot)

            T.ascend_wait_flag('V_MTE2', 0)
            T.ascend_wait_flag('V_MTE2', 1)
            T.ascend_wait_flag('MTE3_V', 0)
            T.ascend_wait_flag('MTE3_V', 1)

        T.ascend_wait_flag('MTE3_MTE2', 0)
