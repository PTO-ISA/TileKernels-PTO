"""T.Parallel VF bodies; scheduling and DMA copied from multilayer_recompute_asc.py."""

import tilelang
from tilelang.ascend import language as T


_VEC = 64

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
}


@tilelang.jit(target='pto', pass_configs=_PASS_CONFIGS)
def mhc_multilayer_recompute_parallel(
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
                with T.SimdVF(lanes=64):
                    for j in T.Parallel(h_blk):
                        acc = T.alloc_var(T.float32, init=T.float32(residual_ub[input_slot, 0, j]) * pre_mix_ub[input_slot, 0])
                        for mi in T.serial(1, mhc):
                            acc += T.float32(residual_ub[input_slot, mi, j]) * pre_mix_ub[input_slot, mi]
                        layer_input_ub[input_slot, j] = T.cast(acc, T.bfloat16)
                    for mo, j in T.Parallel(mhc, h_blk):
                        acc = T.alloc_var(T.float32, init=T.float32(layer_output_ub[input_slot, j]) * post_mix_ub[input_slot, mo])
                        for mi in T.serial(mhc):
                            acc += T.float32(residual_ub[input_slot, mi, j]) * comb_mix_ub[input_slot, mi, mo]
                        residual_ub[output_slot, mo, j] = T.cast(acc, T.bfloat16)
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
                with T.SimdVF(lanes=64):
                    for j in T.Parallel(h_blk):
                        acc = T.alloc_var(T.float32, init=T.float32(residual_ub[final_slot, 0, j]) * pre_mix_ub[final_slot, 0])
                        for mi in T.serial(1, mhc):
                            acc += T.float32(residual_ub[final_slot, mi, j]) * pre_mix_ub[final_slot, mi]
                        layer_input_ub[final_slot, j] = T.cast(acc, T.bfloat16)
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
