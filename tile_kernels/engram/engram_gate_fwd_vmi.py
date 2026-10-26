import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def get_engram_gate_fwd_kernel_vmi(
    hidden_size: int,
    eps: float,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    save_for_backward: bool,
    has_image_token_mask: bool,
    inplace: bool,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 2
    heads_per_task = 2
    num_stages = 2

    assert hc_mult == 4
    assert hidden_size % (vec_size * num_accs) == 0
    assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'

    num_tokens = T.dynamic('num_tokens')
    num_head_groups = hc_mult // heads_per_task
    assert num_vec_cores % num_head_groups == 0
    num_chunks = hidden_size // vec_size

    @T.prim_func
    def engram_gate_fwd_kernel_vmi(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float32],
    ):
        output_target = hidden_states if inplace else output
        with T.Kernel(num_vec_cores) as core_id:
            head_group = core_id % num_head_groups
            head_base = head_group * heads_per_task

            weight_ub = T.alloc_shared((heads_per_task, hidden_size), T.float32)
            x_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            k_output_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            v_ub = T.alloc_shared((hidden_size,), T.bfloat16)

            dot_ub = T.alloc_shared((8,), T.float32)
            gate_ub = T.alloc_shared((8,), T.float32)
            rstd_x_ub = T.alloc_shared((8,), T.float32)
            rstd_k_ub = T.alloc_shared((8,), T.float32)
            buffer_versions = {
                x_ub: num_stages,
                k_output_ub: num_stages,
                v_ub: num_stages,
                gate_ub: num_stages,
            }
            if save_for_backward:
                buffer_versions.update(
                    {
                        dot_ub: num_stages,
                        rstd_x_ub: num_stages,
                        rstd_k_ub: num_stages,
                    }
                )
            T.annotate_buffer_versions(buffer_versions)

            T.copy(weight_fused[head_base : head_base + heads_per_task, :], weight_ub)

            for token_id, task_head_group in T.Persistent(
                [num_tokens, num_head_groups],
                num_vec_cores,
                core_id,
                group_size=num_head_groups,
                num_stages=num_stages,
            ):
                task_head_base = task_head_group * heads_per_task
                if has_image_token_mask and image_token_mask[token_id]:
                    if not inplace:
                        T.copy(
                            hidden_states[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                                :,
                            ],
                            x_ub,
                            l2_cache_ctrl='NOTALLOC_KEEP',
                        )
                        T.copy(
                            x_ub,
                            output_target[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                                :,
                            ],
                        )
                else:
                    T.copy(
                        hidden_states[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                        x_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    T.copy(
                        kv[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                        k_output_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    with T.SimdVF():
                        mask = V.create_mask(vec_size, size=vec_size)
                        mask1 = V.create_mask(1, size=1)
                        f32 = V.vreg(vec_size, T.float32)
                        zero = V.vbrc(T.float32(0.0), size=vec_size)
                        one1 = V.vbrc(T.float32(1.0), size=1)
                        accum = V.alloc_local((3 * num_accs,), f32)

                        for head_id in T.unroll(heads_per_task, explicit=True):
                            for acc_id in T.unroll(3 * num_accs, explicit=True):
                                accum[acc_id] = zero

                            for chunk_group in T.serial(num_chunks // num_accs):
                                for acc_id in T.unroll(num_accs, explicit=True):
                                    offset = (chunk_group * num_accs + acc_id) * vec_size
                                    x_f32 = V.vcvt(V.vload(x_ub[head_id, offset], size=vec_size), 'float32')
                                    k_f32 = V.vcvt(
                                        V.vload(
                                            k_output_ub[
                                                head_id,
                                                offset
                                            ],
                                            size=vec_size
                                        ),
                                        'float32'
                                    )
                                    weight_f32 = V.vload(weight_ub[head_id, offset], size=vec_size)

                                    accum[acc_id] = V.vmula(accum[acc_id], x_f32, x_f32, mask)
                                    accum[num_accs + acc_id] = V.vmula(accum[num_accs + acc_id], k_f32, k_f32, mask)
                                    weighted_x = V.vmul(x_f32, weight_f32)
                                    accum[2 * num_accs + acc_id] = V.vmula(accum[2 * num_accs + acc_id], weighted_x, k_f32, mask)

                            for value_id in T.unroll(3, explicit=True):
                                base = value_id * num_accs
                                accum[base] = V.vadd(accum[base], accum[base + 1])

                            sum_x2 = V.vcadd(accum[0], mask, reassoc=True)
                            sum_k2 = V.vcadd(accum[num_accs], mask, reassoc=True)
                            raw_dot = V.vcadd(accum[2 * num_accs], mask, reassoc=True)
                            rms_x = V.vsqrt(V.vadds(V.vmuls(sum_x2, T.float32(1.0 / hidden_size), mask1), T.float32(eps), mask1))
                            rms_k = V.vsqrt(V.vadds(V.vmuls(sum_k2, T.float32(1.0 / hidden_size), mask1), T.float32(eps), mask1))
                            inv_rms_x = V.vdiv(one1, rms_x, mask1, precision='exact')
                            inv_rms_k = V.vdiv(one1, rms_k, mask1, precision='exact')
                            normalized_dot = V.vmuls(V.vmul(V.vmul(raw_dot, inv_rms_x), inv_rms_k), T.float32(scalar), mask1)
                            signed_root_abs = V.vsqrt(V.vmaxs(V.vabs(normalized_dot), T.float32(clamp_value), mask1))
                            signed_root = V.vsel(
                                V.vcmps(normalized_dot, T.float32(0.0), mask1, 'ge'),
                                signed_root_abs,
                                V.vneg(signed_root_abs),
                            )
                            gate = V.vdiv(one1, V.vadd(V.vexpdif(V.vbrc(T.float32(0.0), size=1), signed_root, mask1), one1), mask1, precision='exact')

                            V.vstore(gate, gate_ub[head_id])
                            if save_for_backward:
                                V.vstore(raw_dot, dot_ub[head_id])
                                V.vstore(inv_rms_x, rstd_x_ub[head_id])
                                V.vstore(inv_rms_k, rstd_k_ub[head_id])

                    T.copy(
                        kv[
                            token_id,
                            hc_mult,
                            :,
                        ],
                        v_ub,
                    )

                    with T.SimdVF():
                        mask2 = V.create_mask(vec_size, size=vec_size)
                        f32 = V.vreg(vec_size, T.float32)
                        gates_f32 = V.alloc_local((heads_per_task,), f32)
                        for head_id in T.unroll(heads_per_task, explicit=True):
                            gates_f32[head_id] = V.vbrc(V.vload(gate_ub[head_id], size=1), size=vec_size)

                        out_f32 = V.alloc_local((heads_per_task,), f32)
                        for chunk_id in T.serial(num_chunks):
                            offset = chunk_id * vec_size
                            v_f32 = V.vcvt(V.vload(v_ub[offset], size=vec_size), 'float32')
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                out_f32[head_id] = V.vcvt(
                                    V.vload(x_ub[head_id, offset], size=vec_size),
                                    'float32'
                                )
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                out_f32[head_id] = V.vmula(out_f32[head_id], v_f32, gates_f32[head_id], mask2)
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                V.vstore(
                                    V.vcvt(out_f32[head_id], 'bfloat16'),
                                    k_output_ub[
                                        head_id,
                                        offset,
                                    ],
                                )

                    T.copy(
                        k_output_ub,
                        output_target[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                    )
                    if save_for_backward:
                        T.copy(
                            dot_ub[:heads_per_task],
                            dot_out[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            gate_ub[:heads_per_task],
                            gate_score[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            rstd_x_ub[:heads_per_task],
                            rstd_x[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            rstd_k_ub[:heads_per_task],
                            rstd_k[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )

    return engram_gate_fwd_kernel_vmi
