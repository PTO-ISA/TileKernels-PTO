import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def get_engram_gate_bwd_kernel_vmi(
    hidden_size: int,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    has_image_token_mask: bool,
    num_vec_cores: int,
):
    vec_size = 64
    bf16_vec_size = 128
    num_accs = 2
    num_partials = num_vec_cores // hc_mult
    v_slice_size = hidden_size // hc_mult

    assert hc_mult == 4
    assert num_vec_cores % hc_mult == 0
    assert hidden_size % (hc_mult * bf16_vec_size) == 0

    num_chunks = hidden_size // vec_size
    num_chunk_groups = num_chunks // num_accs
    v_slice_chunks = v_slice_size // vec_size
    num_bf16_chunks = hidden_size // bf16_vec_size
    v_slice_bf16_chunks = v_slice_size // bf16_vec_size
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def engram_gate_bwd_kernel_vmi(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        dot: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_partials, hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            core_head = core_id % hc_mult
            go = T.alloc_shared((hidden_size,), T.bfloat16)
            other_go = T.alloc_shared((hc_mult - 1, v_slice_size), T.bfloat16)
            v = T.alloc_shared((hidden_size,), T.bfloat16)
            x = T.alloc_shared((hidden_size,), T.bfloat16)
            k = T.alloc_shared((hidden_size,), T.bfloat16)
            stats = T.alloc_shared((32,), T.float32)
            weight = T.alloc_shared((hidden_size,), T.float32)
            grad_w = T.alloc_shared((hidden_size,), T.float32)

            T.annotate_buffer_versions(
                {
                    go: 2,
                    other_go: 3,
                    v: 3,
                    x: 3,
                    k: 3,
                    stats: 3,
                }
            )
            T.copy(weight_fused[core_head, :], weight)
            with T.SimdVF():
                T.clear(grad_w)

            for token, task_head in T.Persistent(
                [num_tokens, hc_mult],
                num_vec_cores,
                core_id,
                group_size=hc_mult,
                num_stages=6,
            ):
                v_slice_begin = task_head * v_slice_size

                T.copy(grad_out[token, task_head, :], go)
                if has_image_token_mask and image_token_mask[token]:
                    with T.SimdVF():
                        zero_bf16 = V.vbrc(T.bfloat16(0.0), size=bf16_vec_size)
                        for chunk in range(num_bf16_chunks):
                            V.vstore(zero_bf16, x[chunk * bf16_vec_size])
                        for chunk in range(v_slice_bf16_chunks):
                            V.vstore(zero_bf16, v[v_slice_begin + chunk * bf16_vec_size])
                else:
                    with T.Task():
                        for head_delta in T.unroll(1, hc_mult, explicit=True):
                            other_head = (task_head + head_delta) % hc_mult
                            T.copy(
                                grad_out[
                                    token,
                                    other_head,
                                    v_slice_begin : v_slice_begin + v_slice_size,
                                ],
                                other_go[head_delta - 1, :],
                            )
                        T.copy(hidden_states[token, task_head, :], x)
                        T.copy(kv[token, task_head, :], k)
                        T.copy(kv[token, hc_mult, :], v)
                        T.copy(dot[token, :], stats[0:hc_mult])
                        T.copy(gate[token, :], stats[8 : 8 + hc_mult])
                        T.copy(rstd_x[token, :], stats[16 : 16 + hc_mult])
                        T.copy(rstd_k[token, :], stats[24 : 24 + hc_mult])

                    with T.SimdVF():
                        mask = V.create_mask(vec_size, size=vec_size)
                        f32 = V.vreg(vec_size, T.float32)
                        zero = V.vbrc(T.float32(0.0), size=vec_size)
                        dgate_acc = V.alloc_local((num_accs,), f32)
                        for acc_id in T.unroll(num_accs, explicit=True):
                            dgate_acc[acc_id] = zero

                        for chunk_group in range(num_chunk_groups):
                            for acc_id in T.unroll(num_accs, explicit=True):
                                offset = (chunk_group * num_accs + acc_id) * vec_size
                                go_f32 = V.vcvt(V.vload(go[offset], size=vec_size), 'float32')
                                v_f32 = V.vcvt(V.vload(v[offset], size=vec_size), 'float32')
                                dgate_acc[acc_id] = V.vmula(dgate_acc[acc_id], go_f32, v_f32, mask)
                        dgate = V.vbrc(V.vcadd(V.vadd(dgate_acc[0], dgate_acc[1]), mask, reassoc=True), size=vec_size)

                        gate_value = V.vbrc(V.vload(stats[8 + task_head], size=1), size=vec_size)
                        raw_dot = V.vbrc(V.vload(stats[task_head], size=1), size=vec_size)
                        inv_rms_x = V.vbrc(V.vload(stats[16 + task_head], size=1), size=vec_size)
                        inv_rms_k = V.vbrc(V.vload(stats[24 + task_head], size=1), size=vec_size)
                        dot_scale = V.vmuls(V.vmul(inv_rms_x, inv_rms_k), T.float32(scalar), mask)
                        abs_raw_dot = V.vabs(raw_dot)
                        active_gate = V.vcmps(V.vmul(abs_raw_dot, dot_scale), T.float32(clamp_value), mask, 'ge')
                        sqrt_grad = V.vmuls(V.vsqrt(V.vdiv(dot_scale, abs_raw_dot, active_gate, precision='exact')), T.float32(0.5), mask)
                        gate_grad = V.vsub(gate_value, V.vmul(gate_value, gate_value))
                        beta = V.vmul(V.vmul(dgate, gate_grad), sqrt_grad, active_gate)

                        neg_beta_raw_dot_over_h = V.vmuls(V.vmul(beta, raw_dot), T.float32(-1.0 / hidden_size), mask)
                        neg_dot_x = V.vmul(V.vmul(neg_beta_raw_dot_over_h, inv_rms_x), inv_rms_x)
                        neg_dot_k = V.vmul(V.vmul(neg_beta_raw_dot_over_h, inv_rms_k), inv_rms_k)

                        grad_v_acc = V.alloc_local((1,), f32)
                        other_gates = V.alloc_local((hc_mult - 1,), f32)
                        for head_delta in T.unroll(1, hc_mult, explicit=True):
                            other_head = (task_head + head_delta) % hc_mult
                            other_gates[head_delta - 1] = V.vbrc(V.vload(stats[8 + other_head], size=1), size=vec_size)

                        for chunk in range(v_slice_chunks):
                            local_offset = chunk * vec_size
                            v_offset = v_slice_begin + local_offset
                            own_go = V.vcvt(V.vload(go[v_offset], size=vec_size), 'float32')
                            grad_v_acc[0] = V.vmul(own_go, gate_value)
                            for head_delta in T.unroll(1, hc_mult, explicit=True):
                                other_go_f32 = V.vcvt(
                                    V.vload(
                                        other_go[
                                            head_delta - 1,
                                            local_offset,
                                        ],
                                        size=vec_size,
                                    ),
                                    'float32',
                                )
                                grad_v_acc[0] = V.vmula(grad_v_acc[0], other_go_f32, other_gates[head_delta - 1], mask)
                            V.vstore(V.vcvt(grad_v_acc[0], 'bfloat16'), v[v_offset])
                        accum = V.alloc_local((3,), f32)

                        for chunk in range(num_chunks):
                            offset = chunk * vec_size
                            go_f32 = V.vcvt(V.vload(go[offset], size=vec_size), 'float32')
                            x_f32 = V.vcvt(V.vload(x[offset], size=vec_size), 'float32')
                            k_f32 = V.vcvt(V.vload(k[offset], size=vec_size), 'float32')
                            weight_f32 = V.vload(weight[offset], size=vec_size)
                            beta_x = V.vmul(beta, x_f32)
                            beta_k = V.vmul(beta, k_f32)

                            accum[0] = go_f32
                            accum[1] = V.vmul(weight_f32, beta_x)
                            accum[2] = V.vload(grad_w[offset], size=vec_size)
                            accum[0] = V.vmula(accum[0], weight_f32, beta_k, mask)
                            accum[1] = V.vmula(accum[1], neg_dot_k, k_f32, mask)
                            accum[2] = V.vmula(accum[2], beta_x, k_f32, mask)
                            accum[0] = V.vmula(accum[0], neg_dot_x, x_f32, mask)

                            V.vstore(V.vcvt(accum[0], 'bfloat16'), go[offset])
                            V.vstore(V.vcvt(accum[1], 'bfloat16'), x[offset])
                            V.vstore(accum[2], grad_w[offset])

                with T.Task():
                    T.copy(go, grad_x[token, task_head, :])
                    T.copy(x, grad_kv[token, task_head, :])
                    T.copy(
                        v[v_slice_begin : v_slice_begin + v_slice_size],
                        grad_kv[
                            token,
                            hc_mult,
                            v_slice_begin : v_slice_begin + v_slice_size,
                        ],
                    )

            T.ascend_pipe_barrier('PIPE_ALL')
            T.copy(grad_w, grad_w_partial[core_id // hc_mult, core_head, :])

    return engram_gate_bwd_kernel_vmi
