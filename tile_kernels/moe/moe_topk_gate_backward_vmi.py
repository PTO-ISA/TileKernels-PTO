import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V

from tile_kernels.config import get_num_vec_cores
from tile_kernels.moe.sqrt_softplus import sqrt_softplus_backward_vmi, sqrt_softplus_backward_vmi_dense
from tile_kernels.utils import align


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True})
def get_moe_topk_gate_backward_kernel_vmi(num_topk: int, num_routed_experts: int, aux_exists: bool):
    num_stages = 2
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)
    max_num_topk = 32  # Keep the same top-k limit as forward.

    # Give each token a power-of-two lane group so XOR reductions stay within that token.
    reduction_steps = (num_topk - 1).bit_length()
    topk_width = 1 << reduction_steps
    rows_per_vec = vector_length // topk_width
    # Temporary PTO ABI: use int32 indices until T.view(int64 GM, int32)
    # lowering is fixed. Restore the public int64 input together with the ASC
    # int32-word view after that fix is available.
    idx_stride = align(num_topk, 8)
    weight_stride = align(num_topk, 8)
    num_vec_cores = get_num_vec_cores()
    assert num_topk <= max_num_topk
    num_aligned_experts = align(num_routed_experts, vector_length)
    num_vregs = num_aligned_experts // vector_length
    experts_aligned = num_routed_experts == num_aligned_experts
    if num_aligned_experts <= 256 or (not aux_exists and num_aligned_experts <= 384):
        block_m = 32
    else:
        block_m = 16

    num_tokens = T.dynamic('num_tokens')
    num_seqs = T.dynamic('num_seqs')

    @T.prim_func
    def moe_topk_gate_backward_kernel_vmi(
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int32],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        grad_topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        mask: T.Tensor[(num_tokens,), T.bool],
        grad_scores_sum: T.Tensor[(num_seqs, num_routed_experts), T.float32],
        grad_logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            grad_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            idx_ub = T.alloc_shared((block_m, idx_stride), T.int32)
            weight_ub = T.alloc_shared((block_m, weight_stride), T.float32)
            grad_weight_ub = T.alloc_shared((block_m, weight_stride), T.float32)

            versions = {
                scores_ub: num_stages,
                grad_ub: num_stages,
                idx_ub: num_stages,
                weight_ub: num_stages,
                grad_weight_ub: num_stages,
            }
            if aux_exists:
                aux_grad_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
                mask_ub = T.alloc_shared((block_m,), T.bool)
                versions.update({aux_grad_ub: num_stages, mask_ub: num_stages})
            T.annotate_buffer_versions(versions)

            num_rows = T.min(block_m, T.max(1, T.ceildiv(num_tokens, num_vec_cores)))
            for tile_id in T.Persistent(
                [T.ceildiv(num_tokens, num_rows)],
                num_vec_cores,
                core_id,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [grad_ub]},
            ):
                row_start = tile_id * num_rows
                tile_rows = T.min(num_rows, num_tokens - row_start)
                T.copy(scores[row_start : row_start + num_rows, :], scores_ub[:num_rows, :num_routed_experts])
                T.copy(topk_idx[row_start : row_start + num_rows, :], idx_ub[:num_rows, :num_topk])
                T.copy(topk_weights[row_start : row_start + num_rows, :], weight_ub[:num_rows, :num_topk])
                T.copy(grad_topk_weights[row_start : row_start + num_rows, :], grad_weight_ub[:num_rows, :num_topk])
                if aux_exists:
                    tokens_per_seq = T.truncdiv(num_tokens, num_seqs)
                    seq_first = T.truncdiv(row_start, tokens_per_seq)
                    seq_span = T.truncdiv(row_start + tile_rows - 1, tokens_per_seq) - seq_first + 1
                    T.copy(mask[row_start : row_start + num_rows], mask_ub[:num_rows])
                    T.copy(grad_scores_sum[seq_first : seq_first + seq_span, :], aux_grad_ub[:seq_span, :num_routed_experts])

                with T.SimdVF():
                    full = V.create_mask(vector_length, size=vector_length)
                    lanes = V.vci(T.int32(0), size=vector_length)
                    zero = V.vbrc(T.float32(0.0), size=vector_length)
                    one = V.vbrc(T.float32(1.0), size=vector_length)
                    local_row = V.vshr(lanes, V.vbrc(T.int32(reduction_steps), size=vector_length), full)
                    local_k = V.vand(lanes, V.vbrc(T.int32(topk_width - 1), size=vector_length), full)
                    idx_offsets = V.vinterpret_cast(V.vadd(V.vmul(local_row, V.vbrc(T.int32(idx_stride), size=vector_length), full), local_k, full), 'uint32')
                    weight_offsets = V.vinterpret_cast(V.vadd(V.vmul(local_row, V.vbrc(T.int32(weight_stride), size=vector_length), full), local_k, full), 'uint32')
                    score_row_offsets = V.vmul(local_row, V.vbrc(T.int32(num_aligned_experts), size=vector_length), full)
                    for j in T.serial(tile_rows * num_vregs):
                        V.vstore(zero, grad_ub[0, j * vector_length])
                    S.mem_bar('VST_VST')

                    for row_base in T.serial(0, tile_rows, rows_per_vec):
                        valid_rows = V.vcmps(local_row, T.int32(tile_rows - row_base), full, 'lt')
                        topk_lanes = V.vcmps(local_k, T.int32(num_topk), valid_rows, 'lt')
                        idx = V.vgather(idx_ub[row_base, 0], idx_offsets, topk_lanes)
                        valid_topk = V.vcmps(idx, T.int32(0), topk_lanes, 'ge')
                        score_row_offsets_u = V.vinterpret_cast(score_row_offsets, 'uint32')
                        score_offsets = V.vadd(score_row_offsets_u, V.vinterpret_cast(idx, 'uint32'), full)
                        chosen_scores = V.vgather(scores_ub[row_base, 0], score_offsets, valid_topk)
                        weights = V.vgather(weight_ub[row_base, 0], weight_offsets, valid_topk)
                        grad_weights = V.vgather(grad_weight_ub[row_base, 0], weight_offsets, valid_topk)
                        denom = V.alloc_var(T.float32, size=vector_length)
                        denom = chosen_scores
                        dot = V.alloc_var(T.float32, size=vector_length)
                        dot = V.vmul(weights, grad_weights, valid_topk)
                        for step in T.unroll(reduction_steps, explicit=True):
                            partner = V.vxor(lanes, V.vbrc(T.int32(1 << (reduction_steps - 1 - step)), size=vector_length), full)
                            denom = V.vadd(denom, V.vselr(denom, partner), full)
                            dot = V.vadd(dot, V.vselr(dot, partner), full)
                        # For w_i = scale*s_i/sum(s), dL/ds_i = (scale*g_i - sum(w*g))/sum(s).
                        numerator = V.vsub(V.vmuls(grad_weights, T.float32(routed_scaling_factor), valid_topk), dot, valid_topk)
                        denominator = V.vadds(denom, T.float32(1e-20), full)
                        delta = V.vdiv(numerator, denominator, valid_topk, precision='ftz_true')

                        # Fixed routing may repeat experts; sum their gradients before a single scatter write.
                        combined = V.alloc_var(T.float32, size=vector_length)
                        combined = zero
                        last_lane = V.alloc_var(T.int32, size=vector_length)
                        last_lane = V.vbrc(T.int32(-1), size=vector_length)
                        for k in T.unroll(num_topk, explicit=True):
                            selector = V.vadd(V.vmul(local_row, V.vbrc(T.int32(topk_width), size=vector_length), full), V.vbrc(T.int32(k), size=vector_length), full)
                            same = V.vcmp(idx, V.vselr(idx, selector), valid_topk, 'eq')
                            combined = V.vadd(combined, V.vsel(same, V.vselr(delta, selector), zero), full)
                            last_lane = V.vsel(same, V.vbrc(T.int32(k), size=vector_length), last_lane)
                        unique = V.vcmp(local_k, last_lane, valid_topk, 'eq')
                        if not aux_exists:
                            combined = V.vmul(combined, sqrt_softplus_backward_vmi(chosen_scores, valid_topk, vector_length), unique)
                        V.vscatter(combined, grad_ub[row_base, 0], score_offsets, unique)
                    S.mem_bar('VST_VLD')

                    # Add the dense auxiliary gradient to the routed gradient before applying the activation derivative.
                    if aux_exists:
                        hrow = T.alloc_var(T.int32, init=0)
                        boundary = T.alloc_var(T.int32, init=(seq_first + 1) * tokens_per_seq - row_start)
                        for row in T.serial(tile_rows):
                            if row >= boundary:
                                hrow += 1
                                boundary += tokens_per_seq
                            score_sum = V.alloc_var(T.float32, size=vector_length)
                            score_sum = zero
                            aux_dot = V.alloc_var(T.float32, size=vector_length)
                            aux_dot = zero
                            for j in T.serial(num_vregs):
                                score = V.vload(scores_ub[row, j * vector_length], size=vector_length)
                                aux_grad = V.vload(aux_grad_ub[hrow, j * vector_length], size=vector_length)
                                if not experts_aligned:
                                    valid = V.vcmp(V.vadd(lanes, V.vbrc(T.int32(j * vector_length), size=vector_length), full), V.vbrc(T.int32(num_routed_experts), size=vector_length), full, 'lt')
                                    score = V.vsel(valid, score, zero)
                                    aux_grad = V.vsel(valid, aux_grad, zero)
                                score_sum = V.vadd(score_sum, score, full)
                                aux_dot = V.vadd(aux_dot, V.vmul(aux_grad, score, full), full)
                            # PTOAS currently supports only reassociated VMI reductions.
                            score_sum_scalar = V.vcadd(score_sum, full, reassoc=True)
                            aux_dot_scalar = V.vcadd(aux_dot, full, reassoc=True)
                            inv_score_sum = V.vdiv(one, V.vbrc(score_sum_scalar, size=vector_length), full, precision='ftz_true')
                            aux_active = V.vbrc(T.cast(mask_ub[row], T.int32), size=vector_length)
                            aux_active = V.vcmp(aux_active, V.vbrc(T.int32(0), size=vector_length), full, 'ne')
                            inv_score_sum = V.vsel(aux_active, inv_score_sum, zero)
                            aux_bias = V.vneg(V.vmul(V.vmul(V.vbrc(aux_dot_scalar, size=vector_length), inv_score_sum, full), inv_score_sum, full), full)
                            for j in T.serial(num_vregs):
                                score = V.vload(scores_ub[row, j * vector_length], size=vector_length)
                                aux_grad = V.vload(aux_grad_ub[hrow, j * vector_length], size=vector_length)
                                grad_score = V.vadd(V.vload(grad_ub[row, j * vector_length], size=vector_length), V.vadd(V.vmul(aux_grad, inv_score_sum, full), aux_bias, full), full)
                                grad_logit = V.vmul(grad_score, sqrt_softplus_backward_vmi_dense(score, full, vector_length), full)
                                V.vstore(grad_logit, grad_ub[row, j * vector_length])

                T.copy(grad_ub[:num_rows, :num_routed_experts], grad_logits[row_start : row_start + num_rows, :])

    return moe_topk_gate_backward_kernel_vmi
