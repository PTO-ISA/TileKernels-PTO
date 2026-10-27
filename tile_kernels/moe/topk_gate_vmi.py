import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V

from tile_kernels.utils import align, ceil_div


@tilelang.jit(target="pto")
def get_topk_gate_kernel_vmi(num_experts: int, num_topk: int, num_vec_cores: int):
    num_stages = 6
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)

    # Adaptive: use only as many vregs as needed when num_experts <= 256.
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group

    # Round up to a multiple of vector_length for UB alignment of the index output.
    num_aligned_topk = align(num_topk, vector_length)

    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def topk_gate_kernel_vmi(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            out_ub = T.alloc_shared((num_aligned_topk,), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            T.annotate_buffer_versions({scores_ub: num_stages, out_ub: num_stages, invalid_logits_ub: 1})

            with T.SimdVF():
                V.vstore(V.vbrc(T.int32(0), size=vector_length), invalid_logits_ub[0])

            for w in T.Pipelined(
                T.ceildiv(num_tokens, num_vec_cores),
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [scores_ub, out_ub]},
            ):
                row = w * num_vec_cores + core_id
                if row < num_tokens:
                    T.copy(scores[row, :], scores_ub[:num_experts])
                    with T.SimdVF():
                        full = V.create_mask(vector_length, size=vector_length)
                        neg_inf = V.vbrc(-T.infinity(T.float32), size=vector_length)
                        int_max = V.vbrc(T.max_value(T.int32), size=vector_length)
                        zero_1 = V.vbrc(T.int32(0), size=1)
                        expert_limit = V.vbrc(T.int32(num_experts), size=vector_length)

                        # Allocate vector register arrays for 6 vregs per group.
                        scores_vec = V.alloc_local(
                            (num_vregs_per_group,), T.vmi.vreg(vector_length, T.float32)
                        )
                        index_vec = V.alloc_local(
                            (num_vregs_per_group,), T.vmi.vreg(vector_length, T.int32)
                        )
                        tmp_idx_vec = V.alloc_local(
                            (num_vregs_per_group,), T.vmi.vreg(vector_length, T.int32)
                        )

                        acc_max = V.alloc_var(T.float32, size=vector_length)
                        acc_min_idx = V.alloc_var(T.int32, size=vector_length)

                        zero = V.vbrc(T.float32(0.0), size=vector_length)
                        zeros = V.vbrc(T.int32(0), size=vector_length)
                        ones = V.vbrc(T.int32(1), size=vector_length)
                        invalid = V.alloc_var(T.int32, size=vector_length)
                        invalid = V.vload(invalid_logits_ub[0], size=vector_length)
                        for j in T.unroll(ceil_div(num_experts, vector_length), explicit=True):
                            valid = V.vcmps(
                                V.vci(j * vector_length, size=vector_length),
                                T.int32(num_experts),
                                full,
                                'lt',
                            )
                            logit = V.vsel(valid, V.vload(scores_ub[j * vector_length], size=vector_length), zero)
                            nonfinite = V.vcmps(V.vsub(logit, logit), T.float32(0.0), full, 'ne')
                            invalid = V.vmax(invalid, V.vsel(nonfinite, ones, zeros), full)
                        V.vstore(V.vcmax(invalid, full), invalid_logits_ub[0], stride=1, group=1)

                        if num_groups == 1:
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = V.vload(scores_ub[i * vector_length], size=vector_length)
                                index_vec[i] = V.vci(T.int32(i * vector_length), size=vector_length)

                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = V.vsel(V.vcmp(index_vec[i], expert_limit, full, 'lt'), scores_vec[i], neg_inf)

                            for k in T.serial(num_topk):
                                # Get max value.
                                acc_max = scores_vec[0]
                                for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                    acc_max = V.vmax(acc_max, scores_vec[i], full)
                                mx = V.vbrc(V.vcmax(acc_max, full), size=vector_length)

                                # Get minimal idx for all values equal to max.
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    tmp_idx_vec[i] = V.vsel(V.vcmp(scores_vec[i], mx, full, 'eq'), index_vec[i], int_max)
                                acc_min_idx = tmp_idx_vec[0]
                                for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                    acc_min_idx = V.vmin(acc_min_idx, tmp_idx_vec[i], full)
                                winner_1 = V.vcmin(acc_min_idx, full)
                                idx = V.vbrc(winner_1, size=vector_length)
                                V.vstore(winner_1, out_ub[2 * k], stride=1, group=1)
                                V.vstore(zero_1, out_ub[2 * k + 1], stride=1, group=1)

                                # Mask out the winner lane in each vreg so the next k picks the next best.
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    scores_vec[i] = V.vsel(V.vcmp(index_vec[i], idx, full, 'eq'), neg_inf, scores_vec[i])
                        else:
                            # Slow path: multi-group.
                            for k in T.serial(num_topk):
                                # Phase 1: per-group local max -> accumulated global max.
                                acc_max = neg_inf
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * vector_length
                                        index_vec[i] = V.vci(T.int32(offset), size=vector_length)
                                        scores_vec[i] = V.vsel(
                                            V.vcmp(index_vec[i], expert_limit, full, 'lt'),
                                            V.vload(scores_ub[offset], size=vector_length),
                                            neg_inf,
                                        )
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        acc_max = V.vmax(acc_max, scores_vec[i], full)
                                mx = V.vbrc(V.vcmax(acc_max, full), size=vector_length)

                                # Phase 2: per-group winner index -> accumulated global index.
                                acc_min_idx = int_max
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * vector_length
                                        index_vec[i] = V.vci(T.int32(offset), size=vector_length)
                                        scores_vec[i] = V.vsel(
                                            V.vcmp(index_vec[i], expert_limit, full, 'lt'),
                                            V.vload(scores_ub[offset], size=vector_length),
                                            neg_inf,
                                        )
                                        tmp_idx_vec[i] = V.vsel(
                                            V.vcmp(scores_vec[i], mx, full, 'eq'), index_vec[i], int_max
                                        )
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        acc_min_idx = V.vmin(acc_min_idx, tmp_idx_vec[i], full)
                                winner_1 = V.vcmin(acc_min_idx, full)
                                idx = V.vbrc(winner_1, size=vector_length)
                                V.vstore(winner_1, out_ub[2 * k], stride=1, group=1)
                                V.vstore(zero_1, out_ub[2 * k + 1], stride=1, group=1)

                                # Phase 3: mask the winner's lane to -inf in UB.
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * vector_length
                                        index_vec[i] = V.vci(T.int32(offset), size=vector_length)
                                        scores_vec[i] = V.vsel(
                                            V.vcmp(index_vec[i], idx, full, 'eq'),
                                            neg_inf,
                                            V.vload(scores_ub[offset], size=vector_length),
                                        )
                                        V.vstore(scores_vec[i], scores_ub[offset], stride=1, group=1)
                                S.mem_bar('VST_VLD')
                    T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])

            T.device_assert(invalid_logits_ub[0] == 0, msg='topk_gate input contains nan or inf!')

    return topk_gate_kernel_vmi
