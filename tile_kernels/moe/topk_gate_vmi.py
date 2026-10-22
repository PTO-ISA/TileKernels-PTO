import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.language import vmi as V

from tile_kernels.utils import align, ceil_div


@tilelang.jit(target="pto")
def get_topk_gate_kernel_vmi(num_experts: int, num_topk: int, num_vec_cores: int):
    logical_lanes = 64
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * logical_lanes), logical_lanes)
    experts_per_group = num_vregs_per_group * logical_lanes
    num_groups = ceil_div(num_experts, experts_per_group)
    aligned_experts = num_groups * experts_per_group
    aligned_topk_words = align(num_topk * 2, logical_lanes)
    num_tokens = T.dynamic("num_tokens")
    num_stages = 6

    @T.prim_func
    def kernel(
        scores: T.Tensor((num_tokens, num_experts), T.float32),
        topk_idx: T.Tensor((num_tokens, num_topk), T.int64),
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((aligned_experts,), T.float32)
            if num_groups > 1:
                indices_ub = T.alloc_shared((aligned_experts,), T.int32)
            out_ub = T.alloc_shared((aligned_topk_words,), T.int32)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            if num_groups == 1:
                T.annotate_buffer_versions({scores_ub: num_stages, out_ub: num_stages})
            else:
                T.annotate_buffer_versions({scores_ub: num_stages, indices_ub: 1, out_ub: num_stages})

            if num_groups > 1:
                with T.SimdVF():
                    for j in T.unroll(ceil_div(aligned_experts, logical_lanes), explicit=True):
                        index_block = V.vci(T.int32(j * logical_lanes), size=logical_lanes)
                        V.vstore(index_block, indices_ub[j * logical_lanes], stride=1, group=1)
                    S.mem_bar("VST_VLD")

            for w in T.Pipelined(
                T.ceildiv(num_tokens, num_vec_cores),
                num_stages=num_stages,
                annotations={"multi_buffer_eligible": [scores_ub, out_ub]},
            ):
                row = w * num_vec_cores + core_id
                if row < num_tokens:
                    T.copy(scores[row, :num_experts], scores_ub[:num_experts])
                    with T.SimdVF():
                        full = V.create_mask(logical_lanes, size=logical_lanes)
                        neg_inf = V.vbrc(-T.infinity(T.float32), size=logical_lanes)
                        int_max = V.vbrc(T.max_value(T.int32), size=logical_lanes)
                        zero_1 = V.vbrc(T.int32(0), size=1)
                        has_padding = num_experts != aligned_experts
                        if has_padding or num_groups > 1:
                            expert_limit = V.vbrc(T.int32(num_experts), size=logical_lanes)
                        maximum = V.alloc_var(T.float32, size=logical_lanes)
                        winner_vec = V.alloc_var(T.int32, size=logical_lanes)
                        if num_groups == 1:
                            scores_vec = V.alloc_local((num_vregs_per_group,), T.vmi.vreg(logical_lanes, T.float32))
                            index_vec = V.alloc_local((num_vregs_per_group,), T.vmi.vreg(logical_lanes, T.int32))
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = V.vload(scores_ub[i * logical_lanes], size=logical_lanes)
                                index_vec[i] = V.vci(T.int32(i * logical_lanes), size=logical_lanes)
                                if has_padding:
                                    valid = V.vcmp(index_vec[i], expert_limit, full, "lt")
                                    scores_vec[i] = V.vsel(valid, scores_vec[i], neg_inf)
                            for k in T.serial(num_topk):
                                maximum = scores_vec[0]
                                for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                    maximum = V.vmax(maximum, scores_vec[i], full)
                                maximum = V.vbrc(V.vcmax(maximum, full), size=logical_lanes)
                                is_max = V.vcmp(scores_vec[0], maximum, full, "eq")
                                winner_vec = V.vsel(is_max, index_vec[0], int_max)
                                for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                    is_max = V.vcmp(scores_vec[i], maximum, full, "eq")
                                    candidates = V.vsel(is_max, index_vec[i], int_max)
                                    winner_vec = V.vmin(candidates, winner_vec, full)
                                winner_1 = V.vcmin(winner_vec, full)
                                winner = V.vbrc(winner_1, size=logical_lanes)
                                V.vstore(winner_1, out_ub[2 * k], stride=1, group=1)
                                V.vstore(zero_1, out_ub[2 * k + 1], stride=1, group=1)
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    is_winner = V.vcmp(index_vec[i], winner, full, "eq")
                                    scores_vec[i] = V.vsel(is_winner, neg_inf, scores_vec[i])
                        else:
                            for k in T.serial(num_topk):
                                maximum = neg_inf
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * logical_lanes
                                        indices = V.vload(indices_ub[offset], size=logical_lanes)
                                        values = V.vload(scores_ub[offset], size=logical_lanes)
                                        valid = V.vcmp(indices, expert_limit, full, "lt")
                                        values = V.vsel(valid, values, neg_inf)
                                        maximum = V.vmax(maximum, values, full)
                                maximum = V.vbrc(V.vcmax(maximum, full), size=logical_lanes)
                                winner_vec = int_max
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * logical_lanes
                                        indices = V.vload(indices_ub[offset], size=logical_lanes)
                                        values = V.vload(scores_ub[offset], size=logical_lanes)
                                        valid = V.vcmp(indices, expert_limit, full, "lt")
                                        values = V.vsel(valid, values, neg_inf)
                                        is_max = V.vcmp(values, maximum, full, "eq")
                                        candidates = V.vsel(is_max, indices, int_max)
                                        winner_vec = V.vmin(candidates, winner_vec, full)
                                winner_1 = V.vcmin(winner_vec, full)
                                winner = V.vbrc(winner_1, size=logical_lanes)
                                V.vstore(winner_1, out_ub[2 * k], stride=1, group=1)
                                V.vstore(zero_1, out_ub[2 * k + 1], stride=1, group=1)
                                for group_id in T.serial(num_groups):
                                    for i in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_id * experts_per_group + i * logical_lanes
                                        indices = V.vload(indices_ub[offset], size=logical_lanes)
                                        is_winner = V.vcmp(indices, winner, full, "eq")
                                        values = V.vload(scores_ub[offset], size=logical_lanes)
                                        values = V.vsel(is_winner, neg_inf, values)
                                        V.vstore(values, scores_ub[offset], stride=1, group=1)
                                S.mem_bar("VST_VLD")
                    T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])

    return kernel
