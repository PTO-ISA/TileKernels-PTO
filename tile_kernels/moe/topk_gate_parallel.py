"""Stable TopK using the CUDA max/min-index/kill algorithm and ASC scheduling."""

import tilelang
from tilelang.ascend import language as T

from tile_kernels.utils import align, ceil_div


@tilelang.jit(target='pto')
def get_topk_gate_kernel_parallel(num_experts: int, num_topk: int, num_vec_cores: int):
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
    def topk_gate_kernel_parallel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            out_ub = T.alloc_shared((num_aligned_topk,), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            T.annotate_buffer_versions({scores_ub: num_stages, out_ub: num_stages, invalid_logits_ub: 1})

            with T.SimdVF(lanes=64):
                T.clear(out_ub)
                T.clear(invalid_logits_ub)

            for w in T.Pipelined(
                T.ceildiv(num_tokens, num_vec_cores),
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [scores_ub, out_ub]},
            ):
                row = w * num_vec_cores + core_id
                if row < num_tokens:
                    T.copy(scores[row, :], scores_ub[:num_experts])

                    with T.SimdVF(lanes=64):
                        nonfinite = T.alloc_shared((num_aligned_experts,), T.int32)
                        candidates = T.alloc_shared((num_aligned_experts,), T.int32)
                        max_score = T.alloc_shared((1,), T.float32)
                        winner = T.alloc_shared((1,), T.int32)
                        invalid_row = T.alloc_shared((1,), T.int32)
                        for j in T.Parallel(num_aligned_experts):
                            score = T.if_then_else(j < num_experts, scores_ub[j], 0.0)
                            nonfinite[j] = T.if_then_else(score - score != 0.0, 1, 0)
                            scores_ub[j] = T.if_then_else(j < num_experts, score, -T.infinity(T.float32))
                        T.reduce_max(nonfinite, invalid_row, dim=0)
                        for j in T.Parallel(1):
                            invalid_logits_ub[j] = T.max(invalid_logits_ub[j], invalid_row[j])
                        for k in T.serial(num_topk):
                            T.reduce_max(scores_ub, max_score, dim=0)
                            for j in T.Parallel(num_aligned_experts):
                                candidates[j] = T.if_then_else((j < num_experts) & (scores_ub[j] == max_score[0]), j, T.max_value(T.int32))
                            T.reduce_min(candidates, winner, dim=0)
                            for j in T.Parallel(1):
                                out_ub[2 * k + j] = winner[j]
                            for j in T.Parallel(num_aligned_experts):
                                scores_ub[j] = T.if_then_else(j == winner[0], -T.infinity(T.float32), scores_ub[j])

                    T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])

            T.device_assert(invalid_logits_ub[0] == 0, msg='topk_gate input contains nan or inf!')

    return topk_gate_kernel_parallel
