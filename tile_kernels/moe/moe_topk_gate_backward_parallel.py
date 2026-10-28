import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_num_vec_cores
from tile_kernels.moe.sqrt_softplus_parallel import sqrt_softplus_backward_parallel
from tile_kernels.utils import align


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True})
def get_moe_topk_gate_backward_kernel_parallel(num_topk: int, num_routed_experts: int, aux_exists: bool):
    num_stages = 2
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)
    max_num_topk = 32  # Keep the same top-k limit as forward.

    # Give each token a power-of-two lane group so XOR reductions stay within that token.
    reduction_steps = (num_topk - 1).bit_length()
    topk_width = 1 << reduction_steps
    rows_per_vec = vector_length // topk_width
    idx_stride = align(num_topk * 2, 8)  # int64 indices viewed as pairs of int32.
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
    def moe_topk_gate_backward_kernel_parallel(
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
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

            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
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
                group_size=1,
            ):
                row_start = tile_id * num_rows
                tile_rows = T.min(num_rows, num_tokens - row_start)
                T.copy(scores[row_start : row_start + num_rows, :], scores_ub[:num_rows, :num_routed_experts])
                T.copy(topk_idx_view[row_start : row_start + num_rows, :], idx_ub[:num_rows, : num_topk * 2])
                T.copy(topk_weights[row_start : row_start + num_rows, :], weight_ub[:num_rows, :num_topk])
                T.copy(grad_topk_weights[row_start : row_start + num_rows, :], grad_weight_ub[:num_rows, :num_topk])
                if aux_exists:
                    # Token and sequence indices are nonnegative; truncating division
                    # equals floor division and avoids signed correction expressions.
                    tokens_per_seq = T.truncdiv(num_tokens, num_seqs)
                    seq_first = T.truncdiv(row_start, tokens_per_seq)
                    seq_span = T.truncdiv(row_start + tile_rows - 1, tokens_per_seq) - seq_first + 1
                    T.copy(mask[row_start : row_start + num_rows], mask_ub[:num_rows])
                    T.copy(grad_scores_sum[T.int64(seq_first) : T.int64(seq_first + seq_span), :], aux_grad_ub[:seq_span, :num_routed_experts])

                with T.SimdVF(lanes=64):
                    selected = T.alloc_shared((num_aligned_experts,), T.float32)
                    weighted = T.alloc_shared((weight_stride,), T.float32)
                    denominator = T.alloc_shared((1,), T.float32)
                    dot = T.alloc_shared((1,), T.float32)
                    aux_products = T.alloc_shared((num_aligned_experts,), T.float32)
                    score_sum = T.alloc_shared((1,), T.float32)
                    aux_dot = T.alloc_shared((1,), T.float32)
                    for row in T.serial(block_m):
                        for j in T.Parallel(num_aligned_experts):
                            total = T.alloc_var(T.float32, init=0.0)
                            for k in T.serial(num_topk):
                                total = total + T.if_then_else((j < num_routed_experts) & (j == idx_ub[row, k * 2]), scores_ub[row, j], 0.0)
                            selected[j] = total
                        for k in T.Parallel(num_topk):
                            weighted[k] = T.if_then_else((k < num_topk) & (idx_ub[row, k * 2] >= 0), weight_ub[row, k] * grad_weight_ub[row, k], 0.0)
                        T.reduce_sum(selected, denominator, dim=0)
                        T.reduce_sum(weighted[:num_topk], dot, dim=0)
                        if aux_exists:
                            hrow = T.truncdiv(row_start + row, tokens_per_seq) - seq_first
                            for j in T.Parallel(num_aligned_experts):
                                selected[j] = T.if_then_else(j < num_routed_experts, scores_ub[row, j], 0.0)
                                aux_products[j] = T.if_then_else(j < num_routed_experts, scores_ub[row, j] * aux_grad_ub[T.int64(hrow), j], 0.0)
                            T.reduce_sum(selected, score_sum, dim=0)
                            T.reduce_sum(aux_products, aux_dot, dim=0)
                        for j in T.Parallel(num_aligned_experts):
                            grad_score = T.alloc_var(T.float32, init=0.0)
                            for k in T.serial(num_topk):
                                delta = (routed_scaling_factor * grad_weight_ub[row, k] - dot[0]) / (denominator[0] + 1e-20)
                                grad_score = grad_score + T.if_then_else(j == idx_ub[row, k * 2], delta, 0.0)
                            if aux_exists:
                                inv = T.if_then_else(mask_ub[row], 1.0 / score_sum[0], 0.0)
                                grad_score = grad_score + aux_grad_ub[T.int64(hrow), j] * inv - aux_dot[0] * inv * inv
                            grad_ub[row, j] = grad_score * sqrt_softplus_backward_parallel(scores_ub[row, j])

                T.copy(grad_ub[:num_rows, :num_routed_experts], grad_logits[row_start : row_start + num_rows, :])

    return moe_topk_gate_backward_kernel_parallel
