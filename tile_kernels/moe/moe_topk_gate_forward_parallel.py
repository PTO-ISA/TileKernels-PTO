import tilelang
from tilelang.ascend import language as T

from tile_kernels.moe.sqrt_softplus_parallel import sqrt_softplus_parallel
from tile_kernels.utils import align, ceil_div


@tilelang.jit(
    target='pto',
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_moe_topk_gate_forward_kernel_parallel(
    num_topk: int,
    num_routed_experts: int,
    num_shared_experts: int,
    num_duplicate_experts: int,
    mask_exists: bool, fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool, to_physical_map_exists: bool,
    bias_exists: bool, image_token_mask_exists: bool,
    force_random_exists: bool, scores_exists: bool,
    num_vec_cores: int,
):  # fmt: off
    num_stages = 2
    vector_length = 64  # SIMD vector length (float32 lanes per vreg)
    warp_size = 32
    large_prime_number = 23333  # Keep the same with the CUDA implementation

    num_vregs = ceil_div(num_routed_experts, vector_length)
    num_aligned_experts = num_vregs * vector_length
    # No padding lanes when experts exactly fill the vregs
    experts_aligned = num_routed_experts % vector_length == 0

    num_physical_topk = num_topk + num_shared_experts
    num_logical_experts = num_routed_experts + num_shared_experts
    if num_aligned_experts <= 256:
        block_m = 32
    else:
        block_m = 16
    max_num_topk = min(warp_size, vector_length // 2)
    assert num_topk <= max_num_topk, f'num_topk must be <= {max_num_topk}, got {num_topk}'
    assert num_physical_topk <= warp_size, f'num_physical_topk must be <= {warp_size}, got {num_physical_topk}'

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    unmapped_topk_idx_stride = T.dynamic('unmapped_topk_idx_stride')

    # fmt: off
    @T.prim_func
    def moe_topk_gate_forward_kernel_parallel(
        logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        bias: T.Tensor[(num_routed_experts,), T.float32],
        image_bias: T.Tensor[(num_routed_experts,), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        mask: T.Tensor[(num_tokens,), T.bool],
        fix_routing_mask: T.Tensor[(num_tokens,), T.bool],
        to_physical_map: T.Tensor[(num_logical_experts, num_duplicate_experts), T.int32],
        logical_count: T.Tensor[(num_logical_experts,), T.int32],
        topk_idx: T.Tensor[(num_tokens, num_physical_topk), T.int64],
        unmapped_topk_idx: T.StridedTensor[(num_tokens, num_topk), (unmapped_topk_idx_stride, 1), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_physical_topk), T.float32],
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        force_random: T.Tensor[(num_tokens,), T.bool],
        _num_shared_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            scores_w_bias_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            scores_wo_bias_ub = T.alloc_shared((block_m, num_aligned_experts), T.float32)
            bias_ub = T.alloc_shared((2, num_aligned_experts), T.float32)
            mask_ub = T.alloc_shared((block_m,), T.bool)
            force_random_ub = T.alloc_shared((block_m,), T.bool)
            fix_routing_mask_ub = T.alloc_shared((block_m,), T.bool)
            image_token_mask_ub = T.alloc_shared((block_m,), T.bool)
            unmapped_idx_ub = T.alloc_shared((block_m, align(num_topk * 2, 8)), T.int32)
            out_idx_ub = T.alloc_shared((block_m, align(num_physical_topk * 2, 8)), T.int32)
            out_weight_ub = T.alloc_shared((block_m, align(num_physical_topk, 8)), T.float32)
            count_ub = T.alloc_shared((align(num_logical_experts, vector_length),), T.int32)
            map_ub = T.alloc_shared((num_logical_experts, num_duplicate_experts), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_physical_topk * 2), T.int32)
            unmapped_idx_view = T.view(unmapped_topk_idx, (num_tokens, num_topk * 2), T.int32, (unmapped_topk_idx_stride * 2, 1))
            versions = {scores_w_bias_ub: num_stages, scores_wo_bias_ub: num_stages,
                        out_idx_ub: num_stages, out_weight_ub: num_stages, invalid_logits_ub: 1}
            outputs = [scores_wo_bias_ub, out_idx_ub, out_weight_ub]
            if mask_exists:
                versions[mask_ub] = num_stages
            if force_random_exists:
                versions[force_random_ub] = num_stages
            if fix_routing_mask_exists:
                versions[fix_routing_mask_ub] = num_stages
            if image_token_mask_exists:
                versions[image_token_mask_ub] = num_stages
            if unmapped_topk_idx_exists:
                versions[unmapped_idx_ub] = num_stages
                outputs.append(unmapped_idx_ub)
            T.annotate_buffer_versions(versions)
            num_physical_experts = T.alloc_var(T.int32, init=num_logical_experts)
            if to_physical_map_exists:
                T.copy(logical_count, count_ub[:num_logical_experts])
                T.copy(to_physical_map, map_ub)
                if force_random_exists:
                    with T.SimdVF(lanes=64):
                        count_masked = T.alloc_shared((align(num_logical_experts,vector_length),),T.int32)
                        for j in T.Parallel(align(num_logical_experts,vector_length)):
                            count_masked[j] = T.if_then_else(j < num_logical_experts,count_ub[j],0)
                        T.reduce_sum(count_masked,invalid_logits_ub[:1],dim=0)
                    num_physical_experts = invalid_logits_ub[0]
            if bias_exists:
                T.copy(bias, bias_ub[0, :num_routed_experts])
            elif image_token_mask_exists:
                with T.SimdVF(lanes=64):
                    for j in T.Parallel(num_aligned_experts):
                        bias_ub[0,j] = 0.0
            if image_token_mask_exists:
                T.copy(image_bias, bias_ub[1, :num_routed_experts])
            with T.SimdVF(lanes=64):
                for j in T.Parallel(vector_length):
                    invalid_logits_ub[j] = 0
            num_rows = T.min(block_m, T.max(1, T.ceildiv(num_tokens, num_vec_cores)))
            for tile_id in T.Persistent([T.ceildiv(num_tokens, num_rows)], num_vec_cores, core_id,
                                        num_stages=num_stages, annotations={'multi_buffer_eligible': outputs}):
                row_start = tile_id * num_rows
                T.copy(logits[row_start:row_start + num_rows, :], scores_w_bias_ub[:num_rows, :num_routed_experts])
                if mask_exists:
                    T.copy(mask[row_start:row_start + num_rows], mask_ub[:num_rows])
                if force_random_exists:
                    T.copy(force_random[row_start:row_start + num_rows], force_random_ub[:num_rows])
                if fix_routing_mask_exists:
                    T.copy(fix_routing_mask[row_start:row_start + num_rows], fix_routing_mask_ub[:num_rows])
                    T.copy(unmapped_idx_view[row_start:row_start + num_rows, :], unmapped_idx_ub[:num_rows, :num_topk * 2])
                if image_token_mask_exists:
                    T.copy(image_token_mask[row_start:row_start + num_rows], image_token_mask_ub[:num_rows])
                with T.SimdVF(lanes=64):
                    invalid_elements = T.alloc_shared((num_aligned_experts,),T.int32)
                    invalid_row = T.alloc_shared((1,),T.int32)
                    for row in T.serial(num_rows):
                        masked_active = ((row_start+row < num_tokens) & mask_ub[row]) if mask_exists else (row_start+row < num_tokens)
                        active = (masked_active & (not force_random_ub[row])) if force_random_exists else masked_active
                        bias_kind = T.if_then_else(image_token_mask_ub[row],1,0) if image_token_mask_exists else 0
                        if fix_routing_mask_exists:
                            if active and fix_routing_mask_ub[row]:
                                for j in T.Parallel(num_topk):
                                    invalid_logits_ub[j] = T.max(invalid_logits_ub[j],T.if_then_else((j<num_topk) & (unmapped_idx_ub[row,j*2]<0),1,0))
                        for j in T.Parallel(num_aligned_experts):
                            logit = T.if_then_else(active & (j<num_routed_experts),scores_w_bias_ub[row,j],0.0)
                            invalid_elements[j] = T.if_then_else(logit-logit != 0.0,1,0)
                            score = sqrt_softplus_parallel(logit)
                            score = T.if_then_else(active & (j<num_routed_experts),score,0.0)
                            scores_wo_bias_ub[row,j] = score
                            if bias_exists or image_token_mask_exists:
                                score = score + bias_ub[bias_kind,j]
                            scores_w_bias_ub[row,j] = T.if_then_else(j<num_routed_experts,score,-T.infinity(T.float32))
                        T.reduce_max(invalid_elements,invalid_row,dim=0)
                        for j in T.Parallel(1):
                            invalid_logits_ub[j] = T.max(invalid_logits_ub[j],invalid_row[j])
                # Use one warp per token to select experts, normalize weights, and map physical indices.
                with T.SimtVF(threads=warp_size * block_m):
                    thread_idx = T.get_thread_binding(0)
                    row = thread_idx // warp_size
                    lane_idx = thread_idx % warp_size
                    active = T.alloc_var(T.bool, init=(row < num_rows) and (row_start + row < num_tokens))
                    if mask_exists:
                        active = active and mask_ub[row]
                    is_force_random_token = T.alloc_var(T.bool, init=False)
                    use_fixed = T.alloc_var(T.bool, init=False)
                    if force_random_exists:
                        is_force_random_token = force_random_ub[row]
                    if fix_routing_mask_exists:
                        use_fixed = fix_routing_mask_ub[row]
                    topk_idx_var = T.alloc_var(T.int32, init=-1)
                    unmapped = T.alloc_var(T.int32, init=-1)
                    weight = T.alloc_var(T.float32, init=0.0)
                    if active:
                        if is_force_random_token:
                            T.rng_init(ep_rank, (row_start + row) * warp_size + lane_idx, 0)
                            weight = T.rng_rand_float()
                            rand_topk_idx_tmp_var = T.alloc_var(T.int32, init=T.max_value(T.int32))
                            if lane_idx < num_physical_topk:
                                rand_topk_idx_tmp_var = T.cast(T.rng_rand() % T.cast(num_physical_experts - lane_idx, T.uint32), T.int32)
                            else:
                                topk_idx_var = T.max_value(T.int32)
                            # Resolve draws in value/lane order, shifting later lanes to keep experts distinct.
                            for j in T.serial(num_physical_topk):
                                min_topk_idx = T.warp_reduce_min(rand_topk_idx_tmp_var)
                                min_lane_idx = T.warp_reduce_min(T.if_then_else(rand_topk_idx_tmp_var == min_topk_idx, lane_idx, T.max_value(T.int32)))
                                if topk_idx_var == -1:
                                    if rand_topk_idx_tmp_var >= min_topk_idx and min_lane_idx < lane_idx:
                                        rand_topk_idx_tmp_var += 1
                                    elif min_lane_idx == lane_idx:
                                        topk_idx_var = rand_topk_idx_tmp_var
                                        rand_topk_idx_tmp_var = T.max_value(T.int32)
                        else:
                            if use_fixed:
                                if lane_idx < num_topk:
                                    topk_idx_var = unmapped_idx_ub[row, lane_idx * 2]
                            else:
                                scores_local = T.alloc_local((num_aligned_experts // warp_size,), T.float32)
                                for j in T.unroll(num_aligned_experts // warp_size, explicit=True):
                                    scores_local[j] = scores_w_bias_ub[row, j * warp_size + lane_idx]
                                for j in T.serial(num_topk):
                                    max_score = T.alloc_var(T.float32, init=-T.infinity(T.float32))
                                    local_winner = T.alloc_var(T.int32, init=T.max_value(T.int32))
                                    for i in T.unroll(num_aligned_experts // warp_size, explicit=True):
                                        local_winner = T.if_then_else(scores_local[i] > max_score, i * warp_size + lane_idx, local_winner)
                                        max_score = T.max(max_score, scores_local[i])
                                    global_max = T.warp_reduce_max(max_score)
                                    winner_idx = T.warp_reduce_min(
                                        T.if_then_else(max_score == global_max, local_winner, T.max_value(T.int32))
                                    )
                                    if lane_idx == j:
                                        topk_idx_var = winner_idx
                                    for i in T.unroll(num_aligned_experts // warp_size, explicit=True):
                                        scores_local[i] = T.if_then_else(winner_idx == i * warp_size + lane_idx, -T.infinity(T.float32), scores_local[i])
                            if lane_idx < num_topk:
                                weight = scores_wo_bias_ub[row, T.max(topk_idx_var, 0)]
                                unmapped = topk_idx_var
                            weight_sum = T.warp_reduce_sum(weight)
                            denominator = weight_sum + 1e-20
                            # Compare adjacent floats by division residual to refine the normalized weight.
                            quotient = T.alloc_var(T.float32, init=weight / denominator)
                            quotient_prev = T.reinterpret(T.reinterpret(quotient, T.int32) - 1, T.float32)
                            quotient_next = T.reinterpret(T.reinterpret(quotient, T.int32) + 1, T.float32)
                            residual = T.abs(quotient * -denominator + weight)
                            residual_prev = T.abs(quotient_prev * -denominator + weight)
                            residual_next = T.abs(quotient_next * -denominator + weight)
                            closest = T.if_then_else(residual < residual_prev, quotient, quotient_prev)
                            min_residual = T.min(residual, residual_prev)
                            corrected = T.if_then_else(residual_next < min_residual, quotient_next, closest)
                            quotient = T.if_then_else(quotient == 0.0, quotient, corrected)
                            weight = T.if_then_else(lane_idx < num_topk, quotient * routed_scaling_factor, 1.0)
                            if lane_idx >= num_topk:
                                topk_idx_var = num_routed_experts + lane_idx - num_topk
                            if to_physical_map_exists:
                                if lane_idx < num_physical_topk:
                                    seed = ep_rank + (row_start + row) * large_prime_number
                                    duplicate_idx = T.alloc_var(T.int32, init=0)
                                    if num_duplicate_experts > 1:
                                        duplicate_idx = T.truncmod(seed, count_ub[topk_idx_var])
                                    topk_idx_var = map_ub[topk_idx_var, duplicate_idx]
                    if lane_idx < num_physical_topk:
                        out_idx_ub[row, lane_idx * 2] = topk_idx_var
                        out_idx_ub[row, lane_idx * 2 + 1] = T.if_then_else(active, 0, -1)
                        out_weight_ub[row, lane_idx] = weight
                    if unmapped_topk_idx_exists:
                        if lane_idx < num_topk:
                            unmapped_idx_ub[row, lane_idx * 2] = unmapped
                            unmapped_idx_ub[row, lane_idx * 2 + 1] = T.if_then_else(unmapped < 0, -1, 0)
                T.copy(out_idx_ub[:num_rows, :num_physical_topk * 2], topk_idx_view[row_start:row_start + num_rows, :])
                T.copy(out_weight_ub[:num_rows, :num_physical_topk], topk_weights[row_start:row_start + num_rows, :])
                if unmapped_topk_idx_exists:
                    T.copy(unmapped_idx_ub[:num_rows, :num_topk * 2], unmapped_idx_view[row_start:row_start + num_rows, :])
                if scores_exists:
                    T.copy(scores_wo_bias_ub[:num_rows, :num_routed_experts], scores[row_start:row_start + num_rows, :])
            with T.SimdVF(lanes=64):
                T.reduce_max(invalid_logits_ub,invalid_logits_ub[:1],dim=0)
            T.device_assert(invalid_logits_ub[0] == 0, msg='moe_topk_gate_forward input contains nan or inf or kernel has internal error!')
    # fmt: on

    return moe_topk_gate_forward_kernel_parallel
