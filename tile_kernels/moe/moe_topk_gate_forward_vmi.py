import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.moe.sqrt_softplus import sqrt_softplus_forward_vmi
from tile_kernels.utils import align, ceil_div


@tilelang.jit(
    target='pto',
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_moe_topk_gate_forward_kernel_vmi(
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
    assert not fix_routing_mask_exists, 'the warp-reduction mock does not support fixed routing'
    assert not unmapped_topk_idx_exists, 'the warp-reduction mock does not produce unmapped indices'
    assert not to_physical_map_exists, 'the warp-reduction mock does not support physical expert mapping'
    assert not force_random_exists, 'the warp-reduction mock does not support random routing'

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    unmapped_topk_idx_stride = T.dynamic('unmapped_topk_idx_stride')

    # fmt: off
    @T.prim_func
    def moe_topk_gate_forward_kernel_vmi(
        logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        bias: T.Tensor[(num_routed_experts,), T.float32],
        image_bias: T.Tensor[(num_routed_experts,), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        mask: T.Tensor[(num_tokens,), T.bool],
        fix_routing_mask: T.Tensor[(num_tokens,), T.bool],
        to_physical_map: T.Tensor[(num_logical_experts, num_duplicate_experts), T.int32],
        logical_count: T.Tensor[(num_logical_experts,), T.int32],
        # Temporary PTO ABI: use int32 until T.view(int64 GM, int32) lowering is fixed.
        # Restore both index outputs to int64 when the public ABI can be lowered correctly.
        topk_idx: T.Tensor[(num_tokens, num_physical_topk), T.int32],
        unmapped_topk_idx: T.StridedTensor[(num_tokens, num_topk), (unmapped_topk_idx_stride, 1), T.int32],
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
            unmapped_idx_ub = T.alloc_shared((block_m, align(num_topk, 8)), T.int32)
            out_idx_ub = T.alloc_shared((block_m, align(num_physical_topk, 8)), T.int32)
            out_weight_ub = T.alloc_shared((block_m, align(num_physical_topk, 8)), T.float32)
            count_ub = T.alloc_shared((align(num_logical_experts, vector_length),), T.int32)
            map_ub = T.alloc_shared((num_logical_experts, num_duplicate_experts), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
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
                    with T.SimdVF():
                        full = V.create_mask(vector_length, size=vector_length)
                        zeros = V.vbrc(T.int32(0), size=vector_length)
                        total = V.alloc_var(T.int32, size=vector_length)
                        total = zeros
                        for j in T.unroll(ceil_div(num_logical_experts, vector_length), explicit=True):
                            group_base = j * vector_length
                            indices = V.vci(group_base, size=vector_length)
                            valid = V.vcmps(indices, T.int32(num_logical_experts), full, 'lt')
                            counts = V.vload(count_ub[j * vector_length], size=vector_length)
                            total = V.vadd(total, V.vsel(valid, counts, zeros))
                        V.vstore(V.vcadd(total, full), invalid_logits_ub[0], stride=1, group=1)
                    num_physical_experts = invalid_logits_ub[0]
            if bias_exists:
                T.copy(bias, bias_ub[0, :num_routed_experts])
            elif image_token_mask_exists:
                with T.SimdVF():
                    for j in T.unroll(num_vregs, explicit=True):
                        V.vstore(V.vbrc(T.float32(0.0), size=vector_length), bias_ub[0, j * vector_length])
            if image_token_mask_exists:
                T.copy(image_bias, bias_ub[1, :num_routed_experts])
            with T.SimdVF():
                T.clear(invalid_logits_ub)
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
                    T.copy(unmapped_topk_idx[row_start:row_start + num_rows, :], unmapped_idx_ub[:num_rows, :num_topk])
                if image_token_mask_exists:
                    T.copy(image_token_mask[row_start:row_start + num_rows], image_token_mask_ub[:num_rows])
                with T.SimdVF():
                    full = V.create_mask(vector_length, size=vector_length)
                    lanes = V.vci(T.int32(0), size=vector_length)
                    zero = V.vbrc(T.float32(0.0), size=vector_length)
                    zeros = V.vbrc(T.int32(0), size=vector_length)
                    ones = V.vbrc(T.int32(1), size=vector_length)
                    invalid = V.alloc_var(T.int32, size=vector_length)
                    invalid = V.vload(invalid_logits_ub[0], size=vector_length)
                    if fix_routing_mask_exists:
                        topk_lanes = V.vcmps(lanes, T.int32(num_topk), full, 'lt')
                    for row in T.serial(num_rows):
                        active = T.alloc_var(T.bool, init=row_start + row < num_tokens)
                        if mask_exists:
                            active = active and mask_ub[row]
                        if force_random_exists:
                            active = active and not force_random_ub[row]
                        bias_kind = T.alloc_var(T.int32, init=0)
                        if image_token_mask_exists:
                            bias_kind = T.if_then_else(image_token_mask_ub[row], 1, 0)
                        if fix_routing_mask_exists:
                            if active and fix_routing_mask_ub[row]:
                                fixed_offsets = V.vinterpret_cast(lanes, 'uint32')
                                fixed_idx = V.vgather(unmapped_idx_ub[row, 0], fixed_offsets, topk_lanes)
                                invalid_idx = V.vcmps(fixed_idx, T.int32(0), topk_lanes, 'lt')
                                invalid = V.vmax(invalid, V.vsel(invalid_idx, ones, zeros))
                        active_vec = V.vbrc(T.cast(active, T.int32), size=vector_length)
                        active_lanes = V.vcmps(active_vec, T.int32(0), full, 'ne')
                        for j in T.serial(num_vregs):
                            group_base = j * vector_length
                            indices = V.vci(group_base, size=vector_length)
                            valid = V.vcmps(indices, T.int32(num_routed_experts), full, 'lt')
                            logit = V.alloc_var(T.float32, size=vector_length)
                            logit = V.vload(scores_w_bias_ub[row, j * vector_length], size=vector_length)
                            if not experts_aligned:
                                logit = V.vsel(valid, logit, zero)
                            logit = V.vsel(active_lanes, logit, zero)
                            nonfinite = V.vcmps(V.vsub(logit, logit), T.float32(0.0), full, 'ne')
                            invalid = V.vmax(invalid, V.vsel(nonfinite, ones, zeros))
                            score = V.alloc_var(T.float32, size=vector_length)
                            score = sqrt_softplus_forward_vmi(logit, full, vector_length)
                            score = V.vsel(active_lanes, score, zero)
                            if not experts_aligned:
                                score = V.vsel(valid, score, zero)
                            # Bias affects expert selection only; normalization uses the original scores.
                            V.vstore(score, scores_wo_bias_ub[row, j * vector_length])
                            if bias_exists or image_token_mask_exists:
                                score = V.vadd(
                                    score, V.vload(bias_ub[bias_kind, j * vector_length], size=vector_length)
                                )
                            if not experts_aligned:
                                neg_inf = V.vbrc(-T.infinity(T.float32), size=vector_length)
                                score = V.vsel(valid, score, neg_inf)
                            V.vstore(score, scores_w_bias_ub[row, j * vector_length])
                    V.vstore(invalid, invalid_logits_ub[0])
                # PTO does not yet lower warp_reduce_{min,max,sum}. This explicit
                # mock keeps the surrounding pipeline benchmarkable, but it is not
                # top-k selection or weight normalization: routed lanes select the
                # first experts and expose their unnormalized scores.
                with T.SimtVF(threads=warp_size * block_m):
                    thread_idx = T.get_thread_binding(0)
                    row = thread_idx // warp_size
                    lane_idx = thread_idx % warp_size
                    active = T.alloc_var(T.bool, init=(row < num_rows) and (row_start + row < num_tokens))
                    if mask_exists:
                        active = active and mask_ub[row]
                    topk_idx_var = T.alloc_var(T.int32, init=-1)
                    weight = T.alloc_var(T.float32, init=0.0)
                    if active:
                        if lane_idx < num_topk:
                            topk_idx_var = lane_idx
                            weight = scores_wo_bias_ub[row, lane_idx]
                        elif lane_idx < num_physical_topk:
                            topk_idx_var = num_routed_experts + lane_idx - num_topk
                            weight = 1.0
                    if lane_idx < num_physical_topk:
                        out_idx_ub[row, lane_idx] = topk_idx_var
                        out_weight_ub[row, lane_idx] = weight
                T.copy(out_idx_ub[:num_rows, :num_physical_topk], topk_idx[row_start:row_start + num_rows, :])
                T.copy(out_weight_ub[:num_rows, :num_physical_topk], topk_weights[row_start:row_start + num_rows, :])
                if unmapped_topk_idx_exists:
                    T.copy(unmapped_idx_ub[:num_rows, :num_topk], unmapped_topk_idx[row_start:row_start + num_rows, :])
                if scores_exists:
                    T.copy(scores_wo_bias_ub[:num_rows, :num_routed_experts], scores[row_start:row_start + num_rows, :])
            with T.SimdVF():
                full = V.create_mask(vector_length, size=vector_length)
                V.vstore(
                    V.vcmax(V.vload(invalid_logits_ub[0], size=vector_length), full),
                    invalid_logits_ub[0], stride=1, group=1,
                )
            # Temporarily disabled for PTO codegen isolation: the current
            # backend has no handler for tl.device_assert_with_msg.
            # T.device_assert(invalid_logits_ub[0] == 0, msg='moe_topk_gate_forward input contains nan or inf or kernel has internal error!')
    # fmt: on

    return moe_topk_gate_forward_kernel_vmi
