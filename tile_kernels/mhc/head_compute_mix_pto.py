import tilelang
from tilelang.ascend import language as T


@T.macro
def _sigmoid_pto(x):
    return T.float32(1.0) / (T.float32(1.0) + T.exp(-x))


@tilelang.jit(target='pto')
def mhc_head_compute_mix_fwd_pto(
    input_mix,
    mhc_scale,
    mhc_base,
    output_mix,
    mhc_pre_eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((1,), T.float32)
        base_ub = T.alloc_shared((mhc_mult,), T.float32)
        T.copy(mhc_scale, scale_ub)
        T.copy(mhc_base, base_ub)

        for pid in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=2,
        ):
            # Buffers that need a per-iteration version are allocated inside the
            # Persistent body: an integer version count would turn every access
            # into `buf[counter[0] % N, ...]`, and a runtime counter load is not
            # a legal UB address expression.
            input_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
            output_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
            T.annotate_buffer_versions({
                input_ub: (2, 'iteration'),
                output_ub: (2, 'iteration'),
            })

            row = pid * token_block_size
            valid_rows = T.min(token_block_size, num_tokens - row)
            T.copy(input_mix[row : row + valid_rows, :], input_ub[:valid_rows, :])

            with T.SimdVF(lanes=256):
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    output_ub[i, j] = _sigmoid_pto(
                        input_ub[i, j] * scale_ub[0] + base_ub[j]
                    ) + T.float32(mhc_pre_eps)

            T.copy(output_ub[:valid_rows, :], output_mix[row : row + valid_rows, :])


@tilelang.jit(target='pto')
def mhc_head_compute_mix_bwd_pto(
    output_mix_grad,
    input_mix,
    mhc_scale,
    mhc_base,
    input_mix_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    token_block_size: int,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale_grad_partial: T.Tensor[(num_vec_cores, 1), T.float32]
    mhc_base_grad_partial: T.Tensor[(num_vec_cores, mhc_mult), T.float32]

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((1,), T.float32)
        base_ub = T.alloc_shared((mhc_mult,), T.float32)
        # Cross-tile accumulators. They stay lane-wide: a (1,)-wide add would be
        # a runtime lane-uniform float computation, which SIMD_VF cannot lower.
        scale_acc_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        base_acc_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
        scale_rows_ub = T.alloc_shared((mhc_mult,), T.float32)
        scale_final_ub = T.alloc_shared((1,), T.float32)
        base_final_ub = T.alloc_shared((mhc_mult,), T.float32)
        T.copy(mhc_scale, scale_ub)
        T.copy(mhc_base, base_ub)
        with T.SimdVF(lanes=256):
            for i, j in T.Parallel(token_block_size, mhc_mult):
                scale_acc_ub[i, j] = T.float32(0.0)
                base_acc_ub[i, j] = T.float32(0.0)

        for tile_id in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=2,
        ):
            input_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
            output_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
            input_grad_ub = T.alloc_shared((token_block_size, mhc_mult), T.float32)
            T.annotate_buffer_versions({
                input_ub: (2, 'iteration'),
                output_grad_ub: (2, 'iteration'),
                input_grad_ub: (2, 'iteration'),
            })

            row = tile_id * token_block_size
            valid_rows = T.min(token_block_size, num_tokens - row)
            # Zero the tail rows: a Select against a runtime row count would be
            # classified as a conditional gather, so the tile is cleared and the
            # copies overwrite the valid prefix.
            with T.SimdVF(lanes=256):
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    input_ub[i, j] = T.float32(0.0)
                    output_grad_ub[i, j] = T.float32(0.0)
            T.copy(input_mix[row : row + valid_rows, :], input_ub[:valid_rows, :])
            T.copy(output_mix_grad[row : row + valid_rows, :], output_grad_ub[:valid_rows, :])

            with T.SimdVF(lanes=256):
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    x = input_ub[i, j]
                    sig = _sigmoid_pto(x * scale_ub[0] + base_ub[j])
                    grad = sig * (T.float32(1.0) - sig) * output_grad_ub[i, j]
                    input_grad_ub[i, j] = grad * scale_ub[0]
                    scale_acc_ub[i, j] = scale_acc_ub[i, j] + grad * x
                    base_acc_ub[i, j] = base_acc_ub[i, j] + grad

            T.copy(input_grad_ub[:valid_rows, :], input_mix_grad[row : row + valid_rows, :])

        with T.SimdVF(lanes=256):
            T.reduce_sum(scale_acc_ub, scale_rows_ub, dim=0)
            T.reduce_sum(scale_rows_ub, scale_final_ub, dim=0)
            T.reduce_sum(base_acc_ub, base_final_ub, dim=0)
        T.copy(scale_final_ub, mhc_scale_grad_partial[core_id, :])
        T.copy(base_final_ub, mhc_base_grad_partial[core_id, :])
