"""Head mixing using scalar Parallel expressions and ASC token scheduling.

The ASC variant accesses parameters directly from SimtVF. SIMD needs those
parameters in UB, and rows are padded to a 32-byte DMA/store boundary. Parameter
gradients accumulate over tokens within each core, as in the ASC implementation.
"""
import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def mhc_head_compute_mix_fwd_parallel(
    input_mix, mhc_scale, mhc_base, output_mix,
    mhc_pre_eps: float, token_block_size: int, num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    width = T.ceildiv(mhc_mult, 8) * 8
    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((width,), T.float32)
        input_ub = T.alloc_shared((token_block_size, width), T.float32)
        output_ub = T.alloc_shared((token_block_size, width), T.float32)
        T.annotate_buffer_versions({input_ub: 2, output_ub: 2})
        T.copy(mhc_scale, scale_ub[:1])
        T.copy(mhc_base, base_ub[:mhc_mult])
        for pid in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)], num_vec_cores,
            core_id, num_stages=2,
        ):
            start = pid * token_block_size
            rows = T.min(token_block_size, num_tokens - start)
            T.copy(input_mix[start:start + rows, :], input_ub[:rows, :mhc_mult])
            with T.SimdVF(lanes=64):
                for row in T.serial(token_block_size):
                    for j in T.Parallel(width):
                        value = input_ub[row, j] * scale_ub[0] + base_ub[j]
                        if j < mhc_mult:
                            output_ub[row, j] = 1.0 / (1.0 + T.exp(-value)) + mhc_pre_eps
            T.copy(output_ub[:rows, :mhc_mult], output_mix[start:start + rows, :])


@tilelang.jit(target='pto')
def mhc_head_compute_mix_bwd_parallel(
    output_mix_grad, input_mix, mhc_scale, mhc_base, input_mix_grad,
    mhc_scale_grad_partial, mhc_base_grad_partial,
    token_block_size: int, num_vec_cores: int,
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
    width = T.ceildiv(mhc_mult, 8) * 8
    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((8,), T.float32)
        base_ub = T.alloc_shared((width,), T.float32)
        base_acc = T.alloc_shared((width,), T.float32)
        scale_acc = T.alloc_shared((width,), T.float32)
        scale_out = T.alloc_shared((8,), T.float32)
        input_ub = T.alloc_shared((token_block_size, width), T.float32)
        grad_ub = T.alloc_shared((token_block_size, width), T.float32)
        dx_ub = T.alloc_shared((token_block_size, width), T.float32)
        derivative_ub = T.alloc_shared((token_block_size, width), T.float32)
        T.copy(mhc_scale, scale_ub[:1])
        with T.SimdVF(lanes=64):
            for j in T.Parallel(width):
                base_ub[j] = 0.0
        T.copy(mhc_base, base_ub[:mhc_mult])
        with T.SimdVF(lanes=64):
            for j in T.Parallel(width):
                base_acc[j] = 0.0
                scale_acc[j] = 0.0
        for pid in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)], num_vec_cores,
            core_id, group_size=1,
        ):
            start = pid * token_block_size
            rows = T.min(token_block_size, num_tokens - start)
            # Invalid tail rows must contribute zero to parameter gradients.
            with T.SimdVF(lanes=64):
                for i, j in T.Parallel(token_block_size, width):
                    input_ub[i, j] = 0.0
                    grad_ub[i, j] = 0.0
            T.copy(input_mix[start:start + rows, :], input_ub[:rows, :mhc_mult])
            T.copy(output_mix_grad[start:start + rows, :], grad_ub[:rows, :mhc_mult])
            with T.SimdVF(lanes=64):
                for row in T.serial(token_block_size):
                    for j in T.Parallel(width):
                        value = input_ub[row, j] * scale_ub[0] + base_ub[j]
                        sigmoid = 1.0 / (1.0 + T.exp(-value))
                        grad = sigmoid * (1.0 - sigmoid) * grad_ub[row, j]
                        derivative_ub[row, j] = grad
                        dx_ub[row, j] = grad * scale_ub[0]
            with T.SimdVF(lanes=64):
                for j in T.Parallel(width):
                    base = T.alloc_var(T.float32, init=base_acc[j])
                    scale = T.alloc_var(T.float32, init=scale_acc[j])
                    for row in T.serial(token_block_size):
                        base += derivative_ub[row, j]
                        scale += derivative_ub[row, j] * input_ub[row, j]
                    base_acc[j] = base
                    scale_acc[j] = scale
            T.copy(dx_ub[:rows, :mhc_mult], input_mix_grad[start:start + rows, :])
        # A short reduction over mHC channels; only lane zero writes the
        # scalar output. Keep the destination base 32-byte aligned.
        with T.SimdVF(lanes=64):
            for j in T.Parallel(8):
                total = T.alloc_var(T.float32, init=0.0)
                for col in T.serial(mhc_mult):
                    total += scale_acc[col]
                if j == 0:
                    scale_out[j] = total
        T.copy(base_acc[:mhc_mult], mhc_base_grad_partial[core_id, :])
        T.copy(scale_out[:1], mhc_scale_grad_partial[core_id, :])
