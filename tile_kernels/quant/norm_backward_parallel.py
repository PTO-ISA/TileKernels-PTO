import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.utils import align


@tilelang.jit(target='pto')
def get_norm_backward_kernel_parallel(
    num_tokens: int,
    hidden: int,
    with_weight: bool,
    with_residual_out_grad: bool,
    weight_grad_initialized: bool,
    out_scale: float,
    x_dtype: T.dtype,
    weight_grad_dtype: T.dtype,
    num_vec_cores: int,
):
    assert hidden % 128 == 0
    vec_size = 64
    rows_per_core = 1 if num_tokens < 4 * num_vec_cores else min(64, max(8, 1 << ((num_tokens // num_vec_cores).bit_length() - 1)))
    row_bytes = hidden * x_dtype.bytes
    # Target 16 KiB per input tile; a wide row is the minimum tile.
    block_m = min(rows_per_core, 1 << max(0, (16384 // row_bytes).bit_length() - 1))
    group_size = min(16, rows_per_core // block_m)
    block_k = 128 if hidden > num_vec_cores * vec_size else vec_size
    scratch_size = max(hidden, num_vec_cores * block_k)

    row_stats_bytes = align(block_m * T.float32.bytes, 32)
    fixed_bytes = (hidden + 1) * 4 + (row_bytes + scratch_size * T.float32.bytes) * with_weight + row_stats_bytes
    stage_bytes = 3 * block_m * row_bytes + row_stats_bytes
    num_stages = min(4, (get_max_ub_per_vector_core(use_simt=False) - fixed_bytes) // stage_bytes)
    assert num_stages >= 2, 'norm tile exceeds the available UB for two stages'

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    @T.macro
    def backward_tile(x_ub, weight_ub, out_grad_ub, x_grad_ub, rstd_ub, mean_ub, weight_grad_ub, num_rows):
        with T.SimdVF(lanes=64):
            products = T.alloc_shared((hidden,), T.float32)
            total = T.alloc_shared((1,), T.float32)
            for row in T.serial(num_rows):
                for j in T.Parallel(hidden):
                    xhat = T.float32(x_ub[row, j]) * rstd_ub[row]
                    grad = T.float32(out_grad_ub[row, j]) * out_scale
                    if with_weight:
                        grad = grad * T.float32(weight_ub[j])
                    products[j] = xhat * grad
                T.reduce_sum(products, total, dim=0)
                for j in T.Parallel(1):
                    mean_ub[row + j] = total[j] * (1.0 / hidden)
                for j in T.Parallel(hidden):
                    xhat = T.float32(x_ub[row, j]) * rstd_ub[row]
                    dy = T.float32(out_grad_ub[row, j]) * out_scale
                    if with_weight:
                        weight_grad_ub[j] = weight_grad_ub[j] + xhat * dy
                        dy = dy * T.float32(weight_ub[j])
                    dx = (dy - xhat * mean_ub[row]) * rstd_ub[row]
                    if with_residual_out_grad:
                        dx = dx + T.float32(x_grad_ub[row, j])
                    x_grad_ub[row, j] = T.cast(dx, x_dtype)

    @T.macro
    def reduce_weight_grad(partial_ub, out_ub):
        for j in T.Parallel(block_k):
            total = T.alloc_var(T.float32, init=0.0)
            for core in T.serial(num_vec_cores):
                total = total + partial_ub[core, j]
            if weight_grad_initialized:
                total = total + T.float32(out_ub[j])
            out_ub[j] = T.cast(total, weight_grad_dtype)

    @T.prim_func
    def norm_backward_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (x_stride, 1), x_dtype],
        weight: T.Tensor[(hidden,), x_dtype],
        out_grad: T.StridedTensor[(num_tokens, hidden), (out_grad_stride, 1), x_dtype],
        rstd: T.Tensor[(num_tokens,), T.float32],
        x_grad: T.Tensor[(num_tokens, hidden), x_dtype],
        weight_grad_partial: T.Tensor[(num_vec_cores, hidden), T.float32],
        residual_out_grad: T.StridedTensor[(num_tokens, hidden), (residual_out_grad_stride, 1), x_dtype],
        weight_grad: T.Tensor[(hidden,), weight_grad_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, hidden), x_dtype)
            weight_ub = T.alloc_shared((hidden,), x_dtype)
            out_grad_ub = T.alloc_shared((block_m, hidden), x_dtype)
            x_grad_ub = T.alloc_shared((block_m, hidden), x_dtype)
            scratch_ub = T.alloc_shared((scratch_size,), T.float32)
            weight_grad_ub = T.Tensor((hidden,), T.float32, scratch_ub.data, scope='shared')
            rstd_ub = T.alloc_shared((block_m,), T.float32)
            mean_ub = T.alloc_shared((block_m,), T.float32)
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    out_grad_ub: num_stages,
                    x_grad_ub: num_stages,
                    rstd_ub: num_stages,
                    mean_ub: 1,
                }
            )
            if with_weight:
                T.copy(weight, weight_ub)
                with T.SimdVF(lanes=64):
                    for j in T.Parallel(hidden):
                        weight_grad_ub[j] = 0.0

            for tile in T.Pipelined(T.ceildiv(num_tokens, block_m * group_size * num_vec_cores) * group_size, num_stages=num_stages):
                group = tile // group_size * num_vec_cores + core_id
                token_start = (group * group_size + tile % group_size) * block_m
                if token_start < num_tokens:
                    T.copy(rstd[token_start], rstd_ub)
                    T.copy(x[token_start, 0], x_ub)
                    T.copy(out_grad[token_start, 0], out_grad_ub)
                    if with_residual_out_grad:
                        T.copy(residual_out_grad[token_start, 0], x_grad_ub)
                    backward_tile(
                        x_ub,
                        weight_ub,
                        out_grad_ub,
                        x_grad_ub,
                        rstd_ub,
                        mean_ub,
                        weight_grad_ub,
                        T.min(block_m, num_tokens - token_start),
                    )
                    T.copy(x_grad_ub, x_grad[token_start, 0])
            if with_weight:
                if x_dtype == T.bfloat16:
                    with T.SimdVF(lanes=64):
                        T.evaluate(0)
                T.copy(weight_grad_ub, weight_grad_partial[core_id, 0])
                T.ascend_sync_inter_arrive('PIPE_MTE3', 0)
                T.ascend_sync_inter_wait('PIPE_MTE2', 0)
                partial_ub = T.reshape(scratch_ub, (scratch_size // block_k, block_k))
                out_ub = T.view(weight_ub, (hidden * x_dtype.bytes // weight_grad_dtype.bytes,), weight_grad_dtype)
                for pid_hidden in T.Persistent([hidden // block_k], num_vec_cores, core_id, group_size=1, num_stages=1):
                    col = pid_hidden * block_k
                    T.copy(weight_grad_partial[0, col], partial_ub[:num_vec_cores, :block_k])
                    if weight_grad_initialized:
                        T.copy(weight_grad[col], out_ub[:block_k])
                    with T.SimdVF():
                        reduce_weight_grad(partial_ub, out_ub)
                    T.copy(out_ub[:block_k], weight_grad[col])

    return norm_backward_kernel
