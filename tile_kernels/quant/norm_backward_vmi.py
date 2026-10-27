import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.norm_forward_vmi import load_f32
from tile_kernels.utils import align


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_norm_backward_kernel_vmi(
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
    fixed_bytes = (row_bytes + scratch_size * T.float32.bytes) * with_weight + row_stats_bytes
    stage_bytes = 3 * block_m * row_bytes + row_stats_bytes
    num_stages = min(4, (get_max_ub_per_vector_core(use_simt=False) - fixed_bytes) // stage_bytes)
    assert num_stages >= 2, 'norm tile exceeds the available UB for two stages'

    num_tokens = T.dynamic('num_tokens')
    x_stride = T.dynamic('x_stride')
    out_grad_stride = T.dynamic('out_grad_stride')
    residual_out_grad_stride = T.dynamic('residual_out_grad_stride')

    @T.macro
    def load_scaled_grad(buffer, col, size):
        dy = load_f32(buffer, col, size)
        if out_scale != 1.0:
            dy = V.vmuls(dy, out_scale, V.create_mask(size, size=size))
        return dy

    @T.macro
    def backward_tile(x_ub, weight_ub, out_grad_ub, x_grad_ub, rstd_ub, mean_ub, weight_grad_ub, num_rows):
        x_ub = T.reshape(x_ub, (block_m * hidden,))
        out_grad_ub = T.reshape(out_grad_ub, (block_m * hidden,))
        x_grad_ub = T.reshape(x_grad_ub, (block_m * hidden,))
        with T.SimdVF():
            mask128 = V.create_mask(128, size=128)
            mask1 = V.create_mask(1, size=1)
            # g = dy * out_scale * weight; dX = (g - xhat * mean(xhat * g)) * rstd.
            for row in T.serial(T.int16(num_rows)):
                rstd_value = V.vbrc(V.vload(rstd_ub[row], size=1), size=128, group=1)
                total = V.alloc_var(T.float32, size=128)
                total = V.vbrc(T.float32(0.0), size=128)
                for vector in T.serial(T.uint16(hidden // 128)):
                    col = vector * 128
                    x = load_f32(x_ub, row * hidden + col, 128)
                    dy = load_scaled_grad(out_grad_ub, row * hidden + col, 128)
                    if with_weight:
                        dy = V.vmul(dy, load_f32(weight_ub, col, 128), mask128)
                    total = V.vmula(total, V.vmul(x, rstd_value, mask128), dy, mask128)
                mean_value = V.vmuls(V.vcadd(total, mask128, reassoc=True), 1.0 / hidden, mask1)
                V.vstore(mean_value, mean_ub[row], mask1)
            T.simd.mem_bar('VST_VLD')
            for vector in T.serial(T.uint16(hidden // 128)):
                col = vector * 128
                dw = V.alloc_var(T.float32, size=128)
                if with_weight:
                    w = load_f32(weight_ub, col, 128)
                    dw = load_f32(weight_grad_ub, col, 128)
                for row in T.serial(T.int16(num_rows)):
                    inv_rms = V.vbrc(V.vload(rstd_ub[row], size=1), size=128, group=1)
                    mean = V.vbrc(V.vload(mean_ub[row], size=1), size=128, group=1)
                    x = load_f32(x_ub, row * hidden + col, 128)
                    dy = load_scaled_grad(out_grad_ub, row * hidden + col, 128)
                    x = V.vmul(x, inv_rms, mask128)
                    if with_weight:
                        dw = V.vmula(dw, x, dy, mask128)
                        dy = V.vmul(dy, w, mask128)
                    dx = V.vmul(V.vsub(dy, V.vmul(x, mean, mask128), mask128), inv_rms, mask128)
                    if with_residual_out_grad:
                        dx = V.vadd(dx, load_f32(x_grad_ub, row * hidden + col, 128), mask128)
                    if x_grad_ub.dtype == T.bfloat16:
                        V.vstore(V.vcvt(dx, 'bfloat16', rounding='R'), x_grad_ub[row * hidden + col], mask128)
                    else:
                        V.vstore(dx, x_grad_ub[row * hidden + col], mask128)
                if with_weight:
                    V.vstore(dw, weight_grad_ub[col], mask128)

    @T.macro
    def reduce_weight_grad(partial_ub, out_ub):
        for chunk in T.serial(T.uint16(block_k // vec_size)):
            total = V.alloc_var(T.float32, size=64)
            total = V.vbrc(T.float32(0.0), size=64)
            for core in T.serial(T.uint16(num_vec_cores)):
                total = V.vadd(total, V.vload(partial_ub[core, chunk * vec_size], size=64))
            if weight_grad_initialized:
                total = V.vadd(total, load_f32(out_ub, chunk * vec_size, 64))
            value = V.vcvt(total, 'bfloat16', rounding='R') if weight_grad_dtype == T.bfloat16 else total
            V.vstore(value, out_ub[chunk * vec_size])

    @T.prim_func
    def norm_backward_kernel_vmi(
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
                with T.SimdVF():
                    T.clear(weight_grad_ub)

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

    return norm_backward_kernel_vmi
