import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def mhc_sinkhorn_fwd_parallel(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """CUDA scalar equations with ASC buffers, persistent traversal and DMA."""
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 4

    with T.Kernel(num_vec_cores) as core_id:
        comb_in_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        comb_out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({comb_in_ub: num_stages, comb_out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(comb_res_mix[token_offset, 0, 0], comb_in_ub)

            with T.SimdVF(lanes=64):
                row_stat = T.alloc_shared((token_block_size, hidden_size), T.float32)
                col_stat = T.alloc_shared((token_block_size, hidden_size), T.float32)
                T.reduce_max(comb_in_ub, row_stat, dim=2)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    comb_out_ub[i, r, c] = T.exp(comb_in_ub[i, r, c] - row_stat[i, r])
                T.reduce_sum(comb_out_ub, row_stat, dim=2)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    comb_out_ub[i, r, c] = comb_out_ub[i, r, c] / row_stat[i, r] + eps
                T.reduce_sum(comb_out_ub, col_stat, dim=1)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    comb_out_ub[i, r, c] = comb_out_ub[i, r, c] / (col_stat[i, c] + eps)
                for step in T.serial(repeat - 1):
                    T.reduce_sum(comb_out_ub, row_stat, dim=2)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        comb_out_ub[i, r, c] = comb_out_ub[i, r, c] / (row_stat[i, r] + eps)
                    T.reduce_sum(comb_out_ub, col_stat, dim=1)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        comb_out_ub[i, r, c] = comb_out_ub[i, r, c] / (col_stat[i, c] + eps)

            T.copy(comb_out_ub, comb_res_mix_out[token_offset, 0, 0])


@tilelang.jit(target='pto')
def mhc_sinkhorn_bwd_parallel(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """CUDA scalar equations with ASC buffers, persistent traversal and DMA."""
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        grad_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        x_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        xs_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({grad_ub: num_stages, x_ub: num_stages, out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(grad_output[token_offset, 0, 0], grad_ub)
            T.copy(x[token_offset, 0, 0], x_ub)

            with T.SimdVF(lanes=64):
                row_stat = T.alloc_shared((token_block_size, hidden_size), T.float32)
                col_stat = T.alloc_shared((token_block_size, hidden_size), T.float32)
                products = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
                T.reduce_max(x_ub, row_stat, dim=2)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    out_ub[i, r, c] = T.exp(x_ub[i, r, c] - row_stat[i, r])
                T.reduce_sum(out_ub, row_stat, dim=2)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    normalized = out_ub[i, r, c] / row_stat[i, r]
                    xs_ub[0, i, r, c] = normalized
                    xs_ub[1, i, r, c] = normalized + eps
                    out_ub[i, r, c] = normalized + eps
                T.reduce_sum(out_ub, col_stat, dim=1)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    sums_ub[1, i, r, c] = col_stat[i, c]
                    out_ub[i, r, c] = xs_ub[1, i, r, c] / (col_stat[i, c] + eps)
                for step in T.serial(repeat - 1):
                    T.reduce_sum(out_ub, row_stat, dim=2)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        xs_ub[step * 2 + 2, i, r, c] = out_ub[i, r, c]
                        sums_ub[step * 2 + 2, i, r, c] = row_stat[i, r]
                        out_ub[i, r, c] = out_ub[i, r, c] / (row_stat[i, r] + eps)
                    T.reduce_sum(out_ub, col_stat, dim=1)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        xs_ub[step * 2 + 3, i, r, c] = out_ub[i, r, c]
                        sums_ub[step * 2 + 3, i, r, c] = col_stat[i, c]
                        out_ub[i, r, c] = out_ub[i, r, c] / (col_stat[i, c] + eps)
                for pair in T.serial(repeat - 1):
                    step = repeat * 2 - 1 - pair * 2
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        products[i, r, c] = grad_ub[i, r, c] * xs_ub[step, i, r, c]
                    T.reduce_sum(products, col_stat, dim=1)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        denom = sums_ub[step, i, r, c] + eps
                        grad_ub[i, r, c] = (grad_ub[i, r, c] - col_stat[i, c] / denom) / denom
                    step = repeat * 2 - 2 - pair * 2
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        products[i, r, c] = grad_ub[i, r, c] * xs_ub[step, i, r, c]
                    T.reduce_sum(products, row_stat, dim=2)
                    for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                        i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                        denom = sums_ub[step, i, r, c] + eps
                        grad_ub[i, r, c] = (grad_ub[i, r, c] - row_stat[i, r] / denom) / denom
                step = 1
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    products[i, r, c] = grad_ub[i, r, c] * xs_ub[step, i, r, c]
                T.reduce_sum(products, col_stat, dim=1)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    denom = sums_ub[step, i, r, c] + eps
                    grad_ub[i, r, c] = (grad_ub[i, r, c] - col_stat[i, c] / denom) / denom
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    products[i, r, c] = grad_ub[i, r, c] * xs_ub[0, i, r, c]
                T.reduce_sum(products, row_stat, dim=2)
                for q in T.Parallel(token_block_size * hidden_size * hidden_size):
                    i, r, c = q // (hidden_size * hidden_size), q // hidden_size % hidden_size, q % hidden_size
                    out_ub[i, r, c] = (grad_ub[i, r, c] - row_stat[i, r]) * xs_ub[0, i, r, c]

            T.copy(out_ub, grad_input[token_offset, 0, 0])
