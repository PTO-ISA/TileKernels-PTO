import tilelang
from tilelang.ascend import language as T


@T.macro
def _sigmoid_pto(x):
    return T.float32(1.0) / (T.float32(1.0) + T.exp(-x))


def _tile_rows_pto(token_block_size: int) -> int:
    return max(token_block_size, 8)


def _fold_strides_pto(token_block_elems: int, mhc_mult: int) -> list[int]:
    strides = []
    stride = token_block_elems // 2
    while stride >= mhc_mult:
        strides.append(stride)
        stride //= 2
    return strides


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

    token_block_size = _tile_rows_pto(token_block_size)
    token_block_elems = token_block_size * mhc_mult

    with T.Kernel(num_vec_cores) as core_id:
        mhc_scale_ub = T.alloc_shared((1,), T.float32)
        mhc_base_ub = T.alloc_shared((mhc_mult,), T.float32)
        mhc_base_rep_ub = T.alloc_shared((token_block_elems,), T.float32)
        T.copy(mhc_scale, mhc_scale_ub)
        T.copy(mhc_base, mhc_base_ub)
        with T.SimdVF(lanes=256):
            for i, j in T.Parallel(token_block_size, mhc_mult):
                mhc_base_rep_ub[i * mhc_mult + j] = mhc_base_ub[j]

        for pid in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=2,
        ):
            input_ub = T.alloc_shared((token_block_elems,), T.float32)
            output_mix_ub = T.alloc_shared((token_block_elems,), T.float32)
            T.annotate_buffer_versions({
                input_ub: (2, 'iteration'),
                output_mix_ub: (2, 'iteration'),
            })

            row = pid * token_block_size
            valid_rows = T.min(token_block_size, num_tokens - row)
            valid_elems = valid_rows * mhc_mult
            T.copy(input_mix[row : row + valid_rows, :], input_ub[:valid_elems])

            with T.SimdVF(lanes=256):
                for k in T.Parallel(token_block_elems):
                    output_mix_ub[k] = _sigmoid_pto(input_ub[k] * mhc_scale_ub[0] + mhc_base_rep_ub[k]) + T.float32(mhc_pre_eps)

            T.copy(output_mix_ub[:valid_elems], output_mix[row : row + valid_rows, :], l2_cache_ctrl='NORMAL_FV')


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

    token_block_size = _tile_rows_pto(token_block_size)
    token_block_elems = token_block_size * mhc_mult

    with T.Kernel(num_vec_cores) as pid:
        mhc_scale_ub = T.alloc_shared((1,), T.float32)
        mhc_base_ub = T.alloc_shared((mhc_mult,), T.float32)
        mhc_base_rep_ub = T.alloc_shared((token_block_elems,), T.float32)
        mhc_scale_grad_acc_ub = T.alloc_shared((token_block_elems,), T.float32)
        mhc_base_grad_acc_ub = T.alloc_shared((token_block_elems,), T.float32)
        mhc_scale_grad_fold_ub = T.alloc_shared((mhc_mult,), T.float32)
        mhc_scale_grad_final_ub = T.alloc_shared((1,), T.float32)
        mhc_base_grad_final_ub = T.alloc_shared((mhc_mult,), T.float32)
        T.copy(mhc_scale, mhc_scale_ub)
        T.copy(mhc_base, mhc_base_ub)
        with T.SimdVF(lanes=256):
            for k in T.Parallel(token_block_elems):
                mhc_scale_grad_acc_ub[k] = T.float32(0.0)
                mhc_base_grad_acc_ub[k] = T.float32(0.0)
            for i, j in T.Parallel(token_block_size, mhc_mult):
                mhc_base_rep_ub[i * mhc_mult + j] = mhc_base_ub[j]

        input_ub = T.alloc_shared((token_block_elems,), T.float32)
        output_mix_grad_ub = T.alloc_shared((token_block_elems,), T.float32)
        input_mix_grad_ub = T.alloc_shared((token_block_elems,), T.float32)
        with T.SimdVF(lanes=256):
            for k in T.Parallel(token_block_elems):
                input_ub[k] = T.float32(0.0)
                output_mix_grad_ub[k] = T.float32(0.0)

        for t in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            pid,
            group_size=1,
            num_stages=1,
        ):
            tile_id = T.ceildiv(num_tokens, token_block_size) - 1 - t
            row = tile_id * token_block_size
            valid_rows = T.min(token_block_size, num_tokens - row)
            valid_elems = valid_rows * mhc_mult
            T.copy(input_mix[row : row + valid_rows, :], input_ub[:valid_elems])
            T.copy(output_mix_grad[row : row + valid_rows, :], output_mix_grad_ub[:valid_elems])

            with T.SimdVF(lanes=256):
                for k in T.Parallel(token_block_elems):
                    input_value = input_ub[k]
                    input_recompute = _sigmoid_pto(input_value * mhc_scale_ub[0] + mhc_base_rep_ub[k])
                    grad = input_recompute * (T.float32(1.0) - input_recompute) * output_mix_grad_ub[k]
                    input_mix_grad_ub[k] = grad * mhc_scale_ub[0]
                    mhc_scale_grad_acc_ub[k] = mhc_scale_grad_acc_ub[k] + grad * input_value
                    mhc_base_grad_acc_ub[k] = mhc_base_grad_acc_ub[k] + grad

            T.copy(input_mix_grad_ub[:valid_elems], input_mix_grad[row : row + valid_rows, :], l2_cache_ctrl='NORMAL_FV')

        with T.SimdVF(lanes=256):
            for stride in _fold_strides_pto(token_block_elems, mhc_mult):
                for k in T.Parallel(stride):
                    mhc_scale_grad_acc_ub[k] = mhc_scale_grad_acc_ub[k] + mhc_scale_grad_acc_ub[k + stride]
                    mhc_base_grad_acc_ub[k] = mhc_base_grad_acc_ub[k] + mhc_base_grad_acc_ub[k + stride]
            for k in T.Parallel(mhc_mult):
                mhc_scale_grad_fold_ub[k] = mhc_scale_grad_acc_ub[k]
                mhc_base_grad_final_ub[k] = mhc_base_grad_acc_ub[k]
            T.reduce_sum(mhc_scale_grad_fold_ub, mhc_scale_grad_final_ub, dim=0)
        T.copy(mhc_scale_grad_final_ub, mhc_scale_grad_partial[pid, :], l2_cache_ctrl='NORMAL_FV')
        T.copy(mhc_base_grad_final_ub, mhc_base_grad_partial[pid, :], l2_cache_ctrl='NORMAL_FV')
