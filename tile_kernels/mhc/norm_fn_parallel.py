"""T.Parallel VF bodies; scheduling and DMA copied from norm_fn_asc.py."""

import tilelang
from tilelang.ascend import language as T
from tile_kernels.config import get_max_ub_per_vector_core

_COMPACT_REDUCE_TOKEN_BLOCK = 16
_COMPACT_REDUCE_SPLIT_CHUNK = 32
_COMPACT_REDUCE_MHC_MULT3 = 24
_COMPACT_REDUCE_VREGS = 6


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_fwd_parallel(
    fn,
    normw,
    out_fn,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):

    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn: T.Tensor[(mhc_mult3, hidden), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)

            with T.SimdVF(lanes=64):
                for row, j in T.Parallel(mhc_mult3, hidden_block):
                    out_fn_ub[row, j] = fn_ub[row, j] * normw_ub[j]

            T.copy(out_fn_ub, out_fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])


@tilelang.jit(target='pto')
def mhc_fn_normw_merge_bwd_parallel(
    fn,
    normw,
    out_fn_grad,
    fn_grad,
    normw_grad,
    mhc_mult3: int,
    hidden_block: int,
    num_stages: int,
    num_vec_cores: int,
):

    hidden = T.const('hidden')
    fn: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw: T.Tensor[(hidden,), T.float32]
    out_fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    fn_grad: T.Tensor[(mhc_mult3, hidden), T.float32]
    normw_grad: T.Tensor[(hidden,), T.float32]

    assert hidden % hidden_block == 0, 'Ascend fn/normw merge requires hidden divisible by hidden_block'
    assert hidden_block % 128 == 0, 'Ascend fn/normw merge SIMD path requires hidden_block divisible by 128'
    assert (3 * mhc_mult3 + 2) * hidden_block * 4 * num_stages <= get_max_ub_per_vector_core(use_simt=False), (
        'merge backward multi-buffer allocation exceeds UB capacity'
    )

    num_hidden_blocks = hidden // hidden_block

    with T.Kernel(num_vec_cores) as core_id:
        fn_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_ub = T.alloc_shared((hidden_block,), T.float32)
        out_fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        fn_grad_ub = T.alloc_shared((mhc_mult3, hidden_block), T.float32)
        normw_grad_ub = T.alloc_shared((hidden_block,), T.float32)
        T.annotate_buffer_versions(
            {
                fn_ub: num_stages,
                normw_ub: num_stages,
                out_fn_grad_ub: num_stages,
                fn_grad_ub: num_stages,
                normw_grad_ub: num_stages,
            }
        )

        for block_id in T.Persistent([num_hidden_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            hidden_offset = block_id * hidden_block
            T.copy(fn[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_ub)
            T.copy(normw[hidden_offset : hidden_offset + hidden_block], normw_ub)
            T.copy(out_fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], out_fn_grad_ub)
            T.copy(fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block], fn_grad_ub)
            T.copy(normw_grad[hidden_offset : hidden_offset + hidden_block], normw_grad_ub)

            with T.SimdVF(lanes=64):
                for row, j in T.Parallel(mhc_mult3, hidden_block):
                    fn_grad_ub[row, j] += out_fn_grad_ub[row, j] * normw_ub[j]
                for j in T.Parallel(hidden_block):
                    acc = T.alloc_var(T.float32, init=normw_grad_ub[j])
                    for row in T.serial(mhc_mult3):
                        acc += out_fn_grad_ub[row, j] * fn_ub[row, j]
                    normw_grad_ub[j] = acc

            T.copy(fn_grad_ub, fn_grad[:mhc_mult3, hidden_offset : hidden_offset + hidden_block])
            T.copy(normw_grad_ub, normw_grad[hidden_offset : hidden_offset + hidden_block])


def _compact_reduce_split_chunk(n_splits: int) -> int:
    for split_chunk in (_COMPACT_REDUCE_SPLIT_CHUNK, 16, 8, 4, 2):
        if n_splits % split_chunk == 0:
            return split_chunk
    return 1


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_fwd_parallel(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
    num_vec_cores: int,
):

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm fwd currently supports one RMS group'
    assert mhc_mult3 == _COMPACT_REDUCE_MHC_MULT3, 'Ascend compact reduce assumes mhc_mult3=24'

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    token_block = _COMPACT_REDUCE_TOKEN_BLOCK
    split_chunk = _compact_reduce_split_chunk(n_splits)
    compact_vregs = _COMPACT_REDUCE_VREGS
    sq_pad64 = 64
    num_stages = 3
    n_token_blocks = T.ceildiv(num_tokens, token_block)
    n_split_chunks = n_splits // split_chunk
    assert n_splits % split_chunk == 0, 'compact reduce requires n_splits divisible by selected split_chunk'

    with T.Kernel(num_vec_cores) as core_id:
        mul_ub = T.alloc_shared((split_chunk, token_block, mhc_mult3), T.float32)
        sq_ub = T.alloc_shared((split_chunk, sq_pad64), T.float32)
        acc_ub = T.alloc_shared((compact_vregs, 64), T.float32)
        sq_acc_ub = T.alloc_shared(64, T.float32)
        mul_out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        out_ub = T.alloc_shared((token_block, mhc_mult3), T.float32)
        sq_out_ub = T.alloc_shared(64, T.float32)
        rms_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                mul_ub: num_stages,
                sq_ub: num_stages,
                acc_ub: 1,
                sq_acc_ub: 1,
                mul_out_ub: 1,
                out_ub: 1,
                sq_out_ub: 1,
                rms_ub: 1,
            }
        )

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            # Ragged tail blocks are handled by the same vectorized path:
            # T.copy clamps out-of-bound GM reads/writes, and the per-token
            # lane masks keep the padded tokens' stale UB lanes isolated,
            # so no separate scalar tail path is needed.
            with T.SimdVF(lanes=64):
                for i, j in T.Parallel(compact_vregs, 64):
                    acc_ub[i, j] = 0.0
                for j in T.Parallel(64):
                    sq_acc_ub[j] = 0.0

            for split_chunk_idx in T.serial(n_split_chunks):
                split_base = split_chunk_idx * split_chunk
                T.copy(
                    out_mul_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0, :mhc_mult3],
                    mul_ub,
                )
                T.copy(
                    sqrsum_splitted[split_base : split_base + split_chunk, token_base : token_base + token_block, 0],
                    sq_ub[:, :token_block],
                )

                with T.SimdVF(lanes=64):
                    for i, j in T.Parallel(compact_vregs, 64):
                        acc = T.alloc_var(T.float32, init=acc_ub[i, j])
                        for s in T.serial(split_chunk):
                            acc += mul_ub[s, (i * 64 + j) // mhc_mult3, (i * 64 + j) % mhc_mult3]
                        acc_ub[i, j] = acc
                    for j in T.Parallel(token_block):
                        acc = T.alloc_var(T.float32, init=sq_acc_ub[j])
                        for s in T.serial(split_chunk):
                            acc += sq_ub[s, j]
                        sq_acc_ub[j] = acc

            with T.SimdVF(lanes=64):
                for i in T.Parallel(token_block):
                    sq_out_ub[i] = sq_acc_ub[i]
                    rms_ub[i] = 1.0 / T.sqrt(sq_acc_ub[i] * (1.0 / rms_group_size) + rms_eps)
            with T.SimdVF(lanes=64):
                acc_view = T.reshape(acc_ub, (token_block, mhc_mult3))
                for i, j in T.Parallel(token_block, mhc_mult3):
                    mul_out_ub[i, j] = acc_view[i, j]
                    out_ub[i, j] = acc_view[i, j] * rms_ub[i]

            T.copy(mul_out_ub, out_mul[token_base : token_base + token_block, 0, :])
            T.copy(out_ub, out[token_base : token_base + token_block, :])
            T.copy(sq_out_ub[:token_block], sqrsum[token_base : token_base + token_block, 0])


@tilelang.jit(target='pto')
def mhc_reduce_partials_and_rmsnorm_bwd_parallel(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
    num_vec_cores: int,
):

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm bwd currently supports one RMS group'
    assert mhc_mult3 <= 32, f'Ascend reduce/rmsnorm bwd SimtVF maps mhc_mult3 to one 32-lane vector, got {mhc_mult3}'

    out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32]

    token_block = 16
    mhc_mult3_pad32 = 32
    num_stages = 6
    n_token_blocks = T.ceildiv(num_tokens, token_block)

    with T.Kernel(num_vec_cores) as core_id:
        out_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        out_mul_grad_ub = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
        sqrsum_ub = T.alloc_shared(token_block, T.float32)
        sqrsum_grad_ub = T.alloc_shared((token_block, 8), T.float32)
        sqrsum_grad_compact_ub = T.alloc_shared(64, T.float32)
        T.annotate_buffer_versions(
            {
                out_grad_ub: num_stages,
                out_mul_ub: num_stages,
                out_mul_grad_ub: num_stages,
                sqrsum_ub: num_stages,
                sqrsum_grad_ub: num_stages,
                sqrsum_grad_compact_ub: num_stages,
            }
        )

        for block_id in T.Persistent([n_token_blocks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            token_base = block_id * token_block
            T.copy(out_grad[token_base : token_base + token_block, :mhc_mult3], out_grad_ub[:, :mhc_mult3])
            T.copy(out_mul[token_base : token_base + token_block, 0, :mhc_mult3], out_mul_ub[:, :mhc_mult3])
            T.copy(sqrsum[token_base : token_base + token_block, 0], sqrsum_ub[:token_block])

            with T.SimdVF(lanes=64):
                products = T.alloc_shared((token_block, mhc_mult3_pad32), T.float32)
                for i, j in T.Parallel(token_block, mhc_mult3_pad32):
                    rms = 1.0 / T.sqrt(sqrsum_ub[i] * (1.0 / rms_group_size) + rms_eps)
                    out_mul_grad_ub[i, j] = out_grad_ub[i, j] * rms
                    products[i, j] = T.if_then_else(j < mhc_mult3, out_grad_ub[i, j] * out_mul_ub[i, j], 0.0)
                T.reduce_sum(products, sqrsum_grad_compact_ub[:token_block], dim=1)
                for i in T.Parallel(token_block):
                    rms = 1.0 / T.sqrt(sqrsum_ub[i] * (1.0 / rms_group_size) + rms_eps)
                    sqrsum_grad_compact_ub[i] = sqrsum_grad_compact_ub[i] * rms / (sqrsum_ub[i] + rms_eps * rms_group_size) * -0.5

            T.copy(out_mul_grad_ub[:, :mhc_mult3], out_mul_grad[token_base : token_base + token_block, 0, :mhc_mult3])
            T.copy(sqrsum_grad_compact_ub[:token_block], sqrsum_grad[token_base : token_base + token_block, 0])
