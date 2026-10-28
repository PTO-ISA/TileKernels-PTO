"""CUDA-style element/reduction expressions with the ASC DMA schedule."""

import tilelang
from tilelang.ascend import language as T


@tilelang.jit(target='pto')
def get_normalize_weight_kernel_parallel(num_topk: int, num_vec_cores: int, use_tile_reduce: bool = False):
    num_tokens = T.dynamic('num_tokens')
    tile = 128
    num_stages = 4
    # Explicit-mask VMI stores require a 32-byte aligned row start.
    topk_stride = (num_topk + 7) // 8 * 8

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        denominator: T.Tensor[(num_tokens,), T.float32],
        normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            for tile_id in T.Persistent(
                [T.ceildiv(num_tokens, tile)], num_vec_cores, core_id,
                group_size=1, num_stages=num_stages,
            ):
                row = tile_id * tile
                valid_rows = T.min(tile, num_tokens - row)
                weights_ub = T.alloc_shared((tile, topk_stride), T.float32)
                norm_ub = T.alloc_shared((tile, topk_stride), T.float32)
                denom_ub = T.alloc_shared((tile,), T.float32)
                T.annotate_buffer_versions({weights_ub: num_stages, norm_ub: num_stages, denom_ub: num_stages})
                T.copy(topk_weights[row:row + valid_rows, :], weights_ub[:valid_rows, :num_topk])

                if use_tile_reduce:
                    with T.SimdVF(lanes=64):
                        T.reduce_sum(weights_ub[:, :num_topk], denom_ub, dim=1)
                        for i in T.Parallel(tile):
                            denom_ub[i] += 1e-20
                else:
                    # M2 compatibility path: a scalar reduction per row, still
                    # authored as a scalar-parallel VF. Separate VF scopes give
                    # the backend an ordering boundary before normalization.
                    # Use the tile reduction above once M3 is available.
                    for row_group in T.serial(tile // 8):
                        for row_lane in T.serial(8):
                            with T.SimdVF(lanes=64):
                                for lane in T.Parallel(8):
                                    total = T.alloc_var(T.float32, init=0.0)
                                    for k in T.serial(num_topk):
                                        total += weights_ub[row_group * 8 + row_lane, k]
                                    # Use an aligned base with exactly one
                                    # masked writer for this row's result.
                                    if lane == row_lane:
                                        denom_ub[row_group * 8 + lane] = total + 1e-20
                with T.SimdVF(lanes=64):
                    for i, j in T.Parallel(tile, topk_stride):
                        if j < num_topk:
                            norm_ub[i, j] = weights_ub[i, j] / denom_ub[i]

                T.copy(denom_ub[:valid_rows], denominator[row:row + valid_rows])
                T.copy(norm_ub[:valid_rows, :num_topk], normalized_weights[row:row + valid_rows, :])

    return normalize_weight_kernel
