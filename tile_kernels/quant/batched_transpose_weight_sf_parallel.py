import tilelang
from tilelang.ascend import language as T

from tile_kernels.quant.common import get_packed_ue8m0_pack_factor


@tilelang.jit(target='pto')
def get_batched_transpose_weight_sf_kernel_parallel(block_size: int, num_vec_cores: int):
    assert block_size == 32
    vec_size = 64
    vec_size_b16 = 2 * vec_size
    num_sf_per_block = 32
    pack_factor = get_packed_ue8m0_pack_factor()
    assert pack_factor == 2
    weight_block_size = block_size * num_sf_per_block
    num_pack_per_block = num_sf_per_block // pack_factor
    num_stages = 2
    num_weights = T.dynamic('num_weights')

    @T.macro
    def pack_sf_block(sf_ub, sf0_out_ub, sf1_out_ub, invalid_sf_bits_ub, num_sf_rows, num_sf_cols):
        rows = num_sf_rows
        cols = num_sf_cols
        with T.SimdVF(lanes=64):
            exponents = T.alloc_shared((num_sf_per_block, vec_size), T.uint32)
            invalid = T.alloc_shared((num_sf_per_block, vec_size), T.int32)
            invalid_row = T.alloc_shared((num_sf_per_block,), T.int32)
            invalid_total = T.alloc_shared((1,), T.int32)
            for row, col in T.Parallel(num_sf_per_block, vec_size):
                bits = T.if_then_else((row < rows) & (col < cols), T.reinterpret(sf_ub[row, col], T.uint32), T.uint32(0))
                exponents[row, col] = bits >> 23
                invalid[row, col] = T.if_then_else((bits & T.uint32(0x807FFFFF)) != T.uint32(0), 1, 0)
            T.reduce_max(invalid, invalid_row, dim=1)
            T.reduce_max(invalid_row, invalid_total, dim=0)
            for point in T.Parallel(1):
                invalid_sf_bits_ub[point] = T.max(invalid_sf_bits_ub[point], T.uint32(invalid_total[point]))
            for row, col in T.Parallel(num_sf_per_block, num_pack_per_block):
                sf0_out_ub[row, col] = T.uint16(exponents[row, col * 2] | (exponents[row, col * 2 + 1] << 8))
            for row, col in T.Parallel(num_pack_per_block, num_sf_per_block):
                sf1_out_ub[row, col] = T.uint16(exponents[row * 2, col] | (exponents[row * 2 + 1, col] << 8))

    @T.macro
    def broadcast_sf_block(sf_out_ub, sf_store_ub, num_rows, num_cols, row_stride, col_stride):
        with T.SimdVF(lanes=64):
            for row, col in T.grid(num_rows, num_cols):
                for lane in T.Parallel(block_size):
                    sf_store_ub[row * row_stride + col * col_stride + lane] = sf_out_ub[row, col]

    @T.prim_func
    def batched_transpose_weight_sf_kernel(
        num_blocks: T.Tensor[(num_weights,), T.int32],
        weight_infos: T.Tensor[(num_weights, 3), T.int32],
        ptrs: T.Tensor[(num_weights, 3), T.ptr],
        num_tasks: T.int32,
    ):
        with T.Kernel(num_vec_cores) as core_id:
            sf_ub = T.alloc_shared((num_sf_per_block, vec_size), T.float32)
            sf0_out_ub = T.alloc_shared((num_sf_per_block, vec_size_b16), T.uint16)
            sf1_out_ub = T.alloc_shared((num_pack_per_block, vec_size_b16), T.uint16)
            sf0_store_ub = T.alloc_shared((num_pack_per_block * weight_block_size,), T.uint16)
            sf1_store_ub = T.alloc_shared((num_pack_per_block * weight_block_size,), T.uint16)
            invalid_sf_bits_ub = T.alloc_shared((1,), T.uint32)
            sf0_store_view = T.view(sf0_store_ub, (num_pack_per_block, weight_block_size))
            sf1_store_view = T.view(sf1_store_ub, (num_pack_per_block, weight_block_size))
            T.annotate_buffer_versions({invalid_sf_bits_ub: 1})
            with T.SimdVF(lanes=64):
                for point in T.Parallel(1):
                    invalid_sf_bits_ub[point] = T.uint32(0)

            weight_idx = T.alloc_var(T.int32, init=0)
            for task_id in T.Persistent([num_tasks], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
                while weight_idx + 1 < num_weights:
                    if num_blocks[weight_idx + 1] > task_id:
                        T.loop_break()
                    weight_idx = weight_idx + 1

                block_idx = T.alloc_var(T.int32, init=task_id - num_blocks[weight_idx])
                T.assume(0 <= block_idx)
                n = T.alloc_var(T.int32, init=weight_infos[weight_idx, 0])
                k = T.alloc_var(T.int32, init=weight_infos[weight_idx, 1])
                num_groups = T.alloc_var(T.int32, init=weight_infos[weight_idx, 2])
                T.assume(n > 0 and k > 0)
                num_block_n, num_block_k = T.ceildiv(n, weight_block_size), T.ceildiv(k, weight_block_size)
                num_block_per_group = num_block_n * num_block_k
                group = T.alloc_var(T.int32, init=block_idx // num_block_per_group)
                block_idx_in_group = block_idx % num_block_per_group
                pid_n = T.alloc_var(T.int32, init=block_idx_in_group // num_block_k)
                pid_k = T.alloc_var(T.int32, init=block_idx_in_group % num_block_k)
                num_sf_n = T.alloc_var(T.int32, init=n // block_size)
                num_sf_k = T.alloc_var(T.int32, init=k // block_size)
                num_packed_sf0 = T.alloc_var(T.int32, init=T.ceildiv(num_sf_k, pack_factor))
                num_packed_sf1 = T.alloc_var(T.int32, init=T.ceildiv(num_sf_n, pack_factor))
                sf_row, sf_col = pid_n * num_sf_per_block, pid_k * num_sf_per_block
                num_sf_rows = T.alloc_var(T.int32, init=T.min(num_sf_per_block, num_sf_n - sf_row))
                num_sf_cols = T.alloc_var(T.int32, init=T.min(num_sf_per_block, num_sf_k - sf_col))

                sf_inv = T.make_tensor(ptrs[weight_idx, 0], (num_groups, num_sf_n, num_sf_k), T.float32)
                sf0_out = T.make_tensor(ptrs[weight_idx, 1], (num_groups, num_packed_sf0, n), T.uint16)
                sf1_out = T.make_tensor(ptrs[weight_idx, 2], (num_groups, num_packed_sf1, k), T.uint16)

                T.copy(sf_inv[group, sf_row, sf_col], sf_ub[:, 0:num_sf_per_block])
                pack_sf_block(sf_ub, sf0_out_ub, sf1_out_ub, invalid_sf_bits_ub, num_sf_rows, num_sf_cols)

                broadcast_sf_block(sf0_out_ub, sf0_store_ub, num_sf_per_block, num_pack_per_block, block_size, weight_block_size)
                T.copy(sf0_store_view, sf0_out[group, pid_k * num_pack_per_block, pid_n * weight_block_size])

                broadcast_sf_block(sf1_out_ub, sf1_store_ub, num_pack_per_block, num_sf_per_block, weight_block_size, block_size)
                T.copy(sf1_store_view, sf1_out[group, pid_n * num_pack_per_block, pid_k * weight_block_size])

            T.device_assert(invalid_sf_bits_ub[0] == 0, no_stack_info=True)

    return batched_transpose_weight_sf_kernel
