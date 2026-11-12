import tilelang
from tilelang.ascend import language as T
from tile_kernels.config import get_max_ub_per_vector_core


@tilelang.jit(target='pto')
def get_cast_kernel_parallel(
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    stochastic_rounding: bool,
    n_hint: int,
    num_vec_cores: int,
):
    n = T.dynamic('n', T.int64)
    max_bytes = max(in_dtype.bytes, out_dtype.bytes)
    # 256B for one vec; 4 vecs needed because 4 numbers to hash.
    vec_len = 1024 // max_bytes
    block_alignment = 8
    reps = get_max_ub_per_vector_core(use_simt=False) // (2 * 1024) // block_alignment * block_alignment

    block_size = vec_len * reps
    half_size = block_size // 2
    half_reps = reps // 2

    # gen rbits hash for fp32->bf16 according to doc

    # stochastic round x1(0-255B), x2(256-512B) to bf16, using rbits. finite saturation not needed.

    # process 1024B elements a time -> vectors arith in hash is 256B wide

    # Convert half_size of fp32 values into bf16 and write into y_ub.
    @T.macro
    def convert_half(x_ub, y_ub, dest, base_id, bits):
        base_group = T.uint32(base_id // 4)
        with T.SimdVF(lanes=64):
            # Stage the input before the ASC in-place BF16 writeback. Each hash
            # uses the same four adjacent FP32 bit patterns as the CUDA kernel.
            for chunk in T.serial(half_reps):
                if stochastic_rounding:
                    for j in T.Parallel(vec_len):
                        bits[j] = T.reinterpret(T.float32(x_ub[chunk * vec_len + j]), T.uint32)
                    for j in T.Parallel(vec_len):
                        group = j // 4
                        h = bits[group * 4] * T.uint32(0x5671D42B) + bits[group * 4 + 1] * T.uint32(0x9995E499)
                        h = (
                            h
                            + bits[group * 4 + 2] * T.uint32(0xACE1B8A5)
                            + bits[group * 4 + 3] * T.uint32(0xE153538D)
                            + base_group
                            + T.uint32(chunk * (vec_len // 4))
                            + T.uint32(group)
                        )
                        h = h ^ (h >> 23)
                        h = h * T.uint32(0x7FEB352D)
                        h = h ^ (h >> 16)
                        r = h * T.if_then_else(j % 4 < 2, T.uint32(0x846CA68B), T.uint32(0xD35A2D97))
                        r = r ^ (r >> 11)
                        random16 = (r >> ((j % 2) * 16)) & T.uint32(65535)
                        value = bits[j]
                        rounded = (value >> 16) + T.if_then_else(random16 < (value & T.uint32(65535)), T.uint32(1), T.uint32(0))
                        y_ub[dest + chunk * vec_len + j] = T.cast(T.reinterpret(rounded << 16, T.float32), out_dtype)
                else:
                    for j in T.Parallel(vec_len):
                        bits[j] = T.reinterpret(T.float32(x_ub[chunk * vec_len + j]), T.uint32)
                    for j in T.Parallel(vec_len):
                        y_ub[dest + chunk * vec_len + j] = T.cast(T.reinterpret(bits[j], T.float32), out_dtype)

    # process a block: copyin x -> front_ub + back_ub; casted bf16 -> front_ub; (read next loop using back_ub); copyout front_ub -> y
    @T.macro
    def process_block(x, y, front_ub, back_ub, bits, base_id, valid_len):
        # bf16 is only half the size of fp32, so it fits in the frist half input (front_ub).
        output = T.view(front_ub, (block_size,), out_dtype)
        front_len = T.min(half_size, valid_len)
        back_len = valid_len - half_size

        T.copy(x[base_id], front_ub[:front_len])

        if back_len > 0:
            T.copy(x[base_id + half_size], back_ub[:back_len])
        convert_half(front_ub, output, 0, base_id, bits)
        if back_len > 0:
            convert_half(back_ub, output, half_size, base_id + half_size, bits)

        T.copy(output[:valid_len], y[base_id])

    @T.prim_func
    def cast_kernel_parallel(
        x: T.Tensor[(n,), in_dtype],
        y: T.Tensor[(n,), out_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            T.assume(n % n_hint == 0)
            # Two independent pairs of half-buffers, 240 KB in total.
            front0_ub = T.alloc_shared((half_size,), in_dtype)
            back0_ub = T.alloc_shared((half_size,), in_dtype)
            front1_ub = T.alloc_shared((half_size,), in_dtype)
            back1_ub = T.alloc_shared((half_size,), in_dtype)

            bits_ub = T.alloc_shared((vec_len,), T.uint32)
            T.annotate_buffer_versions({bits_ub: 1})

            full_groups = n // (block_size * 4)
            for pid in T.Persistent([full_groups], num_vec_cores, core_id, group_size=1, num_stages=1):
                base_id = T.int64(pid) * block_size * 4
                # Stage(0) here is to preserve order of operations in a pipeline
                with T.Stage(0):
                    process_block(x, y, front0_ub, back0_ub, bits_ub, base_id, block_size)
                    process_block(x, y, front1_ub, back1_ub, bits_ub, base_id + block_size, block_size)
                    # Swap roles: the released back buffers receive the front half of the next block.
                    process_block(x, y, back0_ub, front0_ub, bits_ub, base_id + block_size * 2, block_size)
                    process_block(x, y, back1_ub, front1_ub, bits_ub, base_id + block_size * 3, block_size)

            # move tail blocks handling outside the main loop to reduce ifs in the main loop.
            tail_blocks = T.ceildiv(n, block_size) - full_groups * 4
            for tail_id in T.serial(core_id, tail_blocks, num_vec_cores):
                base_id = (full_groups * 4 + T.int64(tail_id)) * block_size
                process_block(x, y, front0_ub, back0_ub, bits_ub, base_id, T.min(block_size, n - base_id))

    return cast_kernel_parallel
