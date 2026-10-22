import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V
from tile_kernels.config import get_max_ub_per_vector_core


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_cast_kernel_vmi(
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
    @T.macro
    def rbits_hash(vec, base_e):
        mask64 = V.create_mask(64, size=64)
        vec[0] = V.vmuls(vec[0], T.uint32(0x5671D42B), mask64)
        vec[1] = V.vmuls(vec[1], T.uint32(0x9995E499), mask64)
        vec[2] = V.vmuls(vec[2], T.uint32(0xACE1B8A5), mask64)
        vec[3] = V.vmuls(vec[3], T.uint32(0xE153538D), mask64)
        vec_e = V.vci(base_e, size=64)

        h = V.vadd(vec[0], vec_e)
        h = V.vadd(h, vec[1])
        vec[2] = V.vadd(vec[2], vec[3])
        h = V.vadd(h, vec[2])

        h = V.vxor(V.vshrs(h, 23, mask64), h)
        h = V.vmuls(h, T.uint32(0x7FEB352D), mask64)
        h = V.vxor(V.vshrs(h, 16, mask64), h)

        h1 = V.vmuls(h, T.uint32(0x846CA68B), mask64)
        h2 = V.vmuls(h, T.uint32(0xD35A2D97), mask64)
        h1 = V.vxor(V.vshrs(h1, 11, mask64), h1)
        h2 = V.vxor(V.vshrs(h2, 11, mask64), h2)

        return h1, h2

    # stochastic round x1(0-255B), x2(256-512B) to bf16, using rbits. finite saturation not needed.
    @T.macro
    def stochastic_round(x1, x2, rbits):
        mask128 = V.create_mask(128, size=128)
        lo, hi = V.vdintlv(V.vinterpret_cast(x1, 'uint16'), V.vinterpret_cast(x2, 'uint16'), mask128)
        should_carry = V.vcmp(V.vinterpret_cast(rbits, 'uint16'), lo, mask128, 'lt')
        carry = V.vsel(should_carry, V.vbrc(T.uint16(1), size=128), V.vbrc(T.uint16(0), size=128))
        result = V.vadd(hi, carry)
        return V.vinterpret_cast(result, 'bfloat16')

    # process 1024B elements a time -> vectors arith in hash is 256B wide
    @T.macro
    def process_chunk(in_chunk, tmp, base_e):
        mask64 = V.create_mask(64, size=64)
        tmp[0], tmp[1] = V.vdintlv(in_chunk[0], in_chunk[1], mask64)
        tmp[2], tmp[3] = V.vdintlv(in_chunk[2], in_chunk[3], mask64)
        tmp[0], tmp[2] = V.vdintlv(tmp[0], tmp[2], mask64)
        tmp[1], tmp[3] = V.vdintlv(tmp[1], tmp[3], mask64)
        h1, h2 = rbits_hash(tmp, base_e)
        h_front, h_back = V.vintlv(h1, h2, mask64)
        res1 = stochastic_round(in_chunk[0], in_chunk[1], h_front)
        res2 = stochastic_round(in_chunk[2], in_chunk[3], h_back)
        return res1, res2

    # Convert half_size of fp32 values into bf16 and write into y_ub.
    @T.macro
    def convert_half(x_ub, y_ub, dest, base_id):
        with T.SimdVF():
            mask64 = V.create_mask(64, size=64)
            mask128 = V.create_mask(128, size=128)
            in_chunk = V.alloc_local((4,), V.vreg(64, T.uint32))
            tmp = V.alloc_local((4,), V.vreg(64, T.uint32))
            for i in T.serial(half_reps):
                offset = i * vec_len
                for j in T.unroll(4, explicit=True):
                    in_chunk[j] = V.vinterpret_cast(V.vload(x_ub[offset + j * vec_len // 4], size=64), 'uint32')
                res1, res2 = process_chunk(in_chunk, tmp, T.uint32((base_id + offset) // 4))
                V.vstore(res1, y_ub[dest + offset], mask128)
                V.vstore(res2, y_ub[dest + offset + vec_len // 2], mask128)

    # process a block: copyin x -> front_ub + back_ub; casted bf16 -> front_ub; (read next loop using back_ub); copyout front_ub -> y
    @T.macro
    def process_block(x, y, front_ub, back_ub, base_id, valid_len):
        # bf16 is only half the size of fp32, so it fits in the frist half input (front_ub).
        output = T.view(front_ub, (block_size,), out_dtype)
        front_len = T.min(half_size, valid_len)
        back_len = valid_len - half_size

        T.copy(x[base_id], front_ub[:front_len])

        if back_len > 0:
            T.copy(x[base_id + half_size], back_ub[:back_len])
        convert_half(front_ub, output, 0, base_id)
        if back_len > 0:
            convert_half(back_ub, output, half_size, base_id + half_size)

        T.copy(output[:valid_len], y[base_id])

    @T.prim_func
    def cast_kernel_vmi(
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

            full_groups = n // (block_size * 4)
            for pid in T.Persistent([full_groups], num_vec_cores, core_id, group_size=1, num_stages=1):
                base_id = T.int64(pid) * block_size * 4
                # Stage(0) here is to preserve order of operations in a pipeline
                with T.Stage(0):
                    process_block(x, y, front0_ub, back0_ub, base_id, block_size)
                    process_block(x, y, front1_ub, back1_ub, base_id + block_size, block_size)
                    # Swap roles: the released back buffers receive the front half of the next block.
                    process_block(x, y, back0_ub, front0_ub, base_id + block_size * 2, block_size)
                    process_block(x, y, back1_ub, front1_ub, base_id + block_size * 3, block_size)

            # move tail blocks handling outside the main loop to reduce ifs in the main loop.
            tail_blocks = T.ceildiv(n, block_size) - full_groups * 4
            for tail_id in T.serial(core_id, tail_blocks, num_vec_cores):
                base_id = (full_groups * 4 + T.int64(tail_id)) * block_size
                process_block(x, y, front0_ub, back0_ub, base_id, T.min(block_size, n - base_id))

    return cast_kernel_vmi
