import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def get_engram_grad_w_reduce_kernel_vmi(
    hidden_size: int,
    num_partials: int,
    hc_mult: int,
    num_vec_cores: int,
):
    vec_size = 64
    max_shards_per_head = 16
    assert hidden_size % vec_size == 0

    num_chunks = hidden_size // vec_size
    shards_per_head = max(divisor for divisor in range(1, min(max_shards_per_head, num_chunks) + 1) if num_chunks % divisor == 0)
    block_width = hidden_size // shards_per_head
    num_block_chunks = block_width // vec_size

    @T.prim_func
    def engram_grad_w_reduce_kernel_vmi(
        grad_w_partial: T.Tensor[(num_partials, hc_mult, hidden_size), T.float32],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float32],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            partial_ub = T.alloc_shared((num_partials, block_width), T.float32)
            weight_hidden_ub = T.alloc_shared((block_width,), T.bfloat16)
            weight_embed_ub = T.alloc_shared((block_width,), T.bfloat16)
            grad_hidden_ub = T.alloc_shared((block_width,), T.float32)
            grad_embed_ub = T.alloc_shared((block_width,), T.float32)

            for shard, head in T.Persistent(
                [shards_per_head, hc_mult],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=0,
            ):
                offset = shard * block_width

                T.copy(grad_w_partial[:, head, offset : offset + block_width], partial_ub)
                T.copy(weight_hidden[head, offset : offset + block_width], weight_hidden_ub)
                T.copy(weight_embed[head, offset : offset + block_width], weight_embed_ub)
                T.copy(grad_weight_hidden[head, offset : offset + block_width], grad_hidden_ub)
                T.copy(grad_weight_embed[head, offset : offset + block_width], grad_embed_ub)

                with T.SimdVF():
                    mask = V.create_mask(vec_size, size=vec_size)
                    f32 = V.vreg(vec_size, T.float32)
                    zero = V.vbrc(T.float32(0.0), size=vec_size)
                    accum = V.alloc_local((num_block_chunks,), f32)
                    for chunk in T.unroll(num_block_chunks, explicit=True):
                        accum[chunk] = zero
                    for partial in T.serial(num_partials):
                        for chunk in T.unroll(num_block_chunks, explicit=True):
                            accum[chunk] = V.vadd(accum[chunk], V.vload(partial_ub[partial, chunk * vec_size], size=vec_size))

                    grad_acc = V.alloc_local((2,), f32)
                    for chunk in T.unroll(num_block_chunks, explicit=True):
                        chunk_offset = chunk * vec_size
                        wh = V.vcvt(V.vload(weight_hidden_ub[chunk_offset], size=vec_size), 'float32')
                        we = V.vcvt(V.vload(weight_embed_ub[chunk_offset], size=vec_size), 'float32')
                        grad_acc[0] = V.vload(grad_hidden_ub[chunk_offset], size=vec_size)
                        grad_acc[1] = V.vload(grad_embed_ub[chunk_offset], size=vec_size)
                        grad_acc[0] = V.vmula(grad_acc[0], accum[chunk], we, mask)
                        grad_acc[1] = V.vmula(grad_acc[1], accum[chunk], wh, mask)
                        V.vstore(grad_acc[0], grad_hidden_ub[chunk_offset])
                        V.vstore(grad_acc[1], grad_embed_ub[chunk_offset])

                T.copy(grad_hidden_ub, grad_weight_hidden[head, offset : offset + block_width])
                T.copy(grad_embed_ub, grad_weight_embed[head, offset : offset + block_width])

    return engram_grad_w_reduce_kernel_vmi
