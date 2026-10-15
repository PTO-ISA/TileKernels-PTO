import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto')
def get_engram_fused_weight_kernel_vmi(
    hidden_size: int,
    hc_mult: int,
    num_vec_cores: int,
):
    vec_size = 64
    blk_d = 256
    num_stages = 3
    assert hidden_size % blk_d == 0

    num_tiles = hidden_size // blk_d

    @T.prim_func
    def engram_fused_weight_kernel_vmi(
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            hidden_ub = T.alloc_shared((blk_d,), T.bfloat16)
            embed_ub = T.alloc_shared((blk_d,), T.bfloat16)
            fused_ub = T.alloc_shared((blk_d,), T.float32)
            T.annotate_buffer_versions(
                {
                    hidden_ub: num_stages,
                    embed_ub: num_stages,
                    fused_ub: num_stages,
                }
            )

            for head, tile in T.Persistent(
                [hc_mult, num_tiles],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                offset = tile * blk_d
                T.copy(weight_hidden[head, offset : offset + blk_d], hidden_ub)
                T.copy(weight_embed[head, offset : offset + blk_d], embed_ub)

                with T.SimdVF():
                    for chunk in T.unroll(blk_d // vec_size, explicit=True):
                        chunk_offset = chunk * vec_size
                        hidden_f32 = V.vcvt(V.vload(hidden_ub[chunk_offset], size=vec_size), 'float32')
                        embed_f32 = V.vcvt(V.vload(embed_ub[chunk_offset], size=vec_size), 'float32')
                        V.vstore(V.vmul(hidden_f32, embed_f32), fused_ub[chunk_offset])

                T.copy(fused_ub, weight_fused[head, offset : offset + blk_d])

    return engram_fused_weight_kernel_vmi
