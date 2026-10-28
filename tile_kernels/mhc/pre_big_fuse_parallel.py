import tilelang
from tilelang.ascend import language as T


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit(target='pto')
def mhc_pre_big_fuse_fwd_parallel(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    h_blk: int,
    n_splits: int,
    token_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    """Ascend fused forward: reduce partials + RMSNorm, split mixes, Sinkhorn, pre-apply.

    Mirrors the arithmetic of the individual Ascend reference kernels
    (mhc_reduce_partials_and_rmsnorm_fwd_asc, mhc_pre_split_mixes_fwd_simd_mhc4_asc,
    mhc_sinkhorn_fwd_asc, mhc_pre_apply_mix_fwd_asc) op-for-op so the fused result
    follows the composed path, with scalar expressions lowered to PTO VMI.

    The Ascend HC prenorm GEMM already reduces split-K internally, so only
    n_splits=1 is supported here.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden = T.const('hidden')
    mhc = 4
    mhc2 = mhc * mhc
    mhc3 = mhc * 2 + mhc2
    assert mhc3 == 24
    assert n_splits == 1, 'Ascend pre_big_fuse expects the GEMM to reduce split-K internally (n_splits=1)'
    assert h_blk % _VEC == 0 and hidden % h_blk == 0

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc3,), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_mix: T.Tensor[(num_tokens, mhc), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc2), T.float32]
    layer_input: T.Tensor[(num_tokens, hidden), T.bfloat16]

    tb = token_block
    num_hidden_tiles = hidden // h_blk
    num_chunks = h_blk // _VEC
    # Same compile-time fp32 constant as the reference reduce kernel.
    inv_rms_group_size = T.float32(1.0 / (mhc * hidden))

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc3, 8), T.float32)
        mul_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        mixes_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        sq_ub = T.alloc_shared((tb,), T.float32)
        pre_ub = T.alloc_shared((tb, mhc), T.float32)
        post_ub = T.alloc_shared((tb, mhc), T.float32)
        comb_in_ub = T.alloc_shared((tb, mhc2), T.float32)
        comb_out_ub = T.alloc_shared((tb, mhc2), T.float32)
        x_ub = T.alloc_shared((tb, mhc, h_blk), T.bfloat16)
        layer_input_ub = T.alloc_shared((tb, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                mul_flat_ub: num_stages,
                mixes_flat_ub: num_stages,
                sq_ub: num_stages,
                pre_ub: num_stages,
                post_ub: num_stages,
                comb_in_ub: num_stages,
                comb_out_ub: num_stages,
                x_ub: num_stages,
                layer_input_ub: num_stages,
            }
        )

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc3):
                base_ub[i, 0] = mhc_base[i]

        for pid in T.Persistent(
            [T.ceildiv(num_tokens, tb)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            token_base = pid * tb
            T.copy(gemm_out_mul[0:1, token_base : token_base + tb, 0:mhc3], T.view(mul_flat_ub, (tb, mhc3)))
            T.copy(gemm_out_sqrsum[0:1, token_base : token_base + tb], sq_ub)

            ##################################################################
            # _mhc_reduce_partials_and_rmsnorm_fwd + _mhc_pre_split_mixes_fwd
            # + _mhc_sinkhorn_fwd in a single SimdVF phase: the whole latency
            # path is one unit so the scheduler can overlap it with the
            # pre-apply memory stream of the previous token block.
            with T.SimdVF(lanes=64):
                row_sum = T.alloc_shared((tb, mhc), T.float32)
                col_sum = T.alloc_shared((tb, mhc), T.float32)
                for row, j in T.Parallel(tb, mhc3):
                    inverse = 1.0 / T.sqrt(sq_ub[row] * inv_rms_group_size + rms_eps)
                    mixes_flat_ub[row * mhc3 + j] = mul_flat_ub[row * mhc3 + j] * inverse
                for flat in T.Parallel(tb * mhc):
                    row = flat // mhc
                    j = flat % mhc
                    pre = mixes_flat_ub[row * mhc3 + j] * scale_ub[0, 0] + base_ub[j, 0]
                    post = mixes_flat_ub[row * mhc3 + j + mhc] * scale_ub[1, 0] + base_ub[j + mhc, 0]
                    pre_ub[row, j] = 1.0 / (1.0 + T.exp(-pre)) + mhc_pre_eps
                    post_ub[row, j] = mhc_post_mult_value / (1.0 + T.exp(-post))
                for flat in T.Parallel(tb * mhc2):
                    row = flat // mhc2
                    j = flat % mhc2
                    comb_in_ub[row, j] = mixes_flat_ub[row * mhc3 + j + 2 * mhc] * scale_ub[2, 0] + base_ub[j + 2 * mhc, 0]
                T.reduce_max(T.reshape(comb_in_ub, (tb, mhc, mhc)), row_sum, dim=2)
                for i in T.Parallel(tb * mhc2):
                    comb_out_ub[i // mhc2, i % mhc2] = T.exp(comb_in_ub[i // mhc2, i % mhc2] - row_sum[i // mhc2, (i // mhc) % mhc])
                T.reduce_sum(T.reshape(comb_out_ub, (tb, mhc, mhc)), row_sum, dim=2)
                for i in T.Parallel(tb * mhc2):
                    comb_out_ub[i // mhc2, i % mhc2] = comb_out_ub[i // mhc2, i % mhc2] / row_sum[i // mhc2, (i // mhc) % mhc] + mhc_sinkhorn_eps
                T.reduce_sum(T.reshape(comb_out_ub, (tb, mhc, mhc)), col_sum, dim=1)
                for i in T.Parallel(tb * mhc2):
                    comb_out_ub[i // mhc2, i % mhc2] = comb_out_ub[i // mhc2, i % mhc2] / (col_sum[i // mhc2, i % mhc] + mhc_sinkhorn_eps)
                for step in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(T.reshape(comb_out_ub, (tb, mhc, mhc)), row_sum, dim=2)
                    for i in T.Parallel(tb * mhc2):
                        comb_out_ub[i // mhc2, i % mhc2] = comb_out_ub[i // mhc2, i % mhc2] / (
                            row_sum[i // mhc2, (i // mhc) % mhc] + mhc_sinkhorn_eps
                        )
                    T.reduce_sum(T.reshape(comb_out_ub, (tb, mhc, mhc)), col_sum, dim=1)
                    for i in T.Parallel(tb * mhc2):
                        comb_out_ub[i // mhc2, i % mhc2] = comb_out_ub[i // mhc2, i % mhc2] / (col_sum[i // mhc2, i % mhc] + mhc_sinkhorn_eps)

            T.copy(post_ub[:, 0:mhc], post_mix[token_base : token_base + tb, 0:mhc])
            T.copy(comb_out_ub, comb_mix[token_base : token_base + tb, 0:mhc2])

            ##################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(num_hidden_tiles, num_stages=2):
                T.copy(
                    residual[
                        token_base : token_base + tb,
                        0:mhc,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                    x_ub,
                )
                with T.SimdVF(lanes=64):
                    for row, j in T.Parallel(tb, h_blk):
                        acc = T.alloc_var(T.float32, init=0.0)
                        for m in T.serial(mhc):
                            acc = acc + T.float32(x_ub[row, m, j]) * pre_ub[row, m]
                        layer_input_ub[row, j] = T.cast(acc, T.bfloat16)

                T.copy(
                    layer_input_ub,
                    layer_input[
                        token_base : token_base + tb,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                )
