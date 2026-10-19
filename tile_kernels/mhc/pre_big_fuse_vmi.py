"""VMI port of the mhc pre_big_fuse kernel (discussion #1616).

The kernel is the ASC original (``mhc_pre_big_fuse_fwd_asc``) with only the
``T.SimdVF()`` bodies rewritten in logical VMI vectors; the outer schedule
(T.Kernel / T.Persistent / T.copy / UB layout / num_stages / h_blk) and the
host contract are kept verbatim from ``pre_big_fuse_asc.py`` so ASC and VMI
stay directly comparable.

Key ASC -> VMI correspondences used here (same base set as the
norm_fn / post / pre_apply_mix ports):

- ``S.pset(32)`` (a full fp32 mask in CCE terms) becomes
  ``V.create_mask(64, size=64)``; elementwise ops on full 64-lane vectors run
  unmasked and only the always-masked ops (vmula, vexpdif, vcmp/vcmps, vsel,
  vcadd, vgather, vshls/vshrs) carry it.
- ``S.vld(scalar_ub[i], dist='BRC_B32')`` becomes
  ``V.vbrc(V.vload(scalar_ub[i], size=1), size=64)``: PTOAS removed the vload
  broadcast dist_mode, so a scalar broadcast is a one-element vload followed
  by vbrc.
- ``S.vdup(c, T.float32)`` / ``S.vadds`` / ``S.vmuls`` become
  ``V.vbrc(T.float32(c), size=64)`` combined with the plain vector op.
- ``S.vsts(dst, v, mask, 'NORM_B32')`` becomes ``V.vstore(v, dst)`` (value
  first, address second); the full-width store needs no mask.
- ``S.vgather2(buf, u32_index, mask)`` becomes
  ``V.vgather(buf, V.vinterpret_cast(index, 'uint32'), mask_full)``.
- ``S.vdiv`` defaults to the precise (0-ULP) CANN division because tilelang
  runs with ``tl.enable_fast_math`` off, while plain ``V.vdiv`` is the bare
  SFU instruction; every divide therefore carries ``precision='exact'``
  (with the mandatory explicit mask).  Without it the kernel still runs but
  drifts 1-2 ULP on post_mix within one sigmoid and up to ~8 bits on
  comb_mix after the 10 Sinkhorn iterations.

Sinkhorn-specific rewrite (the interesting part):

- The ASC row reduce/max (``vcgadd/vcgmax`` twice with the complementary
  4-of-8 masks ``mask_lo``/``mask_hi`` + two ``vselr(bcast_idx2)`` + ``vsel``)
  splits each 8-fp32 VLane, which spans two matrix rows, into its even/odd
  row halves because the reduction hardware is 8-lane aligned.  VMI's
  ``vcadd``/``vcmax`` have the same alignment (``group`` counts result lanes
  and only 1/2/4/8 are legal), so the port keeps the two-pass masked
  structure but expresses every step logically:

    rsum = V.vsel(mask_lo, V.vbrc(V.vcadd(mat, mask_lo, group=8, reassoc=True), size=64, group=8),
                          V.vbrc(V.vcadd(mat, mask_hi, group=8, reassoc=True), size=64, group=8))

  ``vbrc(..., group=8)`` block-broadcasts each reduced row over its own
  8-lane half, which is the logical equivalent of the ASC ``vselr`` with the
  ``bcast_idx2`` index vector (no index arithmetic needed).
  ``reassoc=True`` keeps the pairwise reduction tree the ASC ``vcgadd`` form
  implies.
- The 4x4 in-register transpose stays ``V.vselr(mat, t_idx)`` with the same
  per-lane index arithmetic, since the lane permutation is logical data
  movement, not a physical layout patch.
- The ASC plain ``range()`` loops that carry loop-dependent int broadcast
  scalars (``vbrc(T.int32(vec_id * ...))``) are spelled ``T.unroll(...,
  explicit=True)``: with a TIR loop the PTODSL printer drops the int32 typing
  of those scalars and ptoas sees ``vreg<64xindex>``.

Wide vectors where ASC chunked only for the physical 64-fp32 register --
the legal VMI widths {64, 128, 256} let the same elementwise math run in
wider vectors (bitwise-identical results; loads, stores and multiplies
shrink by the same factor):

- Phase D pre-apply: the per-token ``num_chunks`` loop of 64-lane passes ->
  one h_blk-wide (64/128/256) load/compute/store per row.  Note this pins
  h_blk to the legal lane set; the host dispatch already guarantees it.

Phase A (128-lane reduce/rmsnorm) and Phase B comb (256-lane gather/store)
merges were attempted and are currently BLOCKED by a compiler defect:
``V.vci`` above 64 lanes makes the ptoas VPTO emission pipeline fail with
``'arith.constant' op integer return type must be signless`` (a
``vci(size=128)`` plus one store is enough to trigger it; plain fp32
elementwise at 128/256 compiles and runs fine).  Both phases keep their
64-lane form until that is fixed upstream.

Phase C (Sinkhorn) deliberately stays 64-lane: the ``vcadd/vcmax`` group
cap of 8 result lanes matches ASC's 8-lane VLane reduction alignment, so
the matrices cannot be width-merged (only the alignment constraint, not
register width, forced that granularity).
"""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


def _u32_index(index_i32):
    return V.vinterpret_cast(index_i32, 'uint32')


def _row_reduce(mat, mask_lo, mask_hi):
    """Row sum broadcast: segmented vcadd over each 4-of-8 masked row + group-8 broadcast."""
    lo = V.vbrc(V.vcadd(mat, mask_lo, group=8, reassoc=True), size=64, group=8)
    hi = V.vbrc(V.vcadd(mat, mask_hi, group=8, reassoc=True), size=64, group=8)
    return V.vsel(mask_lo, lo, hi)


def _row_max(mat, mask_lo, mask_hi):
    """Row max broadcast: segmented vcmax over each 4-of-8 masked row + group-8 broadcast."""
    lo = V.vbrc(V.vcmax(mat, mask_lo, group=8), size=64, group=8)
    hi = V.vbrc(V.vcmax(mat, mask_hi, group=8), size=64, group=8)
    return V.vsel(mask_lo, lo, hi)


@tilelang.jit(target='pto')
def mhc_pre_big_fuse_fwd_vmi(
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
    """VMI fused forward: reduce partials + RMSNorm, split mixes, Sinkhorn, pre-apply.

    Mirrors ``mhc_pre_big_fuse_fwd_asc`` op-for-op so the fused result stays
    bitwise identical to the composed path.  Only the SimdVF bodies differ
    from the ASC original.
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
            # + _mhc_sinkhorn_fwd in a single SimdVF phase (schedule verbatim
            # from the ASC fused kernel).
            with T.SimdVF():
                mask_full = V.create_mask(64, size=64)
                one = V.vbrc(T.float32(1), size=64)
                zero = V.vbrc(T.float32(0), size=64)
                inv_group = V.vbrc(inv_rms_group_size, size=64)
                lane_idx = V.vci(T.int32(0), size=64)

                ##################################################################
                # Phase A: mixes = gemm_out_mul * rms, vectorized with the 6-VR
                # compact token layout: 16 tokens x 24 mixes = 384 fp32 = 6
                # VRs of 64 lanes; each token spans 24 lanes of its VR.  One
                # 64-lane vector spans 2 2/3 tokens, so each lane picks up
                # the rsqrt scale of the token it belongs to via a lane-range
                # mask instead of a physical pattern.
                # (128-lane merge blocked: vci above 64 lanes trips the
                # ptoas VPTO pipeline; minimal repro at hand, to be
                # reported upstream.)
                for vec_idx in T.unroll(tb * mhc3 // 64, explicit=True):
                    m_vec = V.vload(mul_flat_ub[vec_idx * 64], size=64)
                    rms_vec = V.alloc_var(T.float32, size=64)
                    rms_vec = zero
                    elem_start = vec_idx * 64
                    elem_end = elem_start + 64
                    token_start = elem_start // mhc3
                    token_end = (elem_end - 1) // mhc3
                    for token_idx in range(token_start, token_end + 1):
                        sq_val = V.vbrc(V.vload(sq_ub[token_idx], size=1), size=64)
                        rms_val = V.vdiv(
                            one,
                            V.vsqrt(V.vadd(V.vmul(sq_val, inv_group), V.vbrc(T.float32(rms_eps), size=64))),
                            mask_full,
                            precision='exact',
                        )
                        lane_start = token_idx * mhc3 - elem_start
                        lane_end = lane_start + mhc3
                        token_mask = V.vand(
                            V.vcmps(lane_idx, T.int32(lane_start), mask_full, 'ge'),
                            V.vcmps(lane_idx, T.int32(lane_end), mask_full, 'lt'),
                        )
                        rms_vec = V.vsel(token_mask, rms_val, rms_vec)
                    V.vstore(V.vmul(m_vec, rms_vec), mixes_flat_ub[vec_idx * 64])

                T.simd.mem_bar('VST_VLD')

                ##################################################################
                # Phase B: split mixes into pre/post/comb.
                idx = V.vci(T.int32(0), size=64)
                col3 = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col15 = V.vand(idx, V.vbrc(T.int32(15), size=64))
                scale0 = V.vbrc(V.vload(scale_ub[0, 0], size=1), size=64)
                scale1 = V.vbrc(V.vload(scale_ub[1, 0], size=1), size=64)
                scale2 = V.vbrc(V.vload(scale_ub[2, 0], size=1), size=64)
                pre_base_vec = V.alloc_var(T.float32, size=64)
                post_base_vec = V.alloc_var(T.float32, size=64)
                comb_base_vec = V.alloc_var(T.float32, size=64)
                pre_base_vec = zero
                post_base_vec = zero
                comb_base_vec = zero
                for base_id in range(mhc):
                    base_mask = V.vcmps(col3, T.int32(base_id), mask_full, 'eq')
                    pre_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id, 0], size=1), size=64), pre_base_vec)
                    post_base_vec = V.vsel(base_mask, V.vbrc(V.vload(base_ub[base_id + mhc, 0], size=1), size=64), post_base_vec)
                for base_id in range(mhc2):
                    base_mask = V.vcmps(col15, T.int32(base_id), mask_full, 'eq')
                    comb_base_vec = V.vsel(
                        base_mask,
                        V.vbrc(V.vload(base_ub[base_id + mhc * 2, 0], size=1), size=64),
                        comb_base_vec,
                    )

                # pre/post: 16 tokens per VR (4 lanes each).  The ASC plain
                # range() loop compiles fine on CCE, but the PTODSL printer
                # drops the int32 typing of the loop-carried vbrc scalar
                # (ptoas then sees vreg<64xindex>), so unroll explicitly and
                # keep vec_id a compile-time constant.
                for vec_id in T.unroll(tb // 16, explicit=True):
                    token = V.vadd(V.vshrs(idx, T.int32(2), mask_full), V.vbrc(T.int32(vec_id * 16), size=64))
                    token_offset = V.vadd(V.vshls(token, T.int32(4), mask_full), V.vshls(token, T.int32(3), mask_full))
                    pre_index = V.vadd(token_offset, col3)
                    post_index = V.vadd(pre_index, V.vbrc(T.int32(4), size=64))

                    pre_val = V.vadd(
                        V.vmul(V.vgather(mixes_flat_ub[0], _u32_index(pre_index), mask_full), scale0),
                        pre_base_vec,
                    )
                    pre_sig = V.vdiv(one, V.vadd(V.vexpdif(zero, pre_val, mask_full), one), mask_full, precision='exact')
                    V.vstore(V.vadd(pre_sig, V.vbrc(T.float32(mhc_pre_eps), size=64)), pre_ub[vec_id * 16, 0])

                    post_val = V.vadd(
                        V.vmul(V.vgather(mixes_flat_ub[0], _u32_index(post_index), mask_full), scale1),
                        post_base_vec,
                    )
                    post_sig = V.vdiv(one, V.vadd(V.vexpdif(zero, post_val, mask_full), one), mask_full, precision='exact')
                    V.vstore(
                        V.vmul(post_sig, V.vbrc(T.float32(mhc_post_mult_value), size=64)),
                        post_ub[vec_id * 16, 0],
                    )

                # comb: 4 tokens per VR (16 lanes each); unrolled for the
                # PTODSL loop-scalar typing reason.  (A 256-lane single
                # gather/store merge is blocked: the lane-index vci above 64
                # lanes trips the ptoas VPTO pipeline; same upstream defect.)
                for vec_id in T.unroll(tb // 4, explicit=True):
                    token = V.vadd(V.vshrs(idx, T.int32(4), mask_full), V.vbrc(T.int32(vec_id * 4), size=64))
                    token_offset = V.vadd(V.vshls(token, T.int32(4), mask_full), V.vshls(token, T.int32(3), mask_full))
                    comb_index = V.vadd(V.vadd(token_offset, V.vbrc(T.int32(8), size=64)), col15)
                    comb_val = V.vadd(
                        V.vmul(V.vgather(mixes_flat_ub[0], _u32_index(comb_index), mask_full), scale2),
                        comb_base_vec,
                    )
                    V.vstore(comb_val, comb_in_ub[vec_id * 4, 0])

                T.simd.mem_bar('VST_VLD')

                ##################################################################
                # _mhc_sinkhorn_fwd: 4 tokens packed per VR (64 fp32 = 4 x 4x4).
                # Each 8-lane half of a VR spans two matrix rows (rows 2k and
                # 2k+1 of one token); the complementary 4-of-8 masks split the
                # halves into single rows for the 8-aligned vcadd/vcmax, and
                # the helpers broadcast each row back over its half.  The 4x4
                # transpose stays a vselr lane permutation.
                mod8 = V.vand(idx, V.vbrc(T.int32(7), size=64))
                mask_lo = V.vcmps(mod8, T.int32(4), mask_full, 'lt')
                mask_hi = V.vcmps(mod8, T.int32(4), mask_full, 'ge')
                two = T.int32(2)
                three = T.int32(3)
                four = T.int32(4)
                local = V.vand(idx, V.vbrc(T.int32(15), size=64))
                # t_idx for 4x4 transpose within each 16-element token:
                # [r][c] at r*4+c, transpose -> c*4+r
                # t_idx = (lane // 16) * 16 + (lane % 4) * 4 + (lane % 16) // 4
                token_base_tidx = V.vshls(V.vshrs(idx, four, mask_full), four, mask_full)
                col = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col4 = V.vshls(col, two, mask_full)
                t_idx = V.vadd(V.vadd(token_base_tidx, col4), V.vshrs(local, two, mask_full))

                cur_a = V.alloc_var(T.float32, size=64)
                cur_b = V.alloc_var(T.float32, size=64)
                cur_c = V.alloc_var(T.float32, size=64)
                cur_d = V.alloc_var(T.float32, size=64)
                for g in range(tb // 16):
                    g0 = 4 * g
                    mat_a_0 = V.vload(comb_in_ub[g0 * 4, 0], size=64)
                    mat_b_0 = V.vload(comb_in_ub[(g0 + 1) * 4, 0], size=64)
                    mat_c_0 = V.vload(comb_in_ub[(g0 + 2) * 4, 0], size=64)
                    mat_d_0 = V.vload(comb_in_ub[(g0 + 3) * 4, 0], size=64)

                    rmax_a = _row_max(mat_a_0, mask_lo, mask_hi)
                    rmax_b = _row_max(mat_b_0, mask_lo, mask_hi)
                    rmax_c = _row_max(mat_c_0, mask_lo, mask_hi)
                    rmax_d = _row_max(mat_d_0, mask_lo, mask_hi)
                    mat_a_1 = V.vexpdif(mat_a_0, rmax_a, mask_full)
                    mat_b_1 = V.vexpdif(mat_b_0, rmax_b, mask_full)
                    mat_c_1 = V.vexpdif(mat_c_0, rmax_c, mask_full)
                    mat_d_1 = V.vexpdif(mat_d_0, rmax_d, mask_full)
                    rsum_a = _row_reduce(mat_a_1, mask_lo, mask_hi)
                    rsum_b = _row_reduce(mat_b_1, mask_lo, mask_hi)
                    rsum_c = _row_reduce(mat_c_1, mask_lo, mask_hi)
                    rsum_d = _row_reduce(mat_d_1, mask_lo, mask_hi)
                    mat_a_2 = V.vdiv(mat_a_1, rsum_a, mask_full, precision='exact')
                    mat_b_2 = V.vdiv(mat_b_1, rsum_b, mask_full, precision='exact')
                    mat_c_2 = V.vdiv(mat_c_1, rsum_c, mask_full, precision='exact')
                    mat_d_2 = V.vdiv(mat_d_1, rsum_d, mask_full, precision='exact')
                    mat_a_3 = V.vadd(mat_a_2, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64))
                    mat_b_3 = V.vadd(mat_b_2, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64))
                    mat_c_3 = V.vadd(mat_c_2, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64))
                    mat_d_3 = V.vadd(mat_d_2, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64))

                    tr_a = V.vselr(mat_a_3, t_idx)
                    tr_b = V.vselr(mat_b_3, t_idx)
                    tr_c = V.vselr(mat_c_3, t_idx)
                    tr_d = V.vselr(mat_d_3, t_idx)
                    csum_a = _row_reduce(tr_a, mask_lo, mask_hi)
                    csum_b = _row_reduce(tr_b, mask_lo, mask_hi)
                    csum_c = _row_reduce(tr_c, mask_lo, mask_hi)
                    csum_d = _row_reduce(tr_d, mask_lo, mask_hi)
                    tr_a = V.vdiv(tr_a, V.vadd(csum_a, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                    tr_b = V.vdiv(tr_b, V.vadd(csum_b, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                    tr_c = V.vdiv(tr_c, V.vadd(csum_c, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                    tr_d = V.vdiv(tr_d, V.vadd(csum_d, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                    cur_a = V.vselr(tr_a, t_idx)
                    cur_b = V.vselr(tr_b, t_idx)
                    cur_c = V.vselr(tr_c, t_idx)
                    cur_d = V.vselr(tr_d, t_idx)

                    for _step in range(sinkhorn_repeat - 1):
                        rsum_a = _row_reduce(cur_a, mask_lo, mask_hi)
                        rsum_b = _row_reduce(cur_b, mask_lo, mask_hi)
                        rsum_c = _row_reduce(cur_c, mask_lo, mask_hi)
                        rsum_d = _row_reduce(cur_d, mask_lo, mask_hi)
                        cur_a = V.vdiv(cur_a, V.vadd(rsum_a, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        cur_b = V.vdiv(cur_b, V.vadd(rsum_b, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        cur_c = V.vdiv(cur_c, V.vadd(rsum_c, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        cur_d = V.vdiv(cur_d, V.vadd(rsum_d, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')

                        tr_a = V.vselr(cur_a, t_idx)
                        tr_b = V.vselr(cur_b, t_idx)
                        tr_c = V.vselr(cur_c, t_idx)
                        tr_d = V.vselr(cur_d, t_idx)
                        csum_a = _row_reduce(tr_a, mask_lo, mask_hi)
                        csum_b = _row_reduce(tr_b, mask_lo, mask_hi)
                        csum_c = _row_reduce(tr_c, mask_lo, mask_hi)
                        csum_d = _row_reduce(tr_d, mask_lo, mask_hi)
                        tr_a = V.vdiv(tr_a, V.vadd(csum_a, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        tr_b = V.vdiv(tr_b, V.vadd(csum_b, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        tr_c = V.vdiv(tr_c, V.vadd(csum_c, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        tr_d = V.vdiv(tr_d, V.vadd(csum_d, V.vbrc(T.float32(mhc_sinkhorn_eps), size=64)), mask_full, precision='exact')
                        cur_a = V.vselr(tr_a, t_idx)
                        cur_b = V.vselr(tr_b, t_idx)
                        cur_c = V.vselr(tr_c, t_idx)
                        cur_d = V.vselr(tr_d, t_idx)

                    V.vstore(cur_a, comb_out_ub[g0 * 4, 0])
                    V.vstore(cur_b, comb_out_ub[(g0 + 1) * 4, 0])
                    V.vstore(cur_c, comb_out_ub[(g0 + 2) * 4, 0])
                    V.vstore(cur_d, comb_out_ub[(g0 + 3) * 4, 0])

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
                with T.SimdVF():
                    # One h_blk-wide vector per (token, mhc) row replaces
                    # ASC's num_chunks loop of 64-lane passes (physical
                    # 64-fp32 VR limit).  h_blk comes from the host dispatch
                    # as one of 64/128/256, all legal VMI lane counts.
                    mix_reg = V.alloc_local((mhc,), V.vreg(h_blk, T.float32))
                    xs = V.alloc_local((mhc,), V.vreg(h_blk, T.float32))
                    acc = V.alloc_var(T.float32, size=h_blk)
                    mask_full = V.create_mask(h_blk, size=h_blk)
                    for t in T.serial(tb):
                        for m in T.unroll(mhc, explicit=True):
                            mix_reg[m] = V.vbrc(V.vload(pre_ub[t, m], size=1), size=h_blk)
                        for m in T.unroll(mhc, explicit=True):
                            xs[m] = V.vcvt(V.vload(x_ub[t, m, 0], size=h_blk), 'float32')
                        acc = V.vmul(xs[0], mix_reg[0])
                        for m in T.unroll(mhc - 1, explicit=True):
                            acc = V.vmula(acc, xs[m + 1], mix_reg[m + 1], mask_full)
                        V.vstore(V.vcvt(acc, 'bfloat16'), layer_input_ub[t, 0])

                T.copy(
                    layer_input_ub,
                    layer_input[
                        token_base : token_base + tb,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                )
