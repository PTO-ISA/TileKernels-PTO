"""VMI port of the mhc post kernels (discussion #1616).

Each kernel is the ASC original with only the ``T.SimdVF()`` bodies rewritten
in logical VMI vectors; the outer schedule (T.Kernel / T.Persistent / T.copy /
UB layout / num_stages) and the host contract are kept verbatim from
``post_asc.py`` so ASC and VMI stay directly comparable.

Key ASC -> VMI correspondences used here:

- ``S.vld(x_ub[col], dist='UNPK_B16')`` + ``S.vcvt(.., T.float32, part=0)``
  becomes ``V.vcvt(V.vload(x_ub[col], size=64), 'float32')``: the load yields
  the logical 64-lane bf16 vector and the physical unpack/packing layout is
  left to ptoas.
- ``S.vsts(.., S.vcvt(v, T.bfloat16), dist='PK_B32')`` becomes
  ``V.vstore(V.vcvt(v, 'bfloat16'), ..)`` for the same reason.
- ``S.vld(scalar_ub[i, j], dist='BRC_B32')`` becomes
  ``V.vbrc(V.vload(scalar_ub[i, j], size=1), size=64)``: PTOAS removed the
  vload broadcast dist_mode, so a scalar broadcast is a one-element vload
  followed by vbrc (per the ptoas commit "remove load/store dist_mode").
- ``S.vmula(dst, a, b)`` (in-place accumulate) becomes
  ``dst = V.vmula(dst, a, b, mask_full)`` (value form; vmula requires an
  explicit mask).
- ``S.vsts(dst, S.vcadd(v), dist='ONEPT_B32')`` becomes ``V.vstore`` of the
  1-lane ``V.vcadd`` result itself (an unmasked one-lane scalar store;
  ptoas rejects a broadcast-plus-lane-0-mask store at unaligned scalar
  slots).
"""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


@tilelang.jit(target='pto')
def mhc_post_fwd_vmi(comb_res_mix, residual, post_layer_mix, x, out, h_blk, num_vec_cores: int):
    """Ascend forward.

    ``out[n, mo, h] = post_layer_mix[n, mo] * x[n, h]
                      + sum_mi comb_res_mix[n, mi, mo] * residual[n, mi, h]``
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), T.bfloat16]
    out: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        out_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                residual_ub: num_stages,
                x_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                out_ub: num_stages,
            }
        )

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])
            T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)
            T.copy(x[pid_n, pid_h * h_blk], x_ub)

            with T.SimdVF():
                # All vectors are full-width 64-lane fp32 (h_blk is a multiple
                # of 64), so only vmula needs an explicit mask.
                comb_mix = V.alloc_local((mhc * mhc,), V.vreg(64, T.float32))
                post_mix = V.alloc_local((mhc,), V.vreg(64, T.float32))
                for mi in T.unroll(mhc, explicit=True):
                    for mo in T.unroll(mhc, explicit=True):
                        comb_mix[mi * mhc + mo] = V.vbrc(V.vload(comb_res_mix_ub[mi, mo], size=1), size=64)
                for mo in T.unroll(mhc, explicit=True):
                    post_mix[mo] = V.vbrc(V.vload(post_layer_mix_ub[mo], size=1), size=64)
                out_reg = V.alloc_local((mhc,), V.vreg(64, T.float32))
                res = V.alloc_local((mhc,), V.vreg(64, T.float32))
                mask_full = V.create_mask(64, size=64)
                for c in T.serial(num_chunks):
                    col = c * _VEC
                    x_f32 = V.vcvt(V.vload(x_ub[col], size=64), 'float32')
                    for mi in T.unroll(mhc, explicit=True):
                        res[mi] = V.vcvt(V.vload(residual_ub[mi, col], size=64), 'float32')
                    for mo in T.unroll(mhc, explicit=True):
                        out_reg[mo] = V.vmul(x_f32, post_mix[mo])
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            out_reg[mo] = V.vmula(out_reg[mo], res[mi], comb_mix[mi * mhc + mo], mask_full)
                    for mo in T.unroll(mhc, explicit=True):
                        V.vstore(V.vcvt(out_reg[mo], 'bfloat16'), out_ub[mo, col])

            T.copy(out_ub, out[pid_n, 0, pid_h * h_blk])


@tilelang.jit(target='pto')
def mhc_post_bwd_vmi(d_o, comb_res_mix, residual, post_layer_mix, x, h_blk, num_stages, num_vec_cores: int):
    """Ascend backward.

    d_x[n, h] = sum_mo d_o[n, mo, h] * post_layer_mix[n, mo]
    d_post_layer_mix[n, mo] = sum_h d_o[n, mo, h] * x[n, h]
    d_residual[n, mi, h] = sum_mo d_o[n, mo, h] * comb_res_mix[n, mi, mo]
    d_comb_res_mix[n, mi, mo] = sum_h d_o[n, mo, h] * residual[n, mi, h]
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    d_o: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    comb_res_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_layer_mix: T.Tensor[(num_tokens, mhc), T.float32]
    x: T.Tensor[(num_tokens, hidden), T.bfloat16]

    d_comb_res_mix = T.empty((num_tokens, mhc, mhc), T.float32)
    d_residual = T.empty((num_tokens, mhc, hidden), T.bfloat16)
    d_post_layer_mix = T.empty((num_tokens, mhc), T.float32)
    d_x = T.empty((num_tokens, hidden), T.bfloat16)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        d_o_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        comb_res_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_layer_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        d_residual_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        d_post_mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        d_comb_mix_ub = T.alloc_shared((mhc, mhc), T.float32)
        post_mix_acc = T.alloc_shared((mhc, _VEC), T.float32)
        comb_mix_acc = T.alloc_shared((mhc * mhc, _VEC), T.float32)
        T.annotate_buffer_versions(
            {
                d_o_ub: num_stages,
                x_ub: num_stages,
                residual_ub: num_stages,
                d_x_ub: num_stages,
                d_residual_ub: num_stages,
                comb_res_mix_ub: num_stages,
                post_layer_mix_ub: num_stages,
                d_post_mix_ub: 1,
                d_comb_mix_ub: 1,
                post_mix_acc: 1,
                comb_mix_acc: 1,
            }
        )

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(comb_res_mix[pid_n, 0, 0], comb_res_mix_ub)
            T.copy(post_layer_mix[pid_n, 0], post_layer_mix_ub[0:mhc])

            with T.SimdVF():
                zero = V.vbrc(T.float32(0.0), size=64)
                for mo in T.unroll(mhc, explicit=True):
                    V.vstore(zero, post_mix_acc[mo, 0])
                for idx in T.unroll(mhc * mhc, explicit=True):
                    V.vstore(zero, comb_mix_acc[idx, 0])

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(d_o[pid_n, 0, pid_h * h_blk], d_o_ub)
                T.copy(x[pid_n, pid_h * h_blk], x_ub)
                T.copy(residual[pid_n, 0, pid_h * h_blk], residual_ub)

                with T.SimdVF():
                    mask_full = V.create_mask(64, size=64)
                    post_mix = V.alloc_local((mhc,), V.vreg(64, T.float32))
                    for mo in T.unroll(mhc, explicit=True):
                        post_mix[mo] = V.vbrc(V.vload(post_layer_mix_ub[mo], size=1), size=64)
                    comb_mix = V.alloc_local((mhc * mhc,), V.vreg(64, T.float32))
                    for mi in T.unroll(mhc, explicit=True):
                        for mo in T.unroll(mhc, explicit=True):
                            comb_mix[mi * mhc + mo] = V.vbrc(V.vload(comb_res_mix_ub[mi, mo], size=1), size=64)

                    d_o_reg = V.alloc_local((mhc,), V.vreg(64, T.float32))
                    d_x_reg = V.alloc_var(T.float32, size=64)
                    d_res_reg = V.alloc_var(T.float32, size=64)
                    post_acc_reg = V.alloc_var(T.float32, size=64)
                    d_comb_mix_mi = V.alloc_local((mhc,), V.vreg(64, T.float32))

                    for c in T.serial(num_chunks):
                        col = c * _VEC
                        x_f32 = V.vcvt(V.vload(x_ub[col], size=64), 'float32')
                        for mo in T.unroll(mhc, explicit=True):
                            d_o_reg[mo] = V.vcvt(V.vload(d_o_ub[mo, col], size=64), 'float32')

                        d_x_reg = V.vbrc(T.float32(0.0), size=64)
                        for mo in T.unroll(mhc, explicit=True):
                            d_x_reg = V.vmula(d_x_reg, d_o_reg[mo], post_mix[mo], mask_full)
                        V.vstore(V.vcvt(d_x_reg, 'bfloat16'), d_x_ub[col])

                        for mo in T.unroll(mhc, explicit=True):
                            post_acc_reg = V.vload(post_mix_acc[mo, 0], size=64)
                            post_acc_reg = V.vmula(post_acc_reg, d_o_reg[mo], x_f32, mask_full)
                            V.vstore(post_acc_reg, post_mix_acc[mo, 0])

                        for mi in T.unroll(mhc, explicit=True):
                            res_mi = V.vcvt(V.vload(residual_ub[mi, col], size=64), 'float32')
                            for mo in T.unroll(mhc, explicit=True):
                                d_comb_mix_mi[mo] = V.vload(comb_mix_acc[mi * mhc + mo, 0], size=64)
                            d_res_reg = V.vbrc(T.float32(0.0), size=64)
                            for mo in T.unroll(mhc, explicit=True):
                                d_res_reg = V.vmula(d_res_reg, d_o_reg[mo], comb_mix[mi * mhc + mo], mask_full)
                                d_comb_mix_mi[mo] = V.vmula(d_comb_mix_mi[mo], d_o_reg[mo], res_mi, mask_full)
                            V.vstore(V.vcvt(d_res_reg, 'bfloat16'), d_residual_ub[mi, col])
                            for mo in T.unroll(mhc, explicit=True):
                                V.vstore(d_comb_mix_mi[mo], comb_mix_acc[mi * mhc + mo, 0])

                T.copy(d_x_ub, d_x[pid_n, pid_h * h_blk])
                T.copy(d_residual_ub, d_residual[pid_n, 0, pid_h * h_blk])

            with T.SimdVF():
                # ASC vcadd pairwise-reduces the 64 accumulation lanes and
                # stores the single result (ONEPT_B32); VMI vcadd returns a
                # 1-lane vreg, so store it directly as an unmasked one-lane
                # (scalar) store.  Broadcasting it back to 64 lanes and
                # masking to lane 0 is rejected by ptoas: a masked store
                # needs a provably aligned destination, and per-element
                # scalar slots are not lane aligned
                # (reassoc=True keeps the tree reduction the pairwise ASC
                # form implies).
                mask_full = V.create_mask(64, size=64)
                for mo in T.unroll(mhc, explicit=True):
                    total = V.vcadd(V.vload(post_mix_acc[mo, 0], size=64), mask_full, reassoc=True)
                    V.vstore(total, d_post_mix_ub[mo])
                for mi in T.unroll(mhc, explicit=True):
                    for mo in T.unroll(mhc, explicit=True):
                        total = V.vcadd(V.vload(comb_mix_acc[mi * mhc + mo, 0], size=64), mask_full, reassoc=True)
                        V.vstore(total, d_comb_mix_ub[mi, mo])

            T.copy(d_post_mix_ub[0:mhc], d_post_layer_mix[pid_n, 0])
            T.copy(d_comb_mix_ub, d_comb_res_mix[pid_n, 0, 0])

    return d_comb_res_mix, d_residual, d_post_layer_mix, d_x
