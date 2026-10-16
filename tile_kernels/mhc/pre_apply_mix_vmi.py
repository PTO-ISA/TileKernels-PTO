"""VMI port of the mhc pre_apply_mix kernels (discussion #1616).

Each kernel is the ASC original with only the ``T.SimdVF()`` bodies rewritten
in logical VMI vectors; the outer schedule (T.Kernel / T.Persistent / T.copy /
UB layout / num_stages) and the host contract are kept verbatim from
``pre_apply_mix_asc.py`` so ASC and VMI stay directly comparable.

Key ASC -> VMI correspondences used here (same set as the post_vmi port):

- ``S.vld(x_ub[col], dist='UNPK_B16')`` + ``S.vcvt(.., T.float32, part=0)``
  becomes ``V.vcvt(V.vload(x_ub[col], size=64), 'float32')``: the load yields
  the logical 64-lane bf16 vector and the physical unpack layout is left to
  ptoas.
- ``S.vsts(.., S.vcvt(v, T.bfloat16), dist='PK_B32')`` becomes
  ``V.vstore(V.vcvt(v, 'bfloat16'), ..)`` for the same reason.
- ``S.vld(scalar_ub[i], dist='BRC_B32')`` becomes
  ``V.vbrc(V.vload(scalar_ub[i], size=1), size=64)``: PTOAS removed the
  vload broadcast dist_mode, so a scalar broadcast is a one-element vload
  followed by vbrc.
- ``S.vdup(0.0, T.float32)`` becomes ``V.vbrc(T.float32(0.0), size=64)``.
- ``S.vmula(dst, a, b)`` (in-place accumulate) becomes
  ``dst = V.vmula(dst, a, b, mask_full)`` (value form; vmula requires an
  explicit mask).
- ``S.vsts(dst, S.vcadd(v)), dist='ONEPT_B32')`` becomes ``V.vstore`` of the
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
def mhc_pre_apply_mix_fwd_vmi(x, mix, o, h_blk, num_vec_cores: int):
    """Ascend forward: ``o[n, h] = sum_mhc mix[n, mhc] * x[n, mhc, h]``."""
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    x: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    o: T.Tensor[(num_tokens, hidden), T.bfloat16]

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC
    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        o_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, mix_ub: num_stages, o_ub: num_stages})

        for pid_n, pid_h in T.Persistent([num_tokens, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])
            T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)

            with T.SimdVF():
                # All vectors are full-width 64-lane fp32 (h_blk is a multiple
                # of 64), so only vmula needs an explicit mask.
                # Preload the mix scalars as broadcast registers once per token-tile.
                mix_reg = V.alloc_local((mhc,), V.vreg(64, T.float32))
                for m in T.unroll(mhc, explicit=True):
                    mix_reg[m] = V.vbrc(V.vload(mix_ub[m], size=1), size=64)
                xs = V.alloc_local((mhc,), V.vreg(64, T.float32))
                acc = V.alloc_var(T.float32, size=64)
                mask_full = V.create_mask(64, size=64)
                for c in T.serial(num_chunks):
                    col = c * _VEC
                    for m in T.unroll(mhc, explicit=True):
                        xs[m] = V.vcvt(V.vload(x_ub[m, col], size=64), 'float32')
                    acc = V.vmul(xs[0], mix_reg[0])
                    for m in T.unroll(mhc - 1, explicit=True):
                        acc = V.vmula(acc, xs[m + 1], mix_reg[m + 1], mask_full)
                    V.vstore(V.vcvt(acc, 'bfloat16'), o_ub[col])

            T.copy(o_ub, o[pid_n, pid_h * h_blk])


@tilelang.jit(target='pto')
def mhc_pre_apply_mix_bwd_vmi(o_grad, x, mix, x_grad, h_blk, num_stages, num_vec_cores: int):
    """Ascend backward.

    ``x_grad[n, m, h] += mix[n, m] * o_grad[n, h]``  (accumulated onto existing values)
    ``mix_grad[n, m] = sum_h o_grad[n, h] * x[n, m, h]``  (reduction over h)
    """
    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    hidden = T.const('hidden')
    o_grad: T.Tensor[(num_tokens, hidden), T.bfloat16]
    x: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix: T.Tensor[(num_tokens, mhc), T.float32]
    x_grad: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    mix_grad = T.empty((num_tokens, mhc), T.float32)

    num_hidden_tiles = T.ceildiv(hidden, h_blk)
    num_chunks = h_blk // _VEC

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((h_blk,), T.bfloat16)
        x_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_grad_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        mix_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        mix_grad_acc = T.alloc_shared((mhc, _VEC), T.float32)
        mix_grad_ub = T.alloc_shared((T.ceildiv(mhc, 8) * 8,), T.float32)  # TODO remove manual padding
        T.annotate_buffer_versions(
            {
                o_grad_ub: num_stages,
                x_ub: num_stages,
                x_grad_ub: num_stages,
                mix_ub: num_stages,
                mix_grad_ub: num_stages,
            }
        )

        for pid_n in T.Persistent([num_tokens], num_vec_cores, core_id, group_size=1, num_stages=num_stages):
            T.copy(mix[pid_n, 0], mix_ub[0:mhc])

            with T.SimdVF():
                zero = V.vbrc(T.float32(0.0), size=64)
                for m in T.unroll(mhc, explicit=True):
                    V.vstore(zero, mix_grad_acc[m, 0])

            for pid_h in T.serial(num_hidden_tiles):
                T.copy(o_grad[pid_n, pid_h * h_blk], o_grad_ub)
                T.copy(x[pid_n, 0, pid_h * h_blk], x_ub)
                T.copy(x_grad[pid_n, 0, pid_h * h_blk], x_grad_ub)

                with T.SimdVF():
                    mix_reg = V.alloc_local((mhc,), V.vreg(64, T.float32))
                    for m in T.unroll(mhc, explicit=True):
                        mix_reg[m] = V.vbrc(V.vload(mix_ub[m], size=1), size=64)
                    mix_grad_reg = V.alloc_var(T.float32, size=64)
                    x_grad_f32 = V.alloc_var(T.float32, size=64)
                    mask_full = V.create_mask(64, size=64)
                    for m in T.unroll(mhc, explicit=True):
                        mix_grad_reg = V.vload(mix_grad_acc[m, 0], size=64)
                        for c in T.serial(num_chunks):
                            col = c * _VEC
                            o_grad_f32 = V.vcvt(V.vload(o_grad_ub[col], size=64), 'float32')
                            x_f32 = V.vcvt(V.vload(x_ub[m, col], size=64), 'float32')
                            x_grad_f32 = V.vcvt(V.vload(x_grad_ub[m, col], size=64), 'float32')
                            x_grad_f32 = V.vmula(x_grad_f32, o_grad_f32, mix_reg[m], mask_full)
                            V.vstore(V.vcvt(x_grad_f32, 'bfloat16'), x_grad_ub[m, col])
                            mix_grad_reg = V.vmula(mix_grad_reg, o_grad_f32, x_f32, mask_full)
                        V.vstore(mix_grad_reg, mix_grad_acc[m, 0])

                T.copy(x_grad_ub, x_grad[pid_n, 0, pid_h * h_blk])

            with T.SimdVF():
                # ASC vcadd pairwise-reduces the 64 accumulation lanes and
                # stores the single result (ONEPT_B32); VMI vcadd returns a
                # 1-lane vreg, so store it directly as an unmasked one-lane
                # (scalar) store (same ptoas alignment constraint as the
                # post port; reassoc=True keeps the tree reduction the
                # pairwise ASC form implies).
                mask_full = V.create_mask(64, size=64)
                for m in T.unroll(mhc, explicit=True):
                    total = V.vcadd(V.vload(mix_grad_acc[m, 0], size=64), mask_full, reassoc=True)
                    V.vstore(total, mix_grad_ub[m])
            T.copy(mix_grad_ub[0:mhc], mix_grad[pid_n, 0])

    return mix_grad
