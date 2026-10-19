"""VMI port of the mhc sinkhorn kernels (discussion #1616).

Each kernel is the ASC original (``sinkhorn_asc.py``) with only the
``T.SimdVF()`` bodies rewritten in logical VMI vectors; the outer schedule
(T.Kernel / T.Persistent / T.copy / UB layout / num_stages) and the host
contract are kept verbatim so ASC and VMI stay directly comparable.

Lane layout (unchanged, fixed by the UB block): one 64-lane fp32 vector holds
4 tokens x 16 elements, ``lane = 16*t + 4*r + c`` for token t, row r, col c of
each 4x4 matrix. Rows are contiguous 4-lane groups; the 4x4 transpose is a
per-lane index shuffle.

Key ASC -> VMI correspondences used here:

- ``S.pset(32)`` (full mask of a 64-lane fp32 VR) becomes
  ``V.create_mask(64, size=64)``.
- Row/col reductions (``vcgadd`` / ``vcgmax`` halves + ``vselr`` + ``vsel``
  in ASC) become two-round ``vselr`` + ``vadd`` / ``vmax`` binary trees over
  each row's 4 lanes: every lane first combines with its 2-lane pair partner
  (``vselr`` by ``lane ^ 1``), then with the opposite pair of the same row.
  This keeps the ASC grouped-add tree pairing ``((a+b)+(c+d))`` -- bitwise
  identical sums (verified against a dumped ASC stage) -- and ``max`` is
  order-insensitive, so both reductions are bitwise equal to the ASC
  originals.

  Intentional difference, forced by two ptoas defects found while porting
  (minimal repros in ``verify/report/sinkhorn_ptoas_issues.md``):

  1. ``vcmax`` with a partial element mask plus ``group=8`` enters inactive
     lanes into the comparison as 0.0 and splats the group-0 reduction over
     every result lane (``verify/probe2_ops.py``);
  2. a ``vcadd``-produced row-sum register is corrupted as soon as it is also
     consumed by a ``vdiv`` in the same kernel (``verify/probe4_stages.py`` /
  ``verify/probe5_vdiv.py``: the same expression stores correct values with
     no div consumer and wrong ones with one).

  The tree formulations use only primitives verified end-to-end
  (``vselr``/``vadd``/``vmax``), stay in logical VMI form, and are shape
  generic -- no ASC fallback, no host-side workarounds.
- ``S.vselr(v, idx)`` (4x4 transpose, per-lane select) stays ``V.vselr(v, idx)``;
  the f32->u32 reinterpretation the ASC frontend needed is handled by the VMI
  lowering.
- ``S.vadds(v, eps, m)`` becomes ``V.vadd(v, eps_b)`` with
  ``eps_b = V.vbrc(T.float32(eps), size=64)`` (broadcast + plain vector add).
- ``S.vdiv(a, b, m)`` becomes ``V.vdiv(a, b, mask_full, precision='exact')`` (default precision lowers to
  the same hardware divide as the ASC form).
- ``S.vexpdif(x, max_b, m)`` stays ``V.vexpdif(x, max_b, mask_full)`` (mask
  is mandatory for vexpdif).
- ``S.vdup(c, T.int32, m)`` becomes ``V.vbrc(T.int32(c), size=64)``; integer
  ``vand``/``vadd``/``vxor`` drop the redundant full mask while ``vshls``/
  ``vshrs`` keep theirs (mask-mandatory ops); ``S.vci(0, T.int32)`` becomes
  ``V.vci(T.int32(0), size=64)``.
- ``S.vld(a)`` / ``S.vsts(a, v, m)`` become ``V.vload(a, size=64)`` /
  ``V.vstore(v, a)``; ``S.mem_bar('VST_VLD')`` becomes
  ``T.simd.mem_bar('VST_VLD')``.
"""

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


def _row_max(mat, pair_swap_idx, cross_pair_idx):
    """Broadcast row max (ASC: vcgmax x2 + vselr x2 + vsel), as a vselr/vmax tree."""
    pair = V.vmax(mat, V.vselr(mat, pair_swap_idx))  # max over each 2-lane pair
    return V.vmax(pair, V.vselr(pair, cross_pair_idx))  # max over the 4-lane row


def _row_reduce(mat, pair_swap_idx, cross_pair_idx):
    """Broadcast row sum (ASC: vcgadd x2 + vselr x2 + vsel), as a pairwise vselr/vadd tree.

    Lane l first sums with its 2-lane pair partner (``vselr`` by ``lane ^ 1``),
    then adds the opposite pair's partial: ``(a+b)+(c+d)`` at every lane.
    This is exactly the ASC ``vcgadd`` fold order (verified bitwise against a
    dumped ASC stage), so every fp32 rounding step matches.
    """
    pair = V.vadd(mat, V.vselr(mat, pair_swap_idx))  # (a+b) / (c+d) per pair
    return V.vadd(pair, V.vselr(pair, cross_pair_idx))  # (a+b)+(c+d)


def _col_reduce(mat, t_idx, pair_swap_idx, cross_pair_idx):
    """Col reduce: transpose, row-reduce -> (transposed, broadcast col sum).

    NOTE: The returned broadcast col sum is in TRANSPOSED layout (each lane
    gets its transposed-row sum = original-col sum). To use with original-layout
    data, transpose the broadcast result back via V.vselr(csum_b, t_idx).
    """
    tr = V.vselr(mat, t_idx)
    return tr, _row_reduce(tr, pair_swap_idx, cross_pair_idx)


@tilelang.jit(target='pto')
def mhc_sinkhorn_fwd_vmi(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """SimdVF forward: 4 tokens packed per VR (64 fp32 = 4 x 4x4).

    Layout per VR: 4 tokens x 16 elements = 64 fp32, ``lane = 16*t + 4*r + c``.
    Row/col sums and maxes are two-round vselr/vadd / vselr/vmax trees over
    each row's 4 contiguous lanes; the 4x4 transpose is a vselr by t_idx.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 4

    with T.Kernel(num_vec_cores) as core_id:
        comb_in_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        comb_out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({comb_in_ub: num_stages, comb_out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(comb_res_mix[token_offset, 0, 0], comb_in_ub)

            with T.SimdVF():
                mask_full = V.create_mask(64, size=64)

                # --- Index vectors (V.vci ramps + integer ops) ---
                idx = V.vci(T.int32(0), size=64)
                four = T.int32(4)
                fifteen = V.vbrc(T.int32(15), size=64)
                local = V.vand(idx, fifteen)
                # t_idx for 4x4 transpose within each 16-element token:
                # [r][c] at r*4+c, transpose -> c*4+r
                # t_idx = (lane // 16) * 16 + col*4 + row
                # where col = lane % 4, row = (lane % 16) // 4
                token_base_tidx = V.vshls(V.vshrs(idx, four, mask_full), four, mask_full)
                col = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col4 = V.vshls(col, T.int32(2), mask_full)
                t_idx = V.vadd(V.vadd(token_base_tidx, col4), V.vshrs(local, T.int32(2), mask_full))

                # Row-reduction tree indices: pair_swap_idx flips the low bit
                # (partner in the same 2-lane pair); cross_pair_idx points at
                # the same-parity lane of the row's other pair.
                one = V.vbrc(T.int32(1), size=64)
                two = V.vbrc(T.int32(2), size=64)
                pair_swap_idx = V.vxor(idx, one)
                row_base = V.vand(idx, V.vbrc(T.int32(-4), size=64))
                cross_pair_idx = V.vadd(
                    V.vsub(V.vadd(row_base, two), V.vand(idx, two)),
                    V.vand(idx, one),
                )
                eps_b = V.vbrc(T.float32(eps), size=64)

                # Load 4 tokens x 16 elements = 64 fp32
                mat_0 = V.vload(comb_in_ub[0, 0, 0], size=64)

                # Step 1: Row softmax + eps
                rmax_b_0 = _row_max(mat_0, pair_swap_idx, cross_pair_idx)
                mat_1 = V.vexpdif(mat_0, rmax_b_0, mask_full)
                rsum_b_0 = _row_reduce(mat_1, pair_swap_idx, cross_pair_idx)
                mat_2 = V.vdiv(mat_1, rsum_b_0, mask_full, precision='exact')
                mat_3 = V.vadd(mat_2, eps_b)

                # Step 2: Col normalize (transpose, reduce, normalize, transpose back)
                tr_0 = V.vselr(mat_3, t_idx)
                csum_b_0 = _row_reduce(tr_0, pair_swap_idx, cross_pair_idx)
                tr_1 = V.vdiv(tr_0, V.vadd(csum_b_0, eps_b), mask_full, precision='exact')
                mat_4 = V.vselr(tr_1, t_idx)

                # Step 3: repeat-1 iterations (Python range() unroll)
                # Use alloc_var for cur to allow cross-iteration assignment.
                # Pre-loop uses unique variable names (mat_0..mat_4) to avoid
                # CSE matching loop body expressions against pre-loop SSA values.
                cur = V.alloc_var(T.float32, size=64)
                cur = mat_4
                for _step in range(repeat - 1):
                    rsum_b = _row_reduce(cur, pair_swap_idx, cross_pair_idx)
                    cur = V.vdiv(cur, V.vadd(rsum_b, eps_b), mask_full, precision='exact')

                    tr = V.vselr(cur, t_idx)
                    csum_b = _row_reduce(tr, pair_swap_idx, cross_pair_idx)
                    tr = V.vdiv(tr, V.vadd(csum_b, eps_b), mask_full, precision='exact')
                    cur = V.vselr(tr, t_idx)

                V.vstore(cur, comb_out_ub[0, 0, 0])

            T.copy(comb_out_ub, comb_res_mix_out[token_offset, 0, 0])


@tilelang.jit(target='pto')
def mhc_sinkhorn_bwd_vmi(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """SimdVF backward: 8 tokens (2 VRs) per block.

    Two phases in one SimdVF:
    1. Forward recompute: softmax + repeat-1 iterations of row/col normalize.
       Save intermediate x states and sums to UB.
    2. Backward iteration: reverse 2*repeat-1 steps, loading x_inter and sum
       from UB, computing grad updates.

    2 VRs are independent (different tokens), so the compiler can interleave
    their loads and computes to hide vload latency.

    UB layout for intermediate storage (per block of 8 tokens):
      xs_ub:   (2*repeat, 8, 4, 4) fp32 = 2*repeat * 2048B  (e.g. 40KB for repeat=10)
      sums_ub: (2*repeat, 8, 4)   fp32 = 2*repeat *  512B  (e.g. 10KB for repeat=10)

    Each VR holds 4 tokens x 16 elements = 64 fp32.
    VR0 = tokens 0-3 (UB offset 0), VR1 = tokens 4-7 (UB offset 64 fp32 = 256B).
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        grad_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        x_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        xs_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({grad_ub: num_stages, x_ub: num_stages, out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(grad_output[token_offset, 0, 0], grad_ub)
            T.copy(x[token_offset, 0, 0], x_ub)

            with T.SimdVF():
                mask_full = V.create_mask(64, size=64)

                # --- Precompute index VRs (shared across both VRs) ---
                idx = V.vci(T.int32(0), size=64)
                four = T.int32(4)
                fifteen = V.vbrc(T.int32(15), size=64)
                local = V.vand(idx, fifteen)
                token_base_tidx = V.vshls(V.vshrs(idx, four, mask_full), four, mask_full)
                col = V.vand(idx, V.vbrc(T.int32(3), size=64))
                col4 = V.vshls(col, T.int32(2), mask_full)
                t_idx = V.vadd(V.vadd(token_base_tidx, col4), V.vshrs(local, T.int32(2), mask_full))
                one = V.vbrc(T.int32(1), size=64)
                two = V.vbrc(T.int32(2), size=64)
                pair_swap_idx = V.vxor(idx, one)
                row_base = V.vand(idx, V.vbrc(T.int32(-4), size=64))
                cross_pair_idx = V.vadd(
                    V.vsub(V.vadd(row_base, two), V.vand(idx, two)),
                    V.vand(idx, one),
                )
                eps_b = V.vbrc(T.float32(eps), size=64)

                # VR1 starts at token 4 (4 tokens per VR)
                vr1_tok = 4

                # ==================== Phase 1: Forward recompute ====================
                # Load both VRs
                x0_0 = V.vload(x_ub[0, 0, 0], size=64)  # VR0: tokens 0-3
                x1_0 = V.vload(x_ub[vr1_tok, 0, 0], size=64)  # VR1: tokens 4-7

                # Softmax + eps
                rmax0_b = _row_max(x0_0, pair_swap_idx, cross_pair_idx)
                rmax1_b = _row_max(x1_0, pair_swap_idx, cross_pair_idx)
                x0_1 = V.alloc_var(T.float32, size=64)
                x1_1 = V.alloc_var(T.float32, size=64)
                x0_1 = V.vexpdif(x0_0, rmax0_b, mask_full)
                x1_1 = V.vexpdif(x1_0, rmax1_b, mask_full)
                rsum0_b = _row_reduce(x0_1, pair_swap_idx, cross_pair_idx)
                rsum1_b = _row_reduce(x1_1, pair_swap_idx, cross_pair_idx)
                x0_2 = V.vdiv(x0_1, rsum0_b, mask_full, precision='exact')
                x1_2 = V.vdiv(x1_1, rsum1_b, mask_full, precision='exact')
                x0_3 = V.vadd(x0_2, eps_b)
                x1_3 = V.vadd(x1_2, eps_b)

                # Save xs[0] = softmax(x) (before +eps)
                V.vstore(x0_2, xs_ub[0, 0, 0, 0])
                V.vstore(x1_2, xs_ub[0, vr1_tok, 0, 0])
                # Save xs[1] = softmax(x) + eps
                V.vstore(x0_3, xs_ub[1, 0, 0, 0])
                V.vstore(x1_3, xs_ub[1, vr1_tok, 0, 0])

                # First col normalize
                tr0_0, csum0_b = _col_reduce(x0_3, t_idx, pair_swap_idx, cross_pair_idx)
                tr1_0, csum1_b = _col_reduce(x1_3, t_idx, pair_swap_idx, cross_pair_idx)
                # Save sums[1] = col_sum (transpose back to original layout for bwd)
                csum0_b_t = V.vselr(csum0_b, t_idx)
                csum1_b_t = V.vselr(csum1_b, t_idx)
                V.vstore(csum0_b_t, sums_ub[1, 0, 0, 0])
                V.vstore(csum1_b_t, sums_ub[1, vr1_tok, 0, 0])
                tr0_1 = V.vdiv(tr0_0, V.vadd(csum0_b, eps_b), mask_full, precision='exact')
                tr1_1 = V.vdiv(tr1_0, V.vadd(csum1_b, eps_b), mask_full, precision='exact')
                x0_4 = V.vselr(tr0_1, t_idx)
                x1_4 = V.vselr(tr1_1, t_idx)

                # repeat-1 iterations of row + col normalize
                # Use alloc_var for loop-carried variables + Python range() unroll
                cur0 = V.alloc_var(T.float32, size=64)
                cur1 = V.alloc_var(T.float32, size=64)
                cur0 = x0_4
                cur1 = x1_4

                for step in range(repeat - 1):
                    s_row = (step + 1) * 2
                    s_col = (step + 1) * 2 + 1

                    # Row normalize
                    rsum0_b = _row_reduce(cur0, pair_swap_idx, cross_pair_idx)
                    rsum1_b = _row_reduce(cur1, pair_swap_idx, cross_pair_idx)
                    V.vstore(rsum0_b, sums_ub[s_row, 0, 0, 0])
                    V.vstore(rsum1_b, sums_ub[s_row, vr1_tok, 0, 0])
                    V.vstore(cur0, xs_ub[s_row, 0, 0, 0])
                    V.vstore(cur1, xs_ub[s_row, vr1_tok, 0, 0])
                    cur0 = V.vdiv(cur0, V.vadd(rsum0_b, eps_b), mask_full, precision='exact')
                    cur1 = V.vdiv(cur1, V.vadd(rsum1_b, eps_b), mask_full, precision='exact')

                    # Col normalize
                    tr0, csum0_b = _col_reduce(cur0, t_idx, pair_swap_idx, cross_pair_idx)
                    tr1, csum1_b = _col_reduce(cur1, t_idx, pair_swap_idx, cross_pair_idx)
                    csum0_b_t = V.vselr(csum0_b, t_idx)
                    csum1_b_t = V.vselr(csum1_b, t_idx)
                    V.vstore(csum0_b_t, sums_ub[s_col, 0, 0, 0])
                    V.vstore(csum1_b_t, sums_ub[s_col, vr1_tok, 0, 0])
                    V.vstore(cur0, xs_ub[s_col, 0, 0, 0])
                    V.vstore(cur1, xs_ub[s_col, vr1_tok, 0, 0])
                    tr0 = V.vdiv(tr0, V.vadd(csum0_b, eps_b), mask_full, precision='exact')
                    tr1 = V.vdiv(tr1, V.vadd(csum1_b, eps_b), mask_full, precision='exact')
                    cur0 = V.vselr(tr0, t_idx)
                    cur1 = V.vselr(tr1, t_idx)

                # ==================== Phase 2: Backward iteration ====================
                # Ensure all Phase 1 stores are visible before Phase 2 loads
                T.simd.mem_bar('VST_VLD')

                # Load grad for both VRs
                g0 = V.alloc_var(T.float32, size=64)
                g1 = V.alloc_var(T.float32, size=64)
                g0 = V.vload(grad_ub[0, 0, 0], size=64)
                g1 = V.vload(grad_ub[vr1_tok, 0, 0], size=64)

                for inv_step in range(2 * repeat - 1):
                    s_idx = 2 * repeat - 1 - inv_step
                    # Load x_inter and sum from UB
                    xi0 = V.vload(xs_ub[s_idx, 0, 0, 0], size=64)
                    xi1 = V.vload(xs_ub[s_idx, vr1_tok, 0, 0], size=64)
                    sm0 = V.vload(sums_ub[s_idx, 0, 0, 0], size=64)
                    sm1 = V.vload(sums_ub[s_idx, vr1_tok, 0, 0], size=64)

                    if inv_step % 2 == 0:
                        # Col normalize backward:
                        # temp = grad * x_inter
                        # col_sum2 = sum(temp, dim=1)  -> need col_reduce
                        # col_sum2 /= sum + eps
                        # grad = (grad - col_sum2) / (sum + eps)
                        temp0 = V.vmul(g0, xi0)
                        temp1 = V.vmul(g1, xi1)
                        _, cs2_0_b = _col_reduce(temp0, t_idx, pair_swap_idx, cross_pair_idx)
                        _, cs2_1_b = _col_reduce(temp1, t_idx, pair_swap_idx, cross_pair_idx)
                        # cs2 is in transposed layout -- transpose back to original
                        cs2_0_b = V.vselr(cs2_0_b, t_idx)
                        cs2_1_b = V.vselr(cs2_1_b, t_idx)
                        cs2_0_b = V.vdiv(cs2_0_b, V.vadd(sm0, eps_b), mask_full, precision='exact')
                        cs2_1_b = V.vdiv(cs2_1_b, V.vadd(sm1, eps_b), mask_full, precision='exact')
                        g0 = V.vdiv(V.vsub(g0, cs2_0_b), V.vadd(sm0, eps_b), mask_full, precision='exact')
                        g1 = V.vdiv(V.vsub(g1, cs2_1_b), V.vadd(sm1, eps_b), mask_full, precision='exact')
                    else:
                        # Row normalize backward:
                        # temp = grad * x_inter
                        # row_sum2 = sum(temp, dim=2)  -> row_reduce
                        # row_sum2 /= sum + eps
                        # grad = (grad - row_sum2) / (sum + eps)
                        temp0 = V.vmul(g0, xi0)
                        temp1 = V.vmul(g1, xi1)
                        rs2_0_b = _row_reduce(temp0, pair_swap_idx, cross_pair_idx)
                        rs2_1_b = _row_reduce(temp1, pair_swap_idx, cross_pair_idx)
                        rs2_0_b = V.vdiv(rs2_0_b, V.vadd(sm0, eps_b), mask_full, precision='exact')
                        rs2_1_b = V.vdiv(rs2_1_b, V.vadd(sm1, eps_b), mask_full, precision='exact')
                        g0 = V.vdiv(V.vsub(g0, rs2_0_b), V.vadd(sm0, eps_b), mask_full, precision='exact')
                        g1 = V.vdiv(V.vsub(g1, rs2_1_b), V.vadd(sm1, eps_b), mask_full, precision='exact')

                # Final step: grad = (grad - row_sum(grad * x_inter)) * x_inter
                xi0 = V.vload(xs_ub[0, 0, 0, 0], size=64)
                xi1 = V.vload(xs_ub[0, vr1_tok, 0, 0], size=64)
                temp0 = V.vmul(g0, xi0)
                temp1 = V.vmul(g1, xi1)
                rs0_b = _row_reduce(temp0, pair_swap_idx, cross_pair_idx)
                rs1_b = _row_reduce(temp1, pair_swap_idx, cross_pair_idx)
                g0 = V.vmul(V.vsub(g0, rs0_b), xi0)
                g1 = V.vmul(V.vsub(g1, rs1_b), xi1)

                V.vstore(g0, out_ub[0, 0, 0])
                V.vstore(g1, out_ub[vr1_tok, 0, 0])

            T.copy(out_ub, grad_input[token_offset, 0, 0])
