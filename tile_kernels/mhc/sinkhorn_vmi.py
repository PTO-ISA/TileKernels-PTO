import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


def _row_max(mat, mask_lo, mask_hi):
    """Broadcast row max: masked grouped vcmax x2 + group-8 vbrc + vsel (ASC structure).

    Mirrors the ASC vcgmax/vselr/vsel sequence: each 8-lane group holds two
    4-lane rows; vcmax with the half-group mask reduces each row, vbrc(group=8)
    broadcasts the compact result inside its group, and vsel picks per lane.
    """
    lo = V.vbrc(V.vcmax(mat, mask_lo, group=8), size=64, group=8)
    hi = V.vbrc(V.vcmax(mat, mask_hi, group=8), size=64, group=8)
    return V.vsel(mask_lo, lo, hi)


def _row_reduce(mat, mask_lo, mask_hi):
    """Broadcast row sum: masked grouped vcadd x2 + group-8 vbrc + vsel (ASC structure).

    ``reassoc=True`` keeps the ASC ``vcgadd`` two-pair fold order
    ((a+b)+(c+d) per 4-lane row), so every fp32 rounding step matches.
    """
    lo = V.vbrc(V.vcadd(mat, mask_lo, group=8, reassoc=True), size=64, group=8)
    hi = V.vbrc(V.vcadd(mat, mask_hi, group=8, reassoc=True), size=64, group=8)
    return V.vsel(mask_lo, lo, hi)


def _col_reduce(mat, mask_lo, mask_hi, t_idx):
    """Col reduce: transpose, row reduce -> (transposed, broadcast col sum).

    NOTE: The returned broadcast col sum is in TRANSPOSED layout (each lane
    gets its transposed-row sum = original-col sum). To use with original-layout
    data, transpose the broadcast result back via V.vselr(csum_b, t_idx).
    """
    tr = V.vselr(mat, t_idx)
    return tr, _row_reduce(tr, mask_lo, mask_hi)


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

                # Half-group masks for the grouped row reductions: mask_lo keeps
                # the first 4 lanes of each 8-lane group (the ASC per-VLane
                # [1,1,1,1,0,0,0,0] pattern, built like the ASC original).
                const_7 = V.vbrc(T.int32(7), size=64)
                mod8 = V.vand(idx, const_7)
                mask_lo = V.vcmps(mod8, T.int32(4), mask_full, 'lt')
                mask_hi = V.vcmps(mod8, T.int32(4), mask_full, 'ge')
                eps_b = V.vbrc(T.float32(eps), size=64)

                # Load 4 tokens x 16 elements = 64 fp32
                mat_0 = V.vload(comb_in_ub[0, 0, 0], size=64)

                # Step 1: Row softmax + eps
                rmax_b_0 = _row_max(mat_0, mask_lo, mask_hi)
                mat_1 = V.vexpdif(mat_0, rmax_b_0, mask_full)
                rsum_b_0 = _row_reduce(mat_1, mask_lo, mask_hi)
                mat_2 = V.vdiv(mat_1, rsum_b_0, mask_full, precision='exact')
                mat_3 = V.vadd(mat_2, eps_b)

                # Step 2: Col normalize (transpose, reduce, normalize, transpose back)
                tr_0 = V.vselr(mat_3, t_idx)
                csum_b_0 = _row_reduce(tr_0, mask_lo, mask_hi)
                tr_1 = V.vdiv(tr_0, V.vadd(csum_b_0, eps_b), mask_full, precision='exact')
                mat_4 = V.vselr(tr_1, t_idx)

                # Step 3: repeat-1 iterations (Python range() unroll)
                # Use alloc_var for cur to allow cross-iteration assignment.
                # Pre-loop uses unique variable names (mat_0..mat_4) to avoid
                # CSE matching loop body expressions against pre-loop SSA values.
                cur = V.alloc_var(T.float32, size=64)
                cur = mat_4
                for _step in range(repeat - 1):
                    rsum_b = _row_reduce(cur, mask_lo, mask_hi)
                    cur = V.vdiv(cur, V.vadd(rsum_b, eps_b), mask_full, precision='exact')

                    tr = V.vselr(cur, t_idx)
                    csum_b = _row_reduce(tr, mask_lo, mask_hi)
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
                const_7 = V.vbrc(T.int32(7), size=64)
                mod8 = V.vand(idx, const_7)
                mask_lo = V.vcmps(mod8, T.int32(4), mask_full, 'lt')
                mask_hi = V.vcmps(mod8, T.int32(4), mask_full, 'ge')
                eps_b = V.vbrc(T.float32(eps), size=64)

                # VR1 starts at token 4 (4 tokens per VR)
                vr1_tok = 4

                # ==================== Phase 1: Forward recompute ====================
                # Load both VRs
                x0_0 = V.vload(x_ub[0, 0, 0], size=64)  # VR0: tokens 0-3
                x1_0 = V.vload(x_ub[vr1_tok, 0, 0], size=64)  # VR1: tokens 4-7

                # Softmax + eps
                rmax0_b = _row_max(x0_0, mask_lo, mask_hi)
                rmax1_b = _row_max(x1_0, mask_lo, mask_hi)
                x0_1 = V.alloc_var(T.float32, size=64)
                x1_1 = V.alloc_var(T.float32, size=64)
                x0_1 = V.vexpdif(x0_0, rmax0_b, mask_full)
                x1_1 = V.vexpdif(x1_0, rmax1_b, mask_full)
                rsum0_b = _row_reduce(x0_1, mask_lo, mask_hi)
                rsum1_b = _row_reduce(x1_1, mask_lo, mask_hi)
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
                tr0_0, csum0_b = _col_reduce(x0_3, mask_lo, mask_hi, t_idx)
                tr1_0, csum1_b = _col_reduce(x1_3, mask_lo, mask_hi, t_idx)
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
                    rsum0_b = _row_reduce(cur0, mask_lo, mask_hi)
                    rsum1_b = _row_reduce(cur1, mask_lo, mask_hi)
                    V.vstore(rsum0_b, sums_ub[s_row, 0, 0, 0])
                    V.vstore(rsum1_b, sums_ub[s_row, vr1_tok, 0, 0])
                    V.vstore(cur0, xs_ub[s_row, 0, 0, 0])
                    V.vstore(cur1, xs_ub[s_row, vr1_tok, 0, 0])
                    cur0 = V.vdiv(cur0, V.vadd(rsum0_b, eps_b), mask_full, precision='exact')
                    cur1 = V.vdiv(cur1, V.vadd(rsum1_b, eps_b), mask_full, precision='exact')

                    # Col normalize
                    tr0, csum0_b = _col_reduce(cur0, mask_lo, mask_hi, t_idx)
                    tr1, csum1_b = _col_reduce(cur1, mask_lo, mask_hi, t_idx)
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
                        _, cs2_0_b = _col_reduce(temp0, mask_lo, mask_hi, t_idx)
                        _, cs2_1_b = _col_reduce(temp1, mask_lo, mask_hi, t_idx)
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
                        rs2_0_b = _row_reduce(temp0, mask_lo, mask_hi)
                        rs2_1_b = _row_reduce(temp1, mask_lo, mask_hi)
                        rs2_0_b = V.vdiv(rs2_0_b, V.vadd(sm0, eps_b), mask_full, precision='exact')
                        rs2_1_b = V.vdiv(rs2_1_b, V.vadd(sm1, eps_b), mask_full, precision='exact')
                        g0 = V.vdiv(V.vsub(g0, rs2_0_b), V.vadd(sm0, eps_b), mask_full, precision='exact')
                        g1 = V.vdiv(V.vsub(g1, rs2_1_b), V.vadd(sm1, eps_b), mask_full, precision='exact')

                # Final step: grad = (grad - row_sum(grad * x_inter)) * x_inter
                xi0 = V.vload(xs_ub[0, 0, 0, 0], size=64)
                xi1 = V.vload(xs_ub[0, vr1_tok, 0, 0], size=64)
                temp0 = V.vmul(g0, xi0)
                temp1 = V.vmul(g1, xi1)
                rs0_b = _row_reduce(temp0, mask_lo, mask_hi)
                rs1_b = _row_reduce(temp1, mask_lo, mask_hi)
                g0 = V.vmul(V.vsub(g0, rs0_b), xi0)
                g1 = V.vmul(V.vsub(g1, rs1_b), xi1)

                V.vstore(g0, out_ub[0, 0, 0])
                V.vstore(g1, out_ub[vr1_tok, 0, 0])

            T.copy(out_ub, grad_input[token_offset, 0, 0])
