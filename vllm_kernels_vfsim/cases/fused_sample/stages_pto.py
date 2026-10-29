"""PTO bodies for the fused_sample regions. Merging adds are add plus vsel."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_sample.stages_asc import MIN_PROB, N, VL, _empty

SCALE = float(1 << 31)


@tilelang.jit(target="pto")
def compile_search():
    @T.prim_func
    def kernel(
        probs: T.Tensor((N,), T.float32),
        quartiles: T.Tensor((3,), T.float32),
        sums: T.Tensor((3,), T.float32),
        counts: T.Tensor((3,), T.int32),
    ):
        with T.Kernel(1):
            probs_ub = T.alloc_shared((N,), T.float32)
            q_ub = T.alloc_shared((8,), T.float32)
            sum_ub = T.alloc_shared((8,), T.float32)
            count_ub = T.alloc_shared((8,), T.int32)
            T.copy(probs, probs_ub)
            T.copy(quartiles, q_ub[0:3])
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                one = V.create_mask(1, size=1)
                q = V.alloc_local((3,), V.vreg(VL, T.float32))
                acc = V.alloc_local((3,), V.vreg(VL, T.float32))
                cnt = V.alloc_local((3,), V.vreg(VL, T.float32))
                zero = V.vbrc(T.float32(0.0), size=VL)
                one_f = V.vbrc(T.float32(1.0), size=VL)
                for j in T.unroll(3, explicit=True):
                    q[j] = V.vbrc(V.vload(q_ub[j], size=1), size=VL)
                    acc[j] = zero
                    cnt[j] = zero
                for v in T.serial(N // VL):
                    p = V.vload(probs_ub[v * VL], size=VL)
                    for j in T.unroll(3, explicit=True):
                        pred = V.vcmp(p, q[j], full, "ge")
                        acc[j] = V.vsel(pred, V.vadd(acc[j], p, full), acc[j])
                        cnt[j] = V.vsel(pred, V.vadd(cnt[j], one_f, full), cnt[j])
                for j in T.unroll(3, explicit=True):
                    V.vstore(V.vcadd(acc[j], full, reassoc=True), sum_ub[j])
                    n = V.vcvt(V.vcadd(cnt[j], full, reassoc=True), "int32", rounding="Z")
                    V.vstore(n, count_ub[j])
            T.copy(sum_ub[0:3], sums)
            T.copy(count_ub[0:3], counts)

    return kernel


@tilelang.jit(target="pto")
def compile_filter():
    @T.prim_func
    def kernel(
        logits: T.Tensor((N,), T.float32),
        weights: T.Tensor((N,), T.float32),
        total: T.Tensor((1,), T.float32),
        count: T.Tensor((1,), T.int32),
        inv_temp: T.float32,
        lse0: T.float32,
        lse1: T.float32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((N,), T.float32)
            sum_ub = T.alloc_shared((8,), T.float32)
            count_ub = T.alloc_shared((8,), T.int32)
            T.copy(logits, ub)
            with T.SimdVF():
                V.vstore(V.vbrc(T.float32(0.0), size=1), sum_ub[0])
                V.vstore(V.vbrc(T.int32(0), size=1), count_ub[0])
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                one = V.create_mask(1, size=1)
                inv = V.vbrc(inv_temp, size=VL)
                l0 = V.vbrc(lse0, size=VL)
                zero = V.vbrc(T.float32(0.0), size=VL)
                one_i = V.vbrc(T.int32(1), size=VL)
                chunk_sum = V.alloc_var(T.float32, size=VL)
                chunk_count = V.alloc_var(T.int32, size=VL)
                chunk_sum = zero
                chunk_count = V.vbrc(T.int32(0), size=VL)
                normalized = V.alloc_var(T.float32, size=VL)
                for g in T.serial(N // VL):
                    base = g * VL
                    normalized = V.vadds(V.vload(ub[base], size=VL), lse1, full)
                    normalized = V.vadd(V.vmul(normalized, inv, full), l0, full)
                    prob = V.vexp(normalized, full)
                    eligible = V.vcmps(prob, T.float32(MIN_PROB), full, "ge")
                    weight = V.vsel(eligible, prob, zero)
                    chunk_count = V.vsel(eligible, V.vadd(chunk_count, one_i, full), chunk_count)
                    V.vstore(weight, ub[base], full)
                    chunk_sum = V.vadd(chunk_sum, weight, full)
                V.vstore(V.vcadd(chunk_sum, full, reassoc=True), sum_ub[0])
                V.vstore(V.vcadd(chunk_count, full), count_ub[0])
            T.copy(ub, weights)
            T.copy(sum_ub[0:1], total)
            T.copy(count_ub[0:1], count)

    return kernel


@tilelang.jit(target="pto")
def compile_quartile():
    @T.prim_func
    def kernel(
        lo_hi: T.Tensor((2,), T.float32),
        mass_in: T.Tensor((6,), T.float32),
        count_in: T.Tensor((6,), T.int32),
        target: T.float32,
        row_k: T.int32,
        quartiles: T.Tensor((3,), T.float32),
        new_lo: T.Tensor((1,), T.float32),
        new_hi: T.Tensor((1,), T.float32),
        n_keep: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(1):
            bounds = T.alloc_shared((8,), T.float32)
            sums = T.alloc_shared((8,), T.float32)
            counts = T.alloc_shared((8,), T.int32)
            q_ub = T.alloc_shared((8,), T.float32)
            lo_ub = T.alloc_shared((8,), T.float32)
            hi_ub = T.alloc_shared((8,), T.float32)
            keep_ub = T.alloc_shared((8,), T.int32)
            T.copy(lo_hi, bounds[0:2])
            T.copy(mass_in, sums[0:6])
            T.copy(count_in, counts[0:6])
            with T.SimdVF():
                one = V.create_mask(1, size=1)
                minimum = V.vbrc(T.float32(MIN_PROB), size=1)
                tgt = V.vmax(minimum, V.vbrc(target, size=1), one)
                V.vstore(tgt, sums[1])
                k = V.vmin(V.vbrc(row_k, size=1), V.vload(counts[0], size=1), one)
                V.vstore(k, counts[1])
                V.vstore(minimum, bounds[0])
                V.vstore(V.vbrc(T.float32(1.0 + 1e-3), size=1), bounds[1])
            with T.SimdVF():
                one = V.create_mask(1, size=1)
                left = V.vinterpret_cast(V.vload(bounds[0], size=1), "int32")
                right = V.vinterpret_cast(V.vload(bounds[1], size=1), "int32")
                width = V.vsub(right, left, one)
                q0 = V.vadd(left, V.vshrs(width, 2, one), one)
                q1 = V.vadd(left, V.vshrs(width, 1, one), one)
                q2 = V.vsub(right, V.vshrs(V.vadds(width, T.int32(3), one), 2, one), one)
                V.vstore(V.vinterpret_cast(q0, "float32"), q_ub[0])
                V.vstore(V.vinterpret_cast(q1, "float32"), q_ub[1])
                V.vstore(V.vinterpret_cast(q2, "float32"), q_ub[2])
            with T.SimdVF():
                one = V.create_mask(1, size=1)
                target_v = V.vload(sums[1], size=1)
                lo = V.alloc_var(T.float32, size=1)
                hi = V.alloc_var(T.float32, size=1)
                lo = V.vload(bounds[0], size=1)
                hi = V.vload(q_ub[0], size=1)
                for j in T.unroll(3, explicit=True):
                    hit = V.vcmp(V.vload(sums[2 + j], size=1), target_v, one, "ge")
                    lo = V.vsel(hit, V.vload(q_ub[j], size=1), lo)
                    nxt = V.vload(q_ub[j + 1], size=1) if j < 2 else V.vload(bounds[1], size=1)
                    hi = V.vsel(hit, nxt, hi)
                V.vstore(lo, lo_ub[0])
                V.vstore(hi, hi_ub[0])
            with T.SimdVF():
                one = V.create_mask(1, size=1)
                need = V.vsub(V.vload(sums[1], size=1), V.vload(sums[5], size=1), one)
                n = V.vdiv(need, V.vload(bounds[0], size=1), one)
                bounded = V.vmins(V.vmaxs(n, T.float32(0.0), one), T.float32(N), one)
                # VMI float-to-int rounding has no ceiling mode. Non-negative
                # ceiling is the toward-zero value, plus one when a fraction remains.
                truncated = V.vcvt(bounded, "int32", rounding="Z")
                whole = V.vcmp(V.vcvt(truncated, "float32"), bounded, one, "eq")
                ceiled = V.vsel(whole, truncated, V.vadds(truncated, T.int32(1), one))
                V.vstore(ceiled, keep_ub[0])
            T.copy(q_ub[0:3], quartiles)
            T.copy(lo_ub[0:1], new_lo)
            T.copy(hi_ub[0:1], new_hi)
            T.copy(keep_ub[0:1], n_keep)

    return kernel


_SEARCH = None
_FILTER = None
_QUARTILE = None


def launch_search(probs, quartiles):
    global _SEARCH
    if _SEARCH is None:
        _SEARCH = compile_search()
    src = probs.npu()
    sums = _empty(src, (3,))
    counts = _empty(src, (3,), dtype=__import__("torch").int32)
    _SEARCH(src, quartiles.npu(), sums, counts)
    return sums.cpu(), counts.cpu()


def launch_filter(logits, inv_temp, lse0, lse1):
    global _FILTER
    if _FILTER is None:
        _FILTER = compile_filter()
    import torch

    src = logits.npu()
    weights = torch.empty_like(src)
    total = torch.empty(1, device=src.device)
    count = torch.empty(1, dtype=torch.int32, device=src.device)
    _FILTER(src, weights, total, count, float(inv_temp), float(lse0), float(lse1))
    return weights.cpu(), total.cpu(), count.cpu()


def launch_quartile(lo_hi, mass_in, count_in, target, row_k):
    global _QUARTILE
    if _QUARTILE is None:
        _QUARTILE = compile_quartile()
    import torch

    src = lo_hi.npu()
    q = torch.empty(3, device=src.device)
    new_lo = torch.empty(1, device=src.device)
    new_hi = torch.empty(1, device=src.device)
    n_keep = torch.empty(1, dtype=torch.int32, device=src.device)
    _QUARTILE(src, mass_in.npu(), count_in.npu(), float(target), int(row_k), q, new_lo, new_hi, n_keep)
    return q.cpu(), new_lo.cpu(), new_hi.cpu(), n_keep.cpu()
