"""ASC bodies for the fused_sample SimdVF regions the lse driver does not run.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 128
VL = 64
MIN_PROB = 2.0 ** -32
SCALE = float(1 << 31)


def _empty(like, shape, dtype=None):
    import torch

    return torch.empty(shape, dtype=dtype or like.dtype, device=like.device)


@tilelang.jit(target="ascend")
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
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                q = S.alloc_local((3,), T.float32)
                acc = S.alloc_local((3,), T.float32)
                cnt = S.alloc_local((3,), T.float32)
                one_f = S.vdup(1.0, T.float32, full)
                for j in T.unroll(3, explicit=True):
                    q[j] = S.vld(q_ub[j], dist="BRC_B32")
                    acc[j] = S.vdup(0.0, T.float32, full)
                    cnt[j] = S.vdup(0.0, T.float32, full)
                for v in T.serial(N // VL):
                    p = S.vld(probs_ub[v * VL])
                    for j in T.unroll(3, explicit=True):
                        pred = S.vcmp(p, q[j], full, "ge")
                        acc[j] = S.vadd(acc[j], p, pred, "MODE_MERGING")
                        cnt[j] = S.vadd(cnt[j], one_f, pred, "MODE_MERGING")
                for j in T.unroll(3, explicit=True):
                    S.vsts(sum_ub[j], S.vcadd(acc[j], full), one, "ONEPT_B32")
                    n = S.vcvt(S.vcadd(cnt[j], full), T.int32, one, round="ROUND_Z")
                    S.vsts(count_ub[j], n, one, "ONEPT_B32")
            T.copy(sum_ub[0:3], sums)
            T.copy(count_ub[0:3], counts)

    return kernel


@tilelang.jit(target="ascend")
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
                one = S.pset(32, "PAT_VL1")
                S.vsts(sum_ub[0], S.vdup(0.0, T.float32, one), one, "ONEPT_B32")
                S.vsts(count_ub[0], S.vdup(0, T.int32, one), one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                inv = S.vdup(inv_temp, T.float32, full)
                l0 = S.vdup(lse0, T.float32, full)
                zero = S.vdup(0.0, T.float32, full)
                one_i = S.vdup(1, T.int32, full)
                chunk_sum = S.alloc_var(T.float32)
                chunk_count = S.alloc_var(T.int32)
                chunk_sum = zero
                chunk_count = S.vdup(0, T.int32, full)
                normalized = S.alloc_var(T.float32)
                for g in T.serial(N // VL):
                    base = g * VL
                    normalized = S.vadds(S.vld(ub[base]), lse1, full)
                    S.vmadd(normalized, inv, l0, full)
                    prob = S.vexp(normalized, full)
                    eligible = S.vcmps(prob, MIN_PROB, full, "ge")
                    weight = S.vsel(prob, zero, eligible)
                    chunk_count = S.vadd(chunk_count, one_i, eligible, "MODE_MERGING")
                    S.vsts(ub[base], weight, full)
                    chunk_sum = S.vadd(chunk_sum, weight, full)
                S.vsts(sum_ub[0], S.vcadd(chunk_sum, full), one, "ONEPT_B32")
                S.vsts(count_ub[0], S.vcadd(chunk_count, full), one, "ONEPT_B32")
            T.copy(ub, weights)
            T.copy(sum_ub[0:1], total)
            T.copy(count_ub[0:1], count)

    return kernel


@tilelang.jit(target="ascend")
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
            T.copy(lo_hi, bounds[0:2])
            T.copy(mass_in, sums[0:6])
            T.copy(count_in, counts[0:6])
            with T.SimdVF():
                one = S.pset(32, "PAT_VL1")
                minimum = S.vdup(MIN_PROB, T.float32, one)
                mass = S.vld(sums[0], dist="BRC_B32")
                tgt = S.vmax(minimum, S.vdup(target, T.float32, one), one)
                S.vsts(sums[1], tgt, one, "ONEPT_B32")
                k = S.vmin(S.vdup(row_k, T.int32, one), S.vld(counts[0], dist="BRC_B32"), one)
                S.vsts(counts[1], k, one, "ONEPT_B32")
                S.vsts(bounds[0], minimum, one, "ONEPT_B32")
                S.vsts(bounds[1], S.vdup(1.0 + 1e-3, T.float32, one), one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                left = T.reinterpret(S.vld(bounds[0], dist="BRC_B32"), "int32x64")
                right = T.reinterpret(S.vld(bounds[1], dist="BRC_B32"), "int32x64")
                width = S.vsub(right, left, full)
                quarter = S.vshrs(width, 2, full)
                q0 = S.vadd(left, quarter, full)
                q1 = S.vadd(left, S.vshrs(width, 1, full), full)
                q2 = S.vsub(right, S.vshrs(S.vadds(width, 3, full), 2, full), full)
                S.vsts(q_ub[0], T.reinterpret(q0, "float32x64"), one, "ONEPT_B32")
                S.vsts(q_ub[1], T.reinterpret(q1, "float32x64"), one, "ONEPT_B32")
                S.vsts(q_ub[2], T.reinterpret(q2, "float32x64"), one, "ONEPT_B32")
            with T.SimdVF():
                one = S.pset(32, "PAT_VL1")
                hits = S.alloc_local((3,), "bool")
                for j in T.unroll(3, explicit=True):
                    hits[j] = S.vcmp(S.vld(sums[2 + j], dist="BRC_B32"), S.vld(sums[1], dist="BRC_B32"), one, "ge")
                lo = S.alloc_var(T.float32)
                hi = S.alloc_var(T.float32)
                lo = S.vld(bounds[0], dist="BRC_B32")
                hi = S.vld(q_ub[0], dist="BRC_B32")
                for j in T.unroll(3, explicit=True):
                    lo = S.vsel(S.vld(q_ub[j], dist="BRC_B32"), lo, hits[j])
                    nxt = S.vld(q_ub[j + 1], dist="BRC_B32") if j < 2 else S.vld(bounds[1], dist="BRC_B32")
                    hi = S.vsel(nxt, hi, hits[j])
                S.vsts(bounds[2], lo, one, "ONEPT_B32")
                S.vsts(bounds[3], hi, one, "ONEPT_B32")
            with T.SimdVF():
                one = S.pset(32, "PAT_VL1")
                need = S.vsub(S.vld(sums[1], dist="BRC_B32"), S.vld(sums[5], dist="BRC_B32"), one)
                n = S.vdiv(need, S.vld(bounds[0], dist="BRC_B32"), one)
                bounded = S.vmins(S.vmaxs(n, 0.0, one), T.float32(N), one)
                S.vsts(counts[2], S.vcvt(bounded, T.int32, one, round="ROUND_C"), one, "ONEPT_B32")
            T.copy(q_ub[0:3], quartiles)
            T.copy(bounds[2:3], new_lo)
            T.copy(bounds[3:4], new_hi)
            T.copy(counts[2:3], n_keep)

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
