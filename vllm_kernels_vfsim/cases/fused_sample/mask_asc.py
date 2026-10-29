"""Mask, log-mass, and CDF SimdVF regions from fused_sample.

ASC source spills the keep predicate with pst/pld. VMI has no predicate
memory ops, so both drivers store the predicate as 0/1 and reload it.
The ASC pst form is what failed to lower in this toolchain; the 0/1 form
is the region that runs. See the case README.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

N = 128
VL = 64
SLOTS = 64
SCALE = float(1 << 31)


@tilelang.jit(target="ascend")
def compile_mask():
    @T.prim_func
    def kernel(
        probs: T.Tensor((N,), T.float32),
        logits: T.Tensor((N,), T.float32),
        boundary: T.float32,
        inv_temp: T.float32,
        lse0: T.float32,
        lse1: T.float32,
        mask: T.Tensor((N,), T.int32),
        log_mass: T.Tensor((1,), T.float32),
        filtered: T.Tensor((N,), T.float32),
        plain: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(1):
            prob_ub = T.alloc_shared((N,), T.float32)
            logit_ub = T.alloc_shared((N,), T.float32)
            mask_ub = T.alloc_shared((N,), T.int32)
            mass_ub = T.alloc_shared((8,), T.float32)
            log_ub = T.alloc_shared((8,), T.float32)
            T.copy(probs, prob_ub)
            T.copy(logits, logit_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                bound = S.vdup(boundary, T.float32, full)
                zero = S.vdup(0, T.int32, full)
                one_i = S.vdup(1, T.int32, full)
                count = S.alloc_var(T.int32)
                count = zero
                for v in T.serial(N // VL):
                    base = v * VL
                    candidate = S.vcmp(S.vld(prob_ub[base]), bound, full, "eq")
                    S.vsts(mask_ub[base], S.vsel(one_i, zero, candidate), full)
                    count = S.vadd(count, one_i, candidate, "MODE_MERGING")
                S.vsts(mask_ub[0], S.vcadd(count, full), one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                one = S.pset(32, "PAT_VL1")
                bound = S.vdup(boundary, T.float32, full)
                kept = S.alloc_var(T.float32)
                kept = S.vdup(0.0, T.float32, full)
                zero_i = S.vdup(0, T.int32, full)
                one_i = S.vdup(1, T.int32, full)
                for v in T.serial(N // VL):
                    base = v * VL
                    prob = S.vld(prob_ub[base])
                    keep = S.vcmp(prob, bound, full, "ge")
                    S.vsts(mask_ub[base], S.vsel(one_i, zero_i, keep), full)
                    kept = S.vadd(kept, prob, keep, "MODE_MERGING")
                S.vsts(mass_ub[0], S.vcadd(kept, full), one, "ONEPT_B32")
            with T.SimdVF():
                one = S.pset(32, "PAT_VL1")
                log_m = S.vln(S.vld(mass_ub[0], dist="BRC_B32"), one)
                S.vsts(log_ub[0], log_m, one, "ONEPT_B32")
            with T.SimdVF():
                full = S.pset(32)
                neg_inf = S.vdup(float("-inf"), T.float32, full)
                inv = S.vdup(inv_temp, T.float32, full)
                l0 = S.vdup(lse0, T.float32, full)
                log_kept = S.vld(log_ub[0], dist="BRC_B32")
                normalized = S.alloc_var(T.float32)
                for g in T.serial(N // VL):
                    base = g * VL
                    raw = S.vld(logit_ub[base])
                    keep = S.vcmps(S.vld(mask_ub[base]), 0, full, "ne")
                    normalized = S.vadds(raw, lse1, full)
                    S.vmadd(normalized, inv, l0, full)
                    res = S.vsub(normalized, log_kept, full)
                    S.vsts(prob_ub[base], S.vsel(res, neg_inf, keep), full)
            with T.SimdVF():
                full = S.pset(32)
                inv = S.vdup(inv_temp, T.float32, full)
                max_vec = S.vneg(S.vdup(lse1, T.float32, full), full)
                logsum_vec = S.vneg(S.vdup(lse0, T.float32, full), full)
                for g in T.serial(N // VL):
                    base = g * VL
                    shifted = S.vsub(S.vld(logit_ub[base]), max_vec, full)
                    scaled = S.vmul(shifted, inv, full)
                    loss = S.vsub(logsum_vec, scaled, full)
                    S.vsts(logit_ub[base], S.vneg(loss, full), full)
            T.copy(mask_ub, mask)
            T.copy(log_ub[0:1], log_mass)
            T.copy(prob_ub, filtered)
            T.copy(logit_ub, plain)

    return kernel


@T.macro
def _scan(buf):
    mask = S.pset(32)
    vals = S.alloc_var(T.int32)
    vals = S.vld(buf[0])
    base = S.vci(0, "int32")
    zero = S.vdup(0, T.int32, mask)
    for i in T.unroll(6, explicit=True):
        step = 1 << i
        index = S.vadds(base, T.int32(-step), mask)
        shuffled = S.vselr(vals, T.reinterpret("uint32x64", index))
        keep = S.vcmps(base, step, mask, "ge")
        vals = S.vadd(vals, S.vsel(shuffled, zero, keep), mask)
    S.vsts(buf[0], vals, mask)


@tilelang.jit(target="ascend")
def compile_cdf():
    @T.prim_func
    def kernel(
        logprobs: T.Tensor((N,), T.float32),
        scanned: T.Tensor((SLOTS,), T.int32),
        slot: T.Tensor((SLOTS,), T.int32),
    ):
        with T.Kernel(1):
            src = T.alloc_shared((N,), T.float32)
            acc = T.alloc_shared((SLOTS,), T.int32)
            slot_ub = T.alloc_shared((SLOTS,), T.int32)
            T.copy(logprobs, src)
            with T.SimdVF():
                full = S.pset(32)
                zero = S.vdup(0, T.int32, full)
                S.vsts(acc[0], zero, full)
            with T.SimdVF():
                full = S.pset(32)
                total = S.alloc_var(T.int32)
                total = S.vld(acc[0])
                for t in T.serial(N // SLOTS):
                    prob = S.vexp(S.vld(src[t * SLOTS]), full)
                    scaled = S.vmuls(prob, T.float32(SCALE), full)
                    inc = S.vcvt(scaled, "int32", full, round="ROUND_Z", sat=True)
                    total = S.vadd(total, inc, full)
                S.vsts(acc[0], total, full)
            with T.SimdVF():
                _scan(acc)
            with T.SimdVF():
                full = S.pset(32)
                S.vsts(slot_ub[0], S.vld(acc[0]), full)
            with T.SimdVF():
                _scan(slot_ub)
            T.copy(acc, scanned)
            T.copy(slot_ub, slot)

    return kernel


_MASK = None
_CDF = None


def launch_mask(probs, logits, boundary, inv_temp, lse0, lse1):
    global _MASK
    if _MASK is None:
        _MASK = compile_mask()
    import torch

    src = probs.npu()
    device = src.device
    mask = torch.empty(N, dtype=torch.int32, device=device)
    log_mass = torch.empty(1, device=device)
    filtered = torch.empty(N, device=device)
    plain = torch.empty(N, device=device)
    _MASK(src, logits.npu(), float(boundary), float(inv_temp), float(lse0), float(lse1), mask, log_mass, filtered, plain)
    return mask.cpu(), log_mass.cpu(), filtered.cpu(), plain.cpu()


def launch_cdf(logprobs):
    global _CDF
    if _CDF is None:
        _CDF = compile_cdf()
    import torch

    src = logprobs.npu()
    scanned = torch.empty(SLOTS, dtype=torch.int32, device=src.device)
    slot = torch.empty(SLOTS, dtype=torch.int32, device=src.device)
    _CDF(src, scanned, slot)
    return torch.stack((scanned.cpu(), slot.cpu()))
