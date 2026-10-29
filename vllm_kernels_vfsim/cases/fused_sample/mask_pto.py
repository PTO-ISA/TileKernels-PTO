"""PTO mask and CDF. The keep predicate is a 0/1 vector, not a predicate spill."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.fused_sample.mask_asc import N, SCALE, SLOTS, VL


@tilelang.jit(target="pto")
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
                full = V.create_mask(VL, size=VL)
                bound = V.vbrc(boundary, size=VL)
                zero = V.vbrc(T.int32(0), size=VL)
                one_i = V.vbrc(T.int32(1), size=VL)
                count = V.alloc_var(T.int32, size=VL)
                count = zero
                for v in T.serial(N // VL):
                    base = v * VL
                    candidate = V.vcmp(V.vload(prob_ub[base], size=VL), bound, full, "eq")
                    V.vstore(V.vsel(candidate, one_i, zero), mask_ub[base], full)
                    count = V.vsel(candidate, V.vadd(count, one_i, full), count)
                V.vstore(V.vcadd(count, full), mask_ub[0])
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                bound = V.vbrc(boundary, size=VL)
                kept = V.alloc_var(T.float32, size=VL)
                kept = V.vbrc(T.float32(0.0), size=VL)
                zero_i = V.vbrc(T.int32(0), size=VL)
                one_i = V.vbrc(T.int32(1), size=VL)
                for v in T.serial(N // VL):
                    base = v * VL
                    prob = V.vload(prob_ub[base], size=VL)
                    keep = V.vcmp(prob, bound, full, "ge")
                    V.vstore(V.vsel(keep, one_i, zero_i), mask_ub[base], full)
                    kept = V.vsel(keep, V.vadd(kept, prob, full), kept)
                V.vstore(V.vcadd(kept, full, reassoc=True), mass_ub[0])
            with T.SimdVF():
                logged = V.vln(V.vload(mass_ub[0], size=1), V.create_mask(1, size=1))
                V.vstore(logged, log_ub[0])
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                neg_inf = V.vbrc(T.float32(float("-inf")), size=VL)
                inv = V.vbrc(inv_temp, size=VL)
                l0 = V.vbrc(lse0, size=VL)
                log_kept = V.vselr(V.vload(log_ub[0], size=VL), V.vbrc(T.uint32(0), size=VL))
                normalized = V.alloc_var(T.float32, size=VL)
                for g in T.serial(N // VL):
                    base = g * VL
                    raw = V.vload(logit_ub[base], size=VL)
                    keep = V.vcmps(V.vload(mask_ub[base], size=VL), T.int32(0), full, "ne")
                    normalized = V.vadds(raw, lse1, full)
                    normalized = V.vadd(V.vmul(normalized, inv, full), l0, full)
                    res = V.vsub(normalized, log_kept, full)
                    V.vstore(V.vsel(keep, res, neg_inf), prob_ub[base], full)
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                inv = V.vbrc(inv_temp, size=VL)
                max_vec = V.vneg(V.vbrc(lse1, size=VL), full)
                logsum_vec = V.vneg(V.vbrc(lse0, size=VL), full)
                for g in T.serial(N // VL):
                    base = g * VL
                    shifted = V.vsub(V.vload(logit_ub[base], size=VL), max_vec, full)
                    scaled = V.vmul(shifted, inv, full)
                    loss = V.vsub(logsum_vec, scaled, full)
                    V.vstore(V.vneg(loss, full), logit_ub[base], full)
            T.copy(mask_ub, mask)
            T.copy(log_ub[0:1], log_mass)
            T.copy(prob_ub, filtered)
            T.copy(logit_ub, plain)

    return kernel


@T.macro
def _scan(buf):
    mask = V.create_mask(SLOTS, size=SLOTS)
    vals = V.alloc_var(T.int32, size=SLOTS)
    vals = V.vload(buf[0], size=SLOTS)
    base = V.vci(T.int32(0), size=SLOTS)
    zero = V.vbrc(T.int32(0), size=SLOTS)
    for i in T.unroll(6, explicit=True):
        step = 1 << i
        index = V.vadds(base, T.int32(-step), mask)
        shuffled = V.vselr(vals, V.vinterpret_cast(index, "uint32"))
        keep = V.vcmps(base, T.int32(step), mask, "ge")
        vals = V.vadd(vals, V.vsel(keep, shuffled, zero), mask)
    V.vstore(vals, buf[0], mask)


@tilelang.jit(target="pto")
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
                full = V.create_mask(SLOTS, size=SLOTS)
                V.vstore(V.vbrc(T.int32(0), size=SLOTS), acc[0], full)
            with T.SimdVF():
                full = V.create_mask(SLOTS, size=SLOTS)
                total = V.alloc_var(T.int32, size=SLOTS)
                total = V.vload(acc[0], size=SLOTS)
                for t in T.serial(N // SLOTS):
                    prob = V.vexp(V.vload(src[t * SLOTS], size=SLOTS), full)
                    scaled = V.vmuls(prob, T.float32(SCALE), full)
                    total = V.vadd(total, V.vcvt(scaled, "int32", rounding="Z"), full)
                V.vstore(total, acc[0], full)
            with T.SimdVF():
                _scan(acc)
            with T.SimdVF():
                full = V.create_mask(SLOTS, size=SLOTS)
                V.vstore(V.vload(acc[0], size=SLOTS), slot_ub[0], full)
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
