"""PTO sampler folds and inclusive scans.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L224
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from cases.random_sampling.asc import SCALE, SLOTS, VOCAB


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
def compile_kernel():
    @T.prim_func
    def kernel(
        probs: T.Tensor((VOCAB,), T.float32),
        scanned: T.Tensor((SLOTS,), T.int32),
        slot_scanned: T.Tensor((SLOTS,), T.int32),
    ):
        with T.Kernel(1):
            probs_ub = T.alloc_shared((VOCAB,), T.float32)
            acc_ub = T.alloc_shared((SLOTS,), T.int32)
            slot_ub = T.alloc_shared((SLOTS,), T.int32)
            T.copy(probs, probs_ub)
            with T.SimdVF():
                mask = V.create_mask(SLOTS, size=SLOTS)
                V.vstore(V.vbrc(T.int32(0), size=SLOTS), acc_ub[0], mask)
            with T.SimdVF():
                mask = V.create_mask(SLOTS, size=SLOTS)
                acc = V.alloc_var(T.int32, size=SLOTS)
                acc = V.vload(acc_ub[0], size=SLOTS)
                for t in T.serial(VOCAB // SLOTS):
                    prob = V.vload(probs_ub[t * SLOTS], size=SLOTS)
                    scaled = V.vmuls(prob, T.float32(SCALE), mask)
                    acc = V.vadd(acc, V.vcvt(scaled, "int32", rounding="Z"), mask)
                V.vstore(acc, acc_ub[0], mask)
            with T.SimdVF():
                _scan(acc_ub)
            with T.SimdVF():
                mask = V.create_mask(SLOTS, size=SLOTS)
                V.vstore(V.vload(acc_ub[0], size=SLOTS), slot_ub[0], mask)
            with T.SimdVF():
                _scan(slot_ub)
            T.copy(acc_ub, scanned)
            T.copy(slot_ub, slot_scanned)

    return kernel


_KERNEL = None


def launch(probs):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    probs_n = probs.npu()
    scanned = torch.empty(SLOTS, dtype=torch.int32, device=probs_n.device)
    slot = torch.empty(SLOTS, dtype=torch.int32, device=probs_n.device)
    _KERNEL(probs_n, scanned, slot)
    return torch.stack((scanned.cpu(), slot.cpu()))
