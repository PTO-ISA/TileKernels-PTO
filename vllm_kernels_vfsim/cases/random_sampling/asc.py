"""ASC sampler folds and inclusive scans. Vocab 128, 64 slots.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L224
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L232
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L235
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L252

SimtVF sample selection is out of scope. The second scan is the slot CDF scan.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

VOCAB = 128
SLOTS = 64
SCALE = float(1 << 31)


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
                mask = S.pset(32)
                S.vsts(acc_ub[0], S.vdup(0, T.int32, mask), mask)
            with T.SimdVF():
                mask = S.pset(32)
                acc = S.alloc_var(T.int32)
                acc = S.vld(acc_ub[0])
                for t in T.serial(VOCAB // SLOTS):
                    prob = S.vld(probs_ub[t * SLOTS])
                    scaled = S.vmuls(prob, T.float32(SCALE), mask)
                    inc = S.vcvt(scaled, "int32", mask, round="ROUND_Z", sat=True)
                    acc = S.vadd(acc, inc, mask)
                S.vsts(acc_ub[0], acc, mask)
            with T.SimdVF():
                _scan(acc_ub)
            with T.SimdVF():
                mask = S.pset(32)
                S.vsts(slot_ub[0], S.vld(acc_ub[0]), mask)
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
