"""PTO sum of exp(logprob) at and before the sampled token.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sum_smaller_probs.py#L58

VMI ``pmode`` is zeroing only, so each ASC merging add is an add plus ``vsel``.
Predicate OR is ``V.vor`` on masks.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

VOCAB = 128
VL = 64
SAMPLE = 70
SPLIT = SAMPLE // VL
LANE = SAMPLE % VL


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        logprobs: T.Tensor((VOCAB,), T.float32),
        pivot: T.float32,
        out: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(1):
            logprobs_ub = T.alloc_shared((VOCAB,), T.float32)
            out_ub = T.alloc_shared((1,), T.float32)
            T.copy(logprobs, logprobs_ub)
            with T.SimdVF():
                full = V.create_mask(VL, size=VL)
                pivot_v = V.vbrc(pivot, size=VL)
                acc = V.alloc_var(T.float32, size=VL)
                acc = V.vbrc(T.float32(0.0), size=VL)
                for v in T.serial(SPLIT):
                    val = V.vload(logprobs_ub[v * VL], size=VL)
                    prob = V.vexp(val, full)
                    pred = V.vcmp(val, pivot_v, full, "le")
                    acc = V.vsel(pred, V.vadd(acc, prob, full), acc)
                for v in T.serial(SPLIT + 1, VOCAB // VL):
                    val = V.vload(logprobs_ub[v * VL], size=VL)
                    prob = V.vexp(val, full)
                    pred = V.vcmp(val, pivot_v, full, "lt")
                    acc = V.vsel(pred, V.vadd(acc, prob, full), acc)
                lane = V.vci(T.int32(0), size=VL)
                val = V.vload(logprobs_ub[SPLIT * VL], size=VL)
                prob = V.vexp(val, full)
                less_pred = V.vcmp(val, pivot_v, full, "lt")
                before_pred = V.vcmps(lane, T.int32(LANE), full, "le")
                equal_pred = V.vcmp(val, pivot_v, before_pred, "eq")
                pred = V.vor(less_pred, equal_pred)
                acc = V.vsel(pred, V.vadd(acc, prob, full), acc)
                V.vstore(V.vcadd(acc, full, reassoc=True), out_ub[0])
            T.copy(out_ub, out)

    return kernel


_KERNEL = None


def launch(logprobs, pivot):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    src = logprobs.npu()
    out = src.new_empty((1,))
    _KERNEL(src, float(pivot), out)
    return out.cpu()
