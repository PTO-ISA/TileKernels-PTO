"""ASC sum of exp(logprob) at and before the sampled token.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sum_smaller_probs.py#L58
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

VOCAB = 128
VL = 64
SAMPLE = 70
SPLIT = SAMPLE // VL
LANE = SAMPLE % VL


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        logprobs: T.Tensor((VOCAB,), T.float32),
        pivot: T.float32,
        out: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(1):
            logprobs_ub = T.alloc_shared((VOCAB,), T.float32)
            out_ub = T.alloc_shared((VL,), T.float32)
            T.copy(logprobs, logprobs_ub)
            with T.SimdVF():
                full = S.pset(32)
                one_lane = S.pset(32, "PAT_VL1")
                pivot_v = S.vdup(pivot, "float32", full)
                acc = S.alloc_var(T.float32)
                acc = S.vdup(0.0, "float32", full)
                for v in T.serial(SPLIT):
                    val = S.vld(logprobs_ub[v * VL])
                    prob = S.vexp(val, full)
                    pred = S.vcmp(val, pivot_v, full, "le")
                    acc = S.vadd(acc, prob, pred, "MODE_MERGING")
                for v in T.serial(SPLIT + 1, VOCAB // VL):
                    val = S.vld(logprobs_ub[v * VL])
                    prob = S.vexp(val, full)
                    pred = S.vcmp(val, pivot_v, full, "lt")
                    acc = S.vadd(acc, prob, pred, "MODE_MERGING")
                lane = S.vci(0, "int32", "INC_ORDER")
                val = S.vld(logprobs_ub[SPLIT * VL])
                prob = S.vexp(val, full)
                less_pred = S.vcmp(val, pivot_v, full, "lt")
                before_pred = S.vcmps(lane, LANE, full, "le")
                equal_pred = S.vcmp(val, pivot_v, before_pred, "eq")
                pred = S.por(less_pred, equal_pred, full)
                acc = S.vadd(acc, prob, pred, "MODE_MERGING")
                result = S.vcadd(acc, full)
                S.vsts(out_ub[0], result, one_lane, "ONEPT_B32")
            T.copy(out_ub[0:1], out)

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
