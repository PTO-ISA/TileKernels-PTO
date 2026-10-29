"""ASC single-core driver for the token bitmask VF.

Source: https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/structure_output.py#L82
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

VOCAB = 128
VL = 64
BITS = 32


@tilelang.jit(target="ascend")
def compile_kernel():
    words = VOCAB // BITS

    @T.prim_func
    def kernel(
        logits: T.Tensor((VOCAB,), T.float32),
        bitmask: T.Tensor((words,), T.int32),
        out: T.Tensor((VOCAB,), T.float32),
    ):
        with T.Kernel(1):
            logits_ub = T.alloc_shared((VOCAB,), T.float32)
            bitmask_ub = T.alloc_shared((words,), T.int32)
            T.copy(logits, logits_ub)
            T.copy(bitmask, bitmask_ub)
            with T.SimdVF():
                full = S.pset(32)
                one = S.vdup(1, "int32", full)
                bit_idx_mask = S.vdup(BITS - 1, "int32", full)
                neg_inf = S.vdup(float("-inf"), "float32", full)
                lane_idx = S.vci(0, "int32", "INC_ORDER")
                bit_idx = S.vand(lane_idx, bit_idx_mask, full)
                bit_flags = S.vshl(one, bit_idx, full)
                low_word = S.pset(32, "PAT_VL32")
                for v in T.serial(VOCAB // VL):
                    word0 = S.vld(bitmask_ub[v * 2], dist="BRC_B32")
                    word1 = S.vld(bitmask_ub[v * 2 + 1], dist="BRC_B32")
                    words_v = S.vsel(word0, word1, low_word)
                    allowed = S.vcmps(S.vand(words_v, bit_flags, full), 0, full, "ne")
                    values = S.vld(logits_ub[v * VL])
                    result = S.vsel(values, neg_inf, allowed)
                    S.vsts(logits_ub[v * VL], result, full)
            T.copy(logits_ub, out)

    return kernel


_KERNEL = None


def launch(logits, bitmask):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    logits_n = logits.npu()
    out_n = logits_n.new_empty(logits_n.shape)
    _KERNEL(logits_n, bitmask.npu(), out_n)
    return out_n.cpu()
