"""PTO single-core driver for the token bitmask VF.

Source: https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/structure_output.py#L82

``BRC_B32`` is a load plus ``vbrc``. ``PAT_VL32`` is ``create_mask(32, size=64)``.
``vsel`` takes the mask first.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

VOCAB = 128
VL = 64
BITS = 32


@tilelang.jit(target="pto")
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
                full = V.create_mask(VL, size=VL)
                low = V.create_mask(32, size=VL)
                one = V.vbrc(T.int32(1), size=VL)
                bit_idx_mask = V.vbrc(T.int32(BITS - 1), size=VL)
                neg_inf = V.vbrc(T.float32(float("-inf")), size=VL)
                lane_idx = V.vci(T.int32(0), size=VL)
                bit_idx = V.vand(lane_idx, bit_idx_mask, full)
                bit_flags = V.vshl(one, bit_idx, full)
                for v in T.serial(VOCAB // VL):
                    word0 = V.vbrc(V.vload(bitmask_ub[v * 2], size=1), size=VL)
                    word1 = V.vbrc(V.vload(bitmask_ub[v * 2 + 1], size=1), size=VL)
                    words_v = V.vsel(low, word0, word1)
                    bits = V.vand(words_v, bit_flags, full)
                    allowed = V.vcmps(bits, T.int32(0), full, "ne")
                    values = V.vload(logits_ub[v * VL], size=VL)
                    V.vstore(V.vsel(allowed, values, neg_inf), logits_ub[v * VL], full)
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
