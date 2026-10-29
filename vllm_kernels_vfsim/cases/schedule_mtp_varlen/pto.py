"""PTO confidence-rate ranking VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/schedule_mtp_varlen.py#L85

The rank is a scalar. PTO keeps it in a 1-lane vector instead of adding a
reduction into every lane and reading lane 0 back.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

SEQS = 2
NEXTN = 4
VL = 64


@tilelang.jit(target="pto")
def compile_kernel():
    @T.prim_func
    def kernel(
        rates: T.Tensor((VL,), T.float32),
        seqlens: T.Tensor((SEQS,), T.int32),
        extra_budget: T.int32,
    ):
        with T.Kernel(1):
            rates_ub = T.alloc_shared((VL,), T.float32)
            out_ub = T.alloc_shared((SEQS,), T.int32)
            T.copy(rates, rates_ub)
            for seq_idx in T.serial(SEQS):
                with T.SimdVF():
                    full = V.create_mask(VL, size=VL)
                    one1 = V.create_mask(1, size=1)
                    one_i = V.vbrc(T.int32(1), size=VL)
                    zero_i = V.vbrc(T.int32(0), size=VL)
                    one_s = V.vbrc(T.int32(1), size=1)
                    zero_s = V.vbrc(T.int32(0), size=1)
                    other = V.vload(rates_ub[0], size=VL)
                    other_flat = V.vci(T.int32(0), size=VL)
                    tokens = V.alloc_var(T.int32, size=1)
                    tokens = one_s
                    for t in T.unroll(NEXTN, explicit=True):
                        my = V.vbrc(V.vload(rates_ub[seq_idx * NEXTN + t], size=1), size=VL)
                        greater = V.vcmp(other, my, full, "gt")
                        equal = V.vcmp(other, my, full, "eq")
                        earlier = V.vcmps(other_flat, seq_idx * NEXTN + t, full, "lt")
                        beats = V.vor(greater, V.vand(equal, earlier))
                        rank = V.vcadd(V.vsel(beats, one_i, zero_i), full)
                        won = V.vcmps(rank, extra_budget, one1, "lt")
                        tokens = V.vadd(tokens, V.vsel(won, one_s, zero_s), one1)
                    V.vstore(tokens, out_ub[seq_idx])
            T.copy(out_ub, seqlens)

    return kernel


_KERNEL = None


def launch(rates, extra_budget):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    import torch

    rates_n = rates.npu()
    out_n = torch.empty(SEQS, dtype=torch.int32, device=rates_n.device)
    _KERNEL(rates_n, out_n, int(extra_budget))
    return out_n.cpu()
