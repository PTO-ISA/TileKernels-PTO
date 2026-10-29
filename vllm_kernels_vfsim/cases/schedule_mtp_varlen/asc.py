"""ASC confidence-rate ranking VF.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/schedule_mtp_varlen.py#L85

The SimtVF that builds cumulative rates is out of scope. This driver consumes
the rates the VF actually reads.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

SEQS = 2
NEXTN = 4
M = SEQS * NEXTN
VL = 64


@tilelang.jit(target="ascend")
def compile_kernel():
    @T.prim_func
    def kernel(
        rates: T.Tensor((VL,), T.float32),
        seqlens: T.Tensor((SEQS,), T.int32),
        extra_budget: T.int32,
    ):
        with T.Kernel(1):
            rates_ub = T.alloc_shared((VL,), T.float32)
            out_ub = T.alloc_shared((VL,), T.int32)
            T.copy(rates, rates_ub)
            for seq_idx in T.serial(SEQS):
                with T.SimdVF():
                    full = S.pset(32, "PAT_ALL")
                    one_i = S.vdup(T.int32(1), "int32", full)
                    zero_i = S.vdup(T.int32(0), "int32", full)
                    my_rate = S.alloc_local((NEXTN,), "float32")
                    rank = S.alloc_local((NEXTN,), "int32")
                    num_tokens = S.alloc_local((1,), "int32")
                    for t in T.unroll(NEXTN, explicit=True):
                        my_rate[t] = S.vld(rates_ub[seq_idx * NEXTN + t], dist="BRC_B32")
                        rank[t] = zero_i
                    other = S.vld(rates_ub[0])
                    other_flat = S.vci(0, "int32")
                    for t in T.unroll(NEXTN, explicit=True):
                        beats = S.por(
                            S.vcmp(other, my_rate[t], full, "gt"),
                            S.pand(
                                S.vcmp(other, my_rate[t], full, "eq"),
                                S.vcmps(other_flat, seq_idx * NEXTN + t, full, "lt"),
                                full,
                            ),
                            full,
                        )
                        rank[t] = S.vadd(rank[t], S.vcadd(S.vsel(one_i, zero_i, beats), full), full)
                    num_tokens[0] = one_i
                    for t in T.unroll(NEXTN, explicit=True):
                        won = S.vcmps(rank[t], extra_budget, full, "lt")
                        num_tokens[0] = S.vadd(num_tokens[0], S.vsel(one_i, zero_i, won), full)
                    S.vsts(out_ub[seq_idx], num_tokens[0], full, "ONEPT_B32")
            T.copy(out_ub[0:SEQS], seqlens)

    return kernel


_KERNEL = None


def launch(rates, extra_budget):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = compile_kernel()
    rates_n = rates.npu()
    out_n = torch_empty_i32(rates_n)
    _KERNEL(rates_n, out_n, int(extra_budget))
    return out_n.cpu()


def torch_empty_i32(rates_n):
    import torch

    return torch.empty(SEQS, dtype=torch.int32, device=rates_n.device)
