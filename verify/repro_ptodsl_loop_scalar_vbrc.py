"""Minimal repro: PTODSL printer drops int32 typing of loop-carried vbrc scalars.

A TIR for-loop (from a plain ``range()``) whose body broadcasts a
loop-dependent ``T.int32`` scalar via ``V.vbrc`` compiles the vbrc operand
back into bare Python arithmetic in kernel.ptodsl.py, which the PTODSL
frontend types as ``index``; ptoas then aborts with

    error: '!pto.vmi.vreg<64xindex>' expected an 8-bit, 16-bit, or 32-bit logical element type
    python: .../StorageUniquerSupport.h:179: Assertion `succeeded(...)' failed.

The TIR-level value IS int32-typed (``T.int32(vec_id * 4)``); only the
PTODSL printing loses it.  Expected behavior: the printer emits
``pto.si32(...)`` (or an explicit cast) for int-typed loop expressions.

Run (238, tkvmi env):  python verify/repro_ptodsl_loop_scalar_vbrc.py
Fix reference: unrolling the loop (T.unroll) sidesteps it because the
scalar folds to a constant (what pre_big_fuse_vmi.py does), but any
non-unrollable loop count would still hit it.
"""

import sys

sys.path.insert(0, "/data/sunkexin/tkvmi/TileKernels-PTO")

import torch

import torch_npu  # noqa: F401

torch.npu.set_device(0)

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target="pto")
def repro(a):
    n = T.dynamic("n")
    a: T.Tensor[(n,), T.float32]
    with T.Kernel(1):
        ub = T.alloc_shared((64,), T.float32)
        with T.SimdVF():
            mask_full = V.create_mask(64, size=64)
            idx = V.vci(T.int32(0), size=64)
            idx = V.vci(T.int32(0), size=64)
            # plain range() becomes a TIR loop; the vbrc scalar below is a
            # loop-dependent int32 expression (int arithmetic, as in the real
            # kernel's token-index computation), consumed inside the loop
            for vec_id in range(4):
                token = V.vadd(idx, V.vbrc(T.int32(vec_id * 4), size=64), mask_full)
                V.vstore(V.vcvt(token, 'float32'), ub[0])
        T.copy(ub[0:64], a[0:64])


if __name__ == "__main__":
    out = torch.zeros(64, dtype=torch.float32, device="npu:0")
    repro(out)
    torch.npu.synchronize()
    print("compiled and ran:", out[:4].tolist())
