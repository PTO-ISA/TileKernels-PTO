"""PTO get_kv_index VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L52
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L214
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

VL = 64
BLOCKS = 4


@tilelang.jit(target="pto")
def compile_blocks():
    @T.prim_func
    def kernel(
        table: T.Tensor((BLOCKS,), T.int32),
        out: T.Tensor((VL,), T.int32),
        length: T.int32,
    ):
        with T.Kernel(1):
            blk_ub = T.alloc_shared((BLOCKS,), T.int32)
            index = T.alloc_shared((VL,), T.int32)
            T.copy(table, blk_ub)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                neg1 = V.vbrc(T.int32(-1), size=VL)
                mask_const = V.vbrc(T.int32(15), size=VL)
                pos = V.vci(T.int32(0), size=VL)
                q = V.vshrs(pos, 4, mask)
                rem = V.vand(pos, mask_const, mask)
                valid_q = V.vcmps(q, T.int32(BLOCKS), mask, "lt")
                blk = V.vgather(blk_ub[0], V.vinterpret_cast(q, "uint32"), valid_q)
                val = V.vadd(V.vshls(blk, 4, mask), rem, mask)
                inside = V.vcmps(pos, length, mask, "lt")
                V.vstore(V.vsel(inside, val, neg1), index[0], mask)
            T.copy(index, out)

    return kernel


@tilelang.jit(target="pto")
def compile_identity():
    @T.prim_func
    def kernel(
        out: T.Tensor((VL,), T.int32),
        offset: T.int32,
        length: T.int32,
    ):
        with T.Kernel(1):
            index = T.alloc_shared((VL,), T.int32)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                neg1 = V.vbrc(T.int32(-1), size=VL)
                gidx = V.vadds(V.vci(T.int32(0), size=VL), offset, mask)
                limit = V.vbrc(offset + length, size=VL)
                inside = V.vcmp(gidx, limit, mask, "lt")
                V.vstore(V.vsel(inside, gidx, neg1), index[0], mask)
            T.copy(index, out)

    return kernel


@tilelang.jit(target="pto")
def compile_dflash():
    swa = 8

    @T.prim_func
    def kernel(
        out: T.Tensor((VL,), T.int32),
        start: T.int32,
        slot_origin: T.int32,
        kv_len: T.int32,
    ):
        with T.Kernel(1):
            kv_ub = T.alloc_shared((VL,), T.int32)
            with T.SimdVF():
                mask = V.create_mask(VL, size=VL)
                neg1 = V.vbrc(T.int32(-1), size=VL)
                so_vec = V.vbrc(slot_origin, size=VL)
                col = V.vci(T.int32(0), size=VL)
                pos = V.vadds(col, start, mask)
                # swa_block_size is 8 here, so the remainder is the low bits.
                # VMI vdiv is floating-point only; ASC keeps the integer divide.
                rem_val = V.vand(pos, V.vbrc(T.int32(7), size=VL), mask)
                val = V.vadd(rem_val, so_vec, mask)
                keep = V.vand(
                    V.vcmps(col, kv_len, mask, "lt"),
                    V.vcmps(pos, T.int32(0), mask, "ge"),
                )
                V.vstore(V.vsel(keep, val, neg1), kv_ub[0], mask)
            T.copy(kv_ub, out)

    return kernel


_BLOCKS = None
_IDENTITY = None
_DFLASH = None


def _out(device_tensor):
    import torch

    return torch.empty(VL, dtype=torch.int32, device=device_tensor.device)


def launch_blocks(table, length):
    global _BLOCKS
    if _BLOCKS is None:
        _BLOCKS = compile_blocks()
    table_n = table.npu()
    out = _out(table_n)
    _BLOCKS(table_n, out, int(length))
    return out.cpu()


def launch_identity(offset, length):
    global _IDENTITY
    if _IDENTITY is None:
        _IDENTITY = compile_identity()
    import torch

    dummy = torch.empty(1, dtype=torch.int32).npu()
    out = _out(dummy)
    _IDENTITY(out, int(offset), int(length))
    return out.cpu()


def launch_dflash(start, slot_origin, kv_len):
    global _DFLASH
    if _DFLASH is None:
        _DFLASH = compile_dflash()
    import torch

    dummy = torch.empty(1, dtype=torch.int32).npu()
    out = _out(dummy)
    _DFLASH(out, int(start), int(slot_origin), int(kv_len))
    return out.cpu()
