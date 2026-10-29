"""ASC get_kv_index VFs.

https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L52
https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L214

``load`` is the block-table branch. ``identity`` is the block_size==0 branch of
the same VF. ``dflash`` is the second SimdVF (SWA modulo). The surrounding
SimtVF is not ported.
"""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

VL = 64
BLOCKS = 4


@tilelang.jit(target="ascend")
def compile_blocks():
    block_bits = 4

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
                mask = S.pset(32)
                neg1 = S.vdup(-1, "int32", mask)
                mask_const = S.vdup(15, "int32", mask)
                pos = S.vci(0, "int32")
                q = S.vshrs(pos, block_bits, mask)
                rem = S.vand(pos, mask_const, mask)
                valid_q = S.vcmps(q, BLOCKS, mask, "lt")
                blk = S.vgather2(blk_ub[0], T.reinterpret("uint32x64", q), valid_q)
                val = S.vadd(S.vshls(blk, block_bits, mask), rem, mask)
                res = S.vsel(val, neg1, S.vcmps(pos, length, mask, "lt"))
                S.vsts(index[0], res, mask)
            T.copy(index, out)

    return kernel


@tilelang.jit(target="ascend")
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
                mask = S.pset(32)
                neg1 = S.vdup(-1, "int32", mask)
                gidx = S.vadds(S.vci(0, "int32"), offset, mask)
                limit = S.vadds(S.vdup(offset, "int32", mask), length, mask)
                res = S.vsel(gidx, neg1, S.vcmp(gidx, limit, mask, "lt"))
                S.vsts(index[0], res, mask)
            T.copy(index, out)

    return kernel


@tilelang.jit(target="ascend")
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
                mask = S.pset(32)
                neg1 = S.vdup(-1, "int32", mask)
                swa_vec = S.vdup(swa, "int32", mask)
                so_vec = S.vdup(slot_origin, "int32", mask)
                col = S.vci(0, "int32")
                pos = S.vadds(col, start, mask)
                pos_q = S.vdiv(pos, swa_vec, mask)
                rem_val = S.vsub(pos, S.vmul(pos_q, swa_vec, mask), mask)
                val = S.vadd(rem_val, so_vec, mask)
                keep = S.pand(S.vcmps(col, kv_len, mask, "lt"), S.vcmps(pos, 0, mask, "ge"), mask)
                S.vsts(kv_ub[0], S.vsel(val, neg1, keep), mask)
            T.copy(kv_ub, out)

    return kernel


_BLOCKS = None
_IDENTITY = None
_DFLASH = None


def _out(like_device_tensor):
    import torch

    return torch.empty(VL, dtype=torch.int32, device=like_device_tensor.device)


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
