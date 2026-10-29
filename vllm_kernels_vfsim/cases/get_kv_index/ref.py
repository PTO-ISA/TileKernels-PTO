import torch

VL = 64


def reference_blocks(table: torch.Tensor, length: int) -> torch.Tensor:
    out = torch.empty(VL, dtype=torch.int32)
    for pos in range(VL):
        if pos < length:
            blk = int(table[pos // 16])
            out[pos] = blk * 16 + (pos & 15)
        else:
            out[pos] = -1
    return out


def reference_identity(offset: int, length: int) -> torch.Tensor:
    out = torch.empty(VL, dtype=torch.int32)
    for i in range(VL):
        gidx = offset + i
        out[i] = gidx if gidx < offset + length else -1
    return out


def reference_dflash(start: int, slot_origin: int, kv_len: int, swa: int = 8) -> torch.Tensor:
    out = torch.full((VL,), -1, dtype=torch.int32)
    for col in range(VL):
        pos = col + start
        if col < kv_len and pos >= 0:
            out[col] = (pos % swa) + slot_origin
    return out
