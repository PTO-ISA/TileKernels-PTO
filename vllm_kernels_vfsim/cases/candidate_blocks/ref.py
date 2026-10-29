import torch

from cases.candidate_blocks.asc import BLOCKS, LANES


def _scan(values: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for step in (1, 2, 4, 8, 16, 32):
        nxt = out.clone()
        nxt[step:] = out[step:] + out[:-step]
        out = nxt
    return out


def reference(lengths, indices, candidates, row_len):
    counts = torch.div(lengths + 16383, 16384, rounding_mode="floor").to(torch.int32)
    total = counts.sum().reshape(1).to(torch.int32)
    prefix = _scan(counts)
    restored = torch.empty_like(indices)
    for i, index in enumerate(indices.tolist()):
        if index < 0:
            restored[i] = -1
            continue
        slot = index >> 4
        if slot >= BLOCKS:
            restored[i] = -1
            continue
        block_id = int(candidates[slot])
        if block_id < 0:
            restored[i] = -1
            continue
        restored[i] = (block_id << 4) + (index & 15)
    compact = 0
    for slot in range(BLOCKS):
        block_id = int(candidates[slot])
        start = block_id * 16
        remain = row_len - start
        if remain < 0:
            remain = 0
        if remain > 16:
            remain = 16
        compact += remain
    return counts, prefix, restored, total, torch.tensor([compact], dtype=torch.int32)
