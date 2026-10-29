import torch


def reference(pivots: torch.Tensor, width: int = 128) -> torch.Tensor:
    rows = pivots.shape[0]
    out = torch.full((rows, width), -1, dtype=torch.int32)
    for r in range(rows):
        offset1, len1, offset2, len2 = (int(v) for v in pivots[r])
        end2 = len1 + len2
        base2 = offset2 - len1
        for col in range(width):
            if col < len1:
                out[r, col] = offset1 + col
            elif col < end2:
                out[r, col] = base2 + col
    return out
