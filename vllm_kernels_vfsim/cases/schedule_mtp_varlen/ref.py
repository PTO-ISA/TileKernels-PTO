import torch

SEQS = 2
NEXTN = 4


def reference(rates: torch.Tensor, extra_budget: int) -> torch.Tensor:
    flat = rates[: SEQS * NEXTN].float()
    out = []
    m = flat.numel()
    for seq in range(SEQS):
        tokens = 1
        for t in range(NEXTN):
            mine_i = seq * NEXTN + t
            mine = flat[mine_i]
            rank = 0
            for j in range(m):
                other = flat[j]
                if other > mine or (other == mine and j < mine_i):
                    rank += 1
            if rank < extra_budget:
                tokens += 1
        out.append(tokens)
    return torch.tensor(out, dtype=torch.int32)
