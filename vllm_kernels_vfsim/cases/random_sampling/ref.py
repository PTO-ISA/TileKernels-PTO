import torch

from cases.random_sampling.asc import SCALE, SLOTS, VOCAB


def _fold(probs: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(SLOTS, dtype=torch.int32)
    scaled = torch.floor(probs.float() * SCALE).to(torch.int32)
    for t in range(VOCAB // SLOTS):
        acc = acc + scaled[t * SLOTS : (t + 1) * SLOTS]
    return acc


def _scan(values: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for step in (1, 2, 4, 8, 16, 32):
        nxt = out.clone()
        nxt[step:] = out[step:] + out[:-step]
        out = nxt
    return out


def reference(probs: torch.Tensor) -> torch.Tensor:
    folded = _fold(probs)
    once = _scan(folded)
    twice = _scan(once)
    return torch.stack((once, twice))
