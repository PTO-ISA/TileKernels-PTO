"""Torch-CPU reference for the token bitmask VF."""

import torch

VOCAB = 128
BITS = 32


def reference(logits: torch.Tensor, bitmask: torch.Tensor) -> torch.Tensor:
    out = logits.clone()
    words = bitmask.to(torch.int64) & 0xFFFFFFFF
    for i in range(VOCAB):
        bit = (int(words[i // BITS]) >> (i % BITS)) & 1
        if bit == 0:
            out[i] = float("-inf")
    return out
