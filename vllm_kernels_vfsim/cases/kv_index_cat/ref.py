"""Torch-CPU reference for kv_index_cat: add the offset, keep -1."""

import torch


def reference(src: torch.Tensor, offset: int) -> torch.Tensor:
    out = src + int(offset)
    return torch.where(src == -1, src, out)
