import torch

from cases.image_embed_ops.asc import D, D2, DIM


def reference_pool(window: torch.Tensor, inv: float) -> torch.Tensor:
    return window.sum(dim=0) * inv


def reference_scatter(window: torch.Tensor) -> torch.Tensor:
    out = torch.empty((DIM, D, D), dtype=torch.float32)
    for i in range(D2):
        out[:, i // D, i % D] = window[i]
    return out
