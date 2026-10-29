import torch


def _row(score: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    maximum = torch.amax(score, dim=0)
    weight = torch.exp(score - maximum)
    mixed = (latent * weight).sum(dim=0) / weight.sum(dim=0)
    return mixed.to(torch.bfloat16)


def reference(score: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    row = _row(score, latent)
    zero = torch.zeros_like(row)
    return torch.stack((row, zero, row))
