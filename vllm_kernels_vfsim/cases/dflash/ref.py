import torch


def reference_dot(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    acc = (x.float() * weight.float()).sum(dim=-1)
    return acc


def reference_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean(dim=0).to(torch.bfloat16)
