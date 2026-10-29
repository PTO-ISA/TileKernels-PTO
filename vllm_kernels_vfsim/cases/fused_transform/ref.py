import torch

from cases.fused_transform.asc import EPS, HIDDEN


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    var = (xf * xf).sum() / HIDDEN + EPS
    return (xf * torch.rsqrt(var) * weight.float()).to(torch.bfloat16)
