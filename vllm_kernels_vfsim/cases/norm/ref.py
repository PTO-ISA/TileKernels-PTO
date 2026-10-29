import torch

from cases.norm.asc import EPS, HIDDEN
from cases.norm.quant_asc import CLAMP, GROUPS, QMAX


def reference_rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    var = (x.float() * x.float()).sum() / HIDDEN + EPS
    return x.float() * torch.rsqrt(var) * weight.float()


def reference_quant(x: torch.Tensor):
    grouped = x.float().reshape(GROUPS, HIDDEN // GROUPS)
    amax = grouped.abs().amax(dim=1).clamp(min=CLAMP)
    scale = amax / QMAX
    quant = (grouped / scale.unsqueeze(1)).to(torch.float8_e4m3fn)
    return quant.reshape(HIDDEN), scale


def reference_layout(table: torch.Tensor):
    values = table.view(torch.int16).to(torch.int32) & 0xFFFF
    lane = torch.arange(128, dtype=torch.int32)
    row = lane & 7
    pack = lane >> 3
    idx = row * 8 + pack
    return idx, values[idx]
