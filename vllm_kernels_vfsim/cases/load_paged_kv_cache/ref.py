import torch


def reference(raw: torch.Tensor, packed_scale: int) -> torch.Tensor:
    fp8 = raw.view(torch.float8_e4m3fn).float()
    b0 = packed_scale & 0xFF
    b1 = (packed_scale >> 8) & 0xFF
    scale = torch.empty_like(fp8)
    scale[:32] = torch.ldexp(torch.ones(32), torch.tensor(b0 - 127))
    scale[32:] = torch.ldexp(torch.ones(32), torch.tensor(b1 - 127))
    dequant = (fp8 * scale).to(torch.bfloat16)
    zero = torch.zeros_like(dequant)
    return torch.stack((dequant, zero))
