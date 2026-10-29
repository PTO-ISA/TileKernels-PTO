import torch


def reference(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    rows = []
    n = x.shape[0]
    for idx in index.tolist():
        if 0 <= int(idx) < n:
            rows.append(x[int(idx)])
        else:
            rows.append(torch.zeros(x.shape[1], dtype=x.dtype))
    return torch.stack(rows)
