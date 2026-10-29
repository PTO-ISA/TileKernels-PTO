"""Numeric checks. Integer cases are exact. Float cases use a stated tolerance."""

from __future__ import annotations

import torch


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous()


def check_exact(got: torch.Tensor, ref: torch.Tensor, name: str) -> None:
    got_c, ref_c = _cpu(got), _cpu(ref)
    if got_c.shape != ref_c.shape or got_c.dtype != ref_c.dtype:
        raise AssertionError(f"{name}: shape/dtype {tuple(got_c.shape)} {got_c.dtype} vs {tuple(ref_c.shape)} {ref_c.dtype}")
    if torch.equal(got_c, ref_c):
        return
    mismatch = (got_c != ref_c).nonzero(as_tuple=False)
    n = mismatch.shape[0]
    sample = mismatch[:8].tolist()
    raise AssertionError(f"{name}: {n} mismatches, first idx {sample}\n got {got_c.view(-1)[:8].tolist()}\n ref {ref_c.view(-1)[:8].tolist()}")


def check_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    name: str,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    got_c, ref_c = _cpu(got).float(), _cpu(ref).float()
    if got_c.shape != ref_c.shape:
        raise AssertionError(f"{name}: shape {tuple(got_c.shape)} vs {tuple(ref_c.shape)}")
    if torch.allclose(got_c, ref_c, rtol=rtol, atol=atol, equal_nan=True):
        return
    diff = (got_c - ref_c).abs()
    # NaN-aware: treat matching non-finite pairs as zero error.
    both_nan = torch.isnan(got_c) & torch.isnan(ref_c)
    both_inf = torch.isinf(got_c) & torch.isinf(ref_c) & (torch.sign(got_c) == torch.sign(ref_c))
    diff = torch.where(both_nan | both_inf, torch.zeros_like(diff), diff)
    worst = int(diff.view(-1).argmax())
    raise AssertionError(
        f"{name}: not close rtol={rtol} atol={atol} max_abs={float(diff.max())} "
        f"at {worst} got={float(got_c.view(-1)[worst])} ref={float(ref_c.view(-1)[worst])}"
    )
