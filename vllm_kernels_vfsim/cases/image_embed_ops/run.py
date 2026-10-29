"""image_embed_ops. python run.py asc|pto copy|pool|scatter"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.driver import finish  # noqa: E402
from harness.sim import maybe_reexec  # noqa: E402

maybe_reexec()

import torch  # noqa: E402

from cases.image_embed_ops import asc, pto  # noqa: E402
from cases.image_embed_ops.ref import reference_pool, reference_scatter  # noqa: E402

CASE = "image_embed_ops"


def main() -> None:
    impl, variant = sys.argv[1], sys.argv[2]
    mod = asc if impl == "asc" else pto
    g = torch.Generator().manual_seed(6)
    if variant == "copy":
        src = torch.randn(128, generator=g, dtype=torch.bfloat16)
        got = mod.launch_copy(src)
        finish(CASE, impl, got.view(torch.int16), src.view(torch.int16), variant=variant)
    elif variant == "pool":
        window = torch.randn(4, 64, generator=g)
        inv = 1.0 / 4
        got = mod.launch_pool(window, inv)
        finish(CASE, impl, got, reference_pool(window, inv), exact=False, rtol=1e-5, atol=1e-5, variant=variant)
    else:
        window = torch.randn(4, 64, generator=g)
        got = mod.launch_scatter(window)
        finish(CASE, impl, got, reference_scatter(window), variant=variant)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[status] variant={CASE} result=FAIL")
        print(f"FAIL {type(exc).__name__}: {exc}")
        raise
