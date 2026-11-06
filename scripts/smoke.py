"""On-device smoke: one DMA kernel, one SimtVF kernel, one SimdVF kernel.

ASC and PTO see the same inputs. Integers must match exactly.
"""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401

from vllm_kernels.asc.compute_seqlens_k_zos import compute_seqlens_k_zos_tl as asc_seqlens
from vllm_kernels.asc.store_mtp_info import store_mtp_info_tl as asc_mtp
from vllm_kernels.asc.swa_index import expand_swa_index_tl as asc_swa
from vllm_kernels.pto.compute_seqlens_k_zos import compute_seqlens_k_zos_tl as pto_seqlens
from vllm_kernels.pto.store_mtp_info import store_mtp_info_tl as pto_mtp
from vllm_kernels.pto.swa_index import expand_swa_index_tl as pto_swa


def _device() -> int:
    dev = int(os.environ.get("BENCH_DEVICES", "1"))
    torch.npu.set_device(dev)
    return dev


def _check(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
        if a.shape == b.shape and a.dtype == b.dtype:
            bad = (a.cpu() != b.cpu()).flatten().nonzero().flatten()[:8].tolist()
            mismatch = int((a.cpu() != b.cpu()).sum())
        else:
            bad, mismatch = [], "shape/dtype"
        raise SystemExit(f"FAIL {name}: mismatch={mismatch} idx={bad} asc={tuple(a.shape)} pto={tuple(b.shape)}")
    print(f"PASS {name}", flush=True)


def smoke_seqlens() -> None:
    nextn, n_seq = 3, 512
    n_prev = n_seq + 16
    seqlens_k = torch.randint(256, 356, (n_seq,), dtype=torch.int32, device="npu")
    seqlens_q = torch.randint(1, nextn + 1, (n_seq,), dtype=torch.int32, device="npu")
    remap = torch.full((n_seq,), -1, dtype=torch.int32, device="npu")
    remap[: n_seq // 2] = torch.randint(0, n_prev, (n_seq // 2,), dtype=torch.int32, device="npu")
    prev = torch.randint(1, nextn + 1, (n_prev,), dtype=torch.int32, device="npu")
    asc_out = seqlens_k.clone()
    pto_out = seqlens_k.clone()
    asc_seqlens(nextn)(asc_out, seqlens_q, remap, prev)
    pto_seqlens(nextn)(pto_out, seqlens_q, remap, prev)
    torch.npu.synchronize()
    _check("compute_seqlens_k_zos", asc_out, pto_out)


def smoke_mtp() -> None:
    req_cnt, nextn, vocab, num_slots = 32, 4, 1024, 48
    write_slots = torch.arange(1, req_cnt + 1, dtype=torch.int32, device="npu")
    write_slots[::8] = 0
    token_ids = torch.randint(0, 1000, (req_cnt, nextn), dtype=torch.int32, device="npu")
    logprobs = torch.randn(req_cnt, nextn, vocab, dtype=torch.float32, device="npu")
    index = torch.randint(0, 1000, (req_cnt, nextn, vocab), dtype=torch.int32, device="npu")
    confidence = torch.rand(req_cnt, nextn, dtype=torch.float32, device="npu")
    pool_rows = (num_slots + 1) * nextn
    pools = []
    for _ in range(2):
        pools.append(
            (
                torch.zeros(pool_rows, dtype=torch.int32, device="npu"),
                torch.zeros(pool_rows, vocab, dtype=torch.float32, device="npu"),
                torch.zeros(pool_rows, vocab, dtype=torch.int32, device="npu"),
                torch.zeros(pool_rows, dtype=torch.float32, device="npu"),
            )
        )
    asc_mtp(nextn, vocab, True, True)(write_slots, token_ids, logprobs, index, confidence, *pools[0])
    pto_mtp(nextn, vocab, True, True)(write_slots, token_ids, logprobs, index, confidence, *pools[1])
    torch.npu.synchronize()
    for name, a, b in zip(("tokens", "logprobs", "index", "confidence"), pools[0], pools[1], strict=True):
        _check(f"store_mtp_info/{name}", a, b)


def smoke_swa() -> None:
    n, width = 128, 512
    pivots = torch.randint(0, 64, (n, 4), dtype=torch.int32, device="npu")
    pivots[:, 1].clamp_(max=width)
    pivots[:, 3].clamp_(max=width)
    asc_out = torch.empty(n, width, dtype=torch.int32, device="npu")
    pto_out = torch.empty(n, width, dtype=torch.int32, device="npu")
    asc_swa(width)(pivots, asc_out)
    pto_swa(width)(pivots, pto_out)
    torch.npu.synchronize()
    _check("expand_swa_index", asc_out, pto_out)


def main() -> None:
    dev = _device()
    print(f"device={dev} name={torch.npu.get_device_name(dev)}", flush=True)
    smoke_seqlens()
    smoke_mtp()
    smoke_swa()
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
