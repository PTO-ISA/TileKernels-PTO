"""Correctness and on-device ASC/PTO benchmarks for paged KV load/store."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "PTO_FLAGS",
    "--pto-backend=vpto --pto-level=level3 "
    "--vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 "
    "--enable-vecscope-mem-bar=false",
)
os.environ.setdefault("TILELANG_CACHE_DIR", str(ROOT / ".tilelang_cache_kv"))

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401
from tilelang.profiler import do_bench  # noqa: E402

from deepinfer_ops.kv_loadstore.common import (  # noqa: E402
    BF16_KV_FORMAT,
    FP4_INDEX_KV_FORMAT,
    MXFP8_INDEX_KV_FORMAT,
    V41_FP4_FORMAT,
    V41_FORMAT,
    KVCacheFormat,
)
from deepinfer_ops.kv_loadstore.load_kv_asc import load_paged_kv_cache_tl as load_asc  # noqa: E402
from deepinfer_ops.kv_loadstore.load_kv_pto import load_paged_kv_cache_tl as load_pto  # noqa: E402
from deepinfer_ops.kv_loadstore.store_kv_asc import store_kv_cache_tl as store_asc  # noqa: E402
from deepinfer_ops.kv_loadstore.store_kv_pto import store_kv_cache_tl as store_pto  # noqa: E402

FORMATS = {
    "bf16": BF16_KV_FORMAT,
    "fp8": V41_FORMAT,
    "fp4": V41_FP4_FORMAT,
    "fp4_index": FP4_INDEX_KV_FORMAT,
    "mxfp8_index": MXFP8_INDEX_KV_FORMAT,
}
RESULTS = Path(__file__).resolve().parent / "results" / "kv_loadstore.jsonl"


@dataclass(frozen=True)
class Case:
    op: str
    format: str
    block_size: int
    tokens: int
    pattern: str

    @property
    def name(self) -> str:
        return f"{self.op}/{self.format}_b{self.block_size}_n{self.tokens}_{self.pattern}"

    @property
    def layout(self) -> KVCacheFormat:
        return FORMATS[self.format]


@dataclass
class Prepared:
    asc_fn: Callable[[], None]
    pto_fn: Callable[[], None]
    compare: Callable[[], None]
    io_bytes: int
    asc_kernel: Any = None
    pto_kernel: Any = None
    asc_matched_fn: Callable[[], None] | None = None


def _device() -> torch.device:
    dev = int(os.environ.get("BENCH_DEVICES", "1"))
    torch.npu.set_device(dev)
    return torch.device("npu", dev)


def _launch(kernel: Any, args: tuple) -> Callable[[], None]:
    return lambda: kernel(*args)


def _paged_storage(layout: KVCacheFormat, num_blocks: int, block_size: int, device: torch.device):
    alignment = math.lcm(32, layout.bytes_per_token) if layout.token_interleaved else 32
    block_bytes = block_size * layout.bytes_per_token
    block_stride = (block_bytes + alignment - 1) // alignment * alignment
    storage = torch.empty((num_blocks, block_stride), dtype=torch.uint8, device=device)
    return storage, block_stride


def _cache_view(storage: torch.Tensor, block_stride: int, block_size: int, layout: KVCacheFormat):
    return torch.as_strided(
        storage,
        (storage.shape[0], block_size, layout.bytes_per_token),
        (block_stride, layout.bytes_per_token, 1),
    )


def _guarded_store_cache(layout, num_blocks, block_size, pattern, device):
    if layout.quantized_dtype is None:
        block_stride = (block_size + 1) * layout.num_elements
        backing = torch.full(
            (32 + num_blocks * block_stride + 32,), 1.25, dtype=torch.bfloat16, device=device
        )
        cache = torch.as_strided(
            backing,
            (num_blocks, block_size, layout.num_elements),
            (block_stride, layout.num_elements, 1),
            storage_offset=32,
        )
        return backing, cache

    alignment = 512 if layout.do_cast else 32
    token_stride = layout.bytes_per_token
    if pattern == "broken_run" and layout.token_interleaved and token_stride % 32 == 0:
        token_stride += 16
    page_bytes = block_size * token_stride
    page_stride = (page_bytes + alignment - 1) // alignment * alignment + 3 * alignment
    storage_offset = 2 * alignment
    backing = torch.full(
        (storage_offset + num_blocks * page_stride + 2 * alignment,),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        backing,
        (num_blocks, block_size, layout.bytes_per_token),
        (page_stride, token_stride, 1),
        storage_offset=storage_offset,
    )
    return backing, cache


def _load_views(layout: KVCacheFormat, storage: torch.Tensor, block_stride: int, block_size: int):
    num_blocks = storage.shape[0]
    data_bytes = layout.quantized_bytes
    if layout.token_interleaved:
        records = _cache_view(storage, block_stride, block_size, layout)
        data = records[..., :data_bytes]
        sf = records[..., data_bytes:]
    else:
        data = torch.as_strided(storage, (num_blocks, block_size, data_bytes), (block_stride, data_bytes, 1))
        sf = torch.as_strided(
            storage.view(torch.int32),
            (num_blocks, block_size, 1),
            (block_stride // 4, 1, 1),
            storage_offset=(block_size * data_bytes) // 4,
        )
    value_dtype = torch.int8 if layout.quantized_dtype == "e2m1" else torch.float8_e4m3fn
    data = data.view(value_dtype)
    if layout.sf_format == "e4m3":
        sf = sf.view(torch.float8_e4m3fn)
    return data, sf


def _slots(n: int, block_size: int, pattern: str, device: torch.device) -> torch.Tensor:
    slots = torch.arange(n, dtype=torch.int32, device=device)
    if pattern in ("random", "random_invalid"):
        slots = slots[torch.randperm(n, device=device)]
    elif pattern == "cross_page":
        slots += block_size - 1
    elif pattern == "broken_run":
        slots[1:3] = slots[1:3].flip(0)
        middle = (n // 2 // 8) * 8
        slots[middle + 1 : middle + 3] = slots[middle + 1 : middle + 3].flip(0)
    elif pattern == "all_invalid":
        slots.fill_(-1)
    elif pattern == "invalid_tiles":
        slots[n // 4 : n // 2] = -1
    elif pattern == "leading_invalid":
        slots -= 1
    elif pattern != "sequential":
        raise ValueError(pattern)
    if pattern in ("random_invalid", "leading_invalid") and n > 1:
        slots[::7] = -1
        slots[n // 2] = -1
        slots[-1] = -1
    return slots


def _aligned_segments(tokens: int, block_size: int, device: torch.device) -> torch.Tensor:
    blocks = (tokens + block_size - 1) // block_size
    src = torch.arange(blocks, dtype=torch.int32, device=device)
    length = (tokens - src * block_size).clamp(max=block_size)
    return torch.stack((src, torch.zeros_like(src), length, src * block_size), dim=1)


def _short_segments(tokens: int, block_size: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260902 + block_size + tokens)
    ranges = []
    remaining = tokens
    while remaining:
        block = int(torch.randint(0, num_blocks, (), generator=generator))
        if torch.rand((), generator=generator) < 0.1:
            block = -1
        length = min(remaining, block_size if block == -1 else int(torch.randint(1, block_size + 1, (), generator=generator)))
        begin = int(torch.randint(0, block_size - length + 1, (), generator=generator))
        ranges.append((block, begin, begin + length))
        remaining -= length
    offset = 0
    rows = []
    for block, begin, end in ranges:
        rows.append((block, begin, end, offset))
        offset += end - begin
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _merged_segments(tokens: int, block_size: int, num_blocks: int, device: torch.device) -> torch.Tensor:
    ranges = [(block, 0, block_size) for block in (11, -1, 3, 7)]
    ranges.append((4, block_size - 1, block_size))
    remaining = tokens - 4 * block_size - 1
    while remaining:
        length = min(block_size, remaining)
        block = len(ranges) * 17 % num_blocks if len(ranges) % 11 else -1
        ranges.append((block, block_size - length, block_size))
        remaining -= length
    rows = []
    offset = 0
    for block, begin, end in ranges:
        rows.append((block, begin, end, offset))
        offset += end - begin
    rows[-4:] = reversed(rows[-4:])
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _load_segments(case: Case, num_blocks: int, device: torch.device) -> torch.Tensor:
    if case.pattern == "short":
        return _short_segments(case.tokens, case.block_size, num_blocks, device)
    if case.pattern == "merged":
        return _merged_segments(case.tokens, case.block_size, num_blocks, device)
    segments = _aligned_segments(case.tokens, case.block_size, device)
    if case.pattern == "unaligned":
        segments[:, 3] += 1
        segments[-1, 2] -= 1
        prefix = torch.tensor(
            [[num_blocks - 1, case.block_size - 1, case.block_size, 0]], dtype=torch.int32, device=device
        )
        segments = torch.cat((prefix, segments))
    elif case.pattern != "aligned":
        raise ValueError(case.pattern)
    return segments


def _compare_exact(label: str, asc: torch.Tensor, pto: torch.Tensor) -> None:
    a, b = asc.detach().cpu(), pto.detach().cpu()
    if a.dtype == torch.float8_e4m3fn:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    if not torch.equal(a, b):
        mismatch = int((a != b).sum())
        raise RuntimeError(f"{label}: {mismatch} mismatches")


def _compare_dequant(label: str, asc: torch.Tensor, pto: torch.Tensor) -> None:
    a, b = asc.detach().float().cpu(), pto.detach().float().cpu()
    if not torch.allclose(a, b, rtol=2e-2, atol=2e-2, equal_nan=True):
        diff = (a - b).abs()
        raise RuntimeError(f"{label}: max_abs={diff.max().item()} mismatches={int((a != b).sum())}")


def prepare_store(case: Case, pto_kwargs: dict) -> Prepared:
    device = _device()
    torch.manual_seed(0)
    layout = case.layout
    extra = 2 if case.pattern in ("cross_page", "leading_invalid") else 0
    num_blocks = (case.tokens + case.block_size - 1) // case.block_size + extra
    kv = torch.randn(case.tokens, layout.num_elements, dtype=torch.bfloat16, device=device)
    slots = _slots(case.tokens, case.block_size, case.pattern, device)
    guarded = case.pattern not in ("sequential", "random")
    if guarded:
        asc_storage, asc_cache = _guarded_store_cache(
            layout, num_blocks, case.block_size, case.pattern, device
        )
        matched_storage, matched_cache = _guarded_store_cache(
            layout, num_blocks, case.block_size, case.pattern, device
        )
        pto_storage, pto_cache = _guarded_store_cache(
            layout, num_blocks, case.block_size, case.pattern, device
        )
        matched_storage.copy_(asc_storage)
        pto_storage.copy_(asc_storage)
    elif layout.quantized_dtype is None:
        base = torch.zeros(num_blocks, case.block_size, layout.num_elements, dtype=torch.bfloat16, device=device)
        asc_cache, matched_cache, pto_cache = base.clone(), base.clone(), base.clone()
        asc_storage = matched_storage = pto_storage = None
    else:
        base, block_stride = _paged_storage(layout, num_blocks, case.block_size, device)
        base.zero_()
        asc_storage, matched_storage, pto_storage = base.clone(), base.clone(), base.clone()
        asc_cache = _cache_view(asc_storage, block_stride, case.block_size, layout)
        matched_cache = _cache_view(matched_storage, block_stride, case.block_size, layout)
        pto_cache = _cache_view(pto_storage, block_stride, case.block_size, layout)
    matched_kwargs = {"tokens_per_tile": pto_kwargs.get("tokens_per_tile", 16)}
    if layout.quantized_dtype is None:
        matched_kwargs["bf16_dma"] = True
    for key in ("num_stages", "contiguous_slots", "skip_invalid_tiles", "input_l2_cache_ctrl"):
        if key in pto_kwargs:
            matched_kwargs[key] = pto_kwargs[key]
    asc_kernel = store_asc(layout, case.block_size)
    asc_matched_kernel = store_asc(layout, case.block_size, **matched_kwargs)
    pto_kernel = store_pto(layout, case.block_size, **pto_kwargs)
    asc_fn = _launch(asc_kernel, (kv, asc_cache, slots))
    asc_matched_fn = _launch(asc_matched_kernel, (kv, matched_cache, slots))
    pto_fn = _launch(pto_kernel, (kv, pto_cache, slots))
    asc_fn()
    asc_matched_fn()
    pto_fn()
    torch.npu.synchronize()

    def compare() -> None:
        if asc_storage is not None:
            _compare_exact("kv_cache_backing/asc_matched", asc_storage, matched_storage)
            _compare_exact("kv_cache_backing/pto", asc_storage, pto_storage)
        else:
            _compare_exact("kv_cache/asc_matched", asc_cache, matched_cache)
            _compare_exact("kv_cache/pto", asc_cache, pto_cache)

    if guarded:
        kv.mul_(-2.0).add_(0.125)
        if layout.token_interleaved and case.tokens >= 128:
            amplitudes = torch.tensor(
                [2.0**exponent for exponent in range(-layout.num_scales // 2, layout.num_scales // 2)],
                dtype=kv.dtype,
                device=device,
            ).repeat_interleave(layout.elements_per_sf)
            kv.mul_(amplitudes)
            bits = torch.arange(65536, dtype=torch.int32)
            bits.masked_fill_((bits & 0x7FFF) >= 0x7F80, 0)
            finite = bits.to(torch.uint16).view(torch.bfloat16).reshape(128, 512)
            kv[:128].copy_(finite.to(device))
        asc_fn()
        asc_matched_fn()
        pto_fn()
        torch.npu.synchronize()
        compare()
        kv.zero_()
        asc_fn()
        asc_matched_fn()
        pto_fn()
        torch.npu.synchronize()
        compare()

    return Prepared(
        asc_fn=asc_fn,
        pto_fn=pto_fn,
        compare=compare,
        io_bytes=kv.nbytes + asc_cache.nbytes + slots.nbytes,
        asc_kernel=asc_kernel,
        pto_kernel=pto_kernel,
        asc_matched_fn=asc_matched_fn,
    )


def prepare_load(case: Case, pto_kwargs: dict) -> Prepared:
    device = _device()
    torch.manual_seed(0)
    layout = case.layout
    num_blocks = (case.tokens + case.block_size - 1) // case.block_size
    segments = _load_segments(case, num_blocks, device)
    if layout.quantized_dtype is None:
        cache = torch.randn(num_blocks, case.block_size, layout.num_elements, dtype=torch.bfloat16, device=device)
        asc_out = torch.empty(case.tokens, layout.num_elements, dtype=torch.bfloat16, device=device)
        matched_out = torch.empty_like(asc_out)
        pto_out = torch.empty_like(asc_out)
        asc_kernel = load_asc(layout, case.block_size)
        asc_matched_kernel = load_asc(layout, case.block_size)
        pto_kernel = load_pto(layout, case.block_size, **pto_kwargs)
        asc_fn = _launch(asc_kernel, (segments, asc_out, asc_out, cache, cache))
        asc_matched_fn = _launch(asc_matched_kernel, (segments, matched_out, matched_out, cache, cache))
        pto_fn = _launch(pto_kernel, (segments, pto_out, pto_out, cache, cache))

        def compared() -> None:
            _compare_exact("kv/asc_matched", asc_out, matched_out)
            _compare_exact("kv/pto", asc_out, pto_out)

        io_bytes = cache.nbytes + asc_out.nbytes + segments.nbytes
    else:
        storage, block_stride = _paged_storage(layout, num_blocks, case.block_size, device)
        storage.copy_(torch.randint(0, 256, storage.shape, dtype=torch.uint8, device=device))
        data, sf = _load_views(layout, storage, block_stride, case.block_size)
        kwargs = {}
        if layout.do_cast and layout.quantized_dtype == "e2m1":
            kwargs = {
                "fp4_value_block_stride": data.stride(0) * 2,
                "fp4_value_stride": data.stride(1) * 2,
                "fp4_scale_stride": sf.stride(1),
            }
        if not layout.do_cast:
            kwargs["index_batch_size"] = 4 if case.tokens >= 16384 and case.pattern != "short" else 1
        matched_kwargs = dict(kwargs)
        if layout.do_cast:
            matched_kwargs["cast_tile_multiple"] = 4
        for key in ("index_batch_size", "index_num_stages"):
            if key in pto_kwargs:
                matched_kwargs[key] = pto_kwargs[key]
        asc_kernel = load_asc(layout, case.block_size, **kwargs)
        asc_matched_kernel = load_asc(layout, case.block_size, **matched_kwargs)
        pto_kernel = load_pto(layout, case.block_size, **{**kwargs, **pto_kwargs})
        if layout.do_cast:
            asc_out = torch.empty(case.tokens, layout.num_elements, dtype=torch.bfloat16, device=device)
            matched_out = torch.empty_like(asc_out)
            pto_out = torch.empty_like(asc_out)
            asc_args = (segments, asc_out, asc_out, data, sf)
            matched_args = (segments, matched_out, matched_out, data, sf)
            pto_args = (segments, pto_out, pto_out, data, sf)

            def compared() -> None:
                _compare_dequant("kv/asc_matched", asc_out, matched_out)
                _compare_dequant("kv/pto", asc_out, pto_out)
        else:
            out_dtype = torch.int8 if layout.quantized_dtype == "e2m1" else torch.float8_e4m3fn
            asc_out = torch.empty(case.tokens, layout.quantized_bytes, dtype=out_dtype, device=device)
            matched_out = torch.empty_like(asc_out)
            pto_out = torch.empty_like(asc_out)
            asc_sf = torch.empty(case.tokens, dtype=torch.int32, device=device)
            matched_sf = torch.empty_like(asc_sf)
            pto_sf = torch.empty_like(asc_sf)
            asc_args = (segments, asc_out, asc_sf, data, sf)
            matched_args = (segments, matched_out, matched_sf, data, sf)
            pto_args = (segments, pto_out, pto_sf, data, sf)

            def compared() -> None:
                _compare_exact("kv/asc_matched", asc_out, matched_out)
                _compare_exact("kv_sf/asc_matched", asc_sf, matched_sf)
                _compare_exact("kv/pto", asc_out, pto_out)
                _compare_exact("kv_sf/pto", asc_sf, pto_sf)

        asc_fn = _launch(asc_kernel, asc_args)
        asc_matched_fn = _launch(asc_matched_kernel, matched_args)
        pto_fn = _launch(pto_kernel, pto_args)
        io_bytes = data.nbytes + sf.nbytes + asc_out.nbytes + segments.nbytes
        if not layout.do_cast:
            io_bytes += asc_sf.nbytes
    asc_fn()
    asc_matched_fn()
    pto_fn()
    torch.npu.synchronize()
    return Prepared(
        asc_fn=asc_fn,
        pto_fn=pto_fn,
        compare=compared,
        io_bytes=io_bytes,
        asc_kernel=asc_kernel,
        pto_kernel=pto_kernel,
        asc_matched_fn=asc_matched_fn,
    )


def prepare(case: Case, pto_kwargs: dict) -> Prepared:
    return prepare_store(case, pto_kwargs) if case.op == "store" else prepare_load(case, pto_kwargs)


def _case_kwargs(case: Case, requested: dict) -> dict:
    if requested:
        return requested
    if (
        case.op == "load"
        and case.format == "fp4_index"
        and case.block_size == 32
        and case.tokens == 4096
        and case.pattern == "aligned"
    ):
        return {"index_batch_size": 4}
    if (
        case.op == "load"
        and case.format == "mxfp8_index"
        and case.block_size == 64
        and case.tokens == 4096
        and case.pattern == "unaligned"
    ):
        return {"index_num_stages": 1}
    if case.op == "store" and case.pattern == "random":
        if case.format == "fp8":
            return {"tokens_per_tile": 4}
        if case.format == "fp4":
            return {"tokens_per_tile": 8}
    return {}


def _time(fn: Callable[[], None], device: torch.device, repeat: int, detail: bool = False):
    return do_bench(
        fn,
        backend="msprof_detail" if detail else "msprof",
        warmup=0,
        rep=0,
        _n_warmup=5,
        _n_repeat=repeat,
        return_mode="median",
        device=device,
        cache_size=256,
        early_stop_baseline=None if detail else 10000,
    )


def baseline_cases() -> list[Case]:
    return [
        Case("store", "bf16", 64, 65536, "sequential"),
        Case("store", "fp8", 64, 65536, "sequential"),
        Case("store", "fp4", 64, 65536, "sequential"),
        Case("load", "bf16", 64, 65536, "aligned"),
        Case("load", "fp8", 64, 65536, "aligned"),
        Case("load", "fp4", 64, 65536, "aligned"),
        Case("load", "mxfp8_index", 64, 65536, "aligned"),
    ]


def perf_cases() -> list[Case]:
    cases = []
    for fmt, blocks in (("bf16", (64, 133)), ("fp8", (64, 133)), ("fp4", (64, 133))):
        for block in blocks:
            for pattern in ("sequential", "random"):
                cases.append(Case("store", fmt, block, 65536, pattern))
            cases.append(Case("load", fmt, block, 65536, "aligned"))
    for fmt in ("fp4_index", "mxfp8_index"):
        for block in (32, 64):
            for pattern in ("sequential", "random"):
                cases.append(Case("store", fmt, block, 65536, pattern))
            for tokens in (4096, 65536, 1048576):
                for pattern in ("aligned", "unaligned", "short"):
                    cases.append(Case("load", fmt, block, tokens, pattern))
    return cases


def correctness_cases() -> list[Case]:
    cases = []
    for fmt, blocks in (("bf16", (64, 133)), ("fp8", (64, 133)), ("fp4", (64, 133))):
        for block in blocks:
            for tokens in (1, 233, 67890):
                cases.append(Case("store", fmt, block, tokens, "random_invalid"))
                cases.append(Case("load", fmt, block, tokens, "short"))
    for fmt in ("fp4_index", "mxfp8_index"):
        for block in (32, 64):
            for tokens in (1, 233, 67890):
                cases.append(Case("store", fmt, block, tokens, "random_invalid"))
                cases.append(Case("load", fmt, block, tokens, "short"))
            cases.append(Case("load", fmt, block, 67890, "merged"))
    for fmt in FORMATS:
        for block, tokens, pattern in (
            (64, 1027, "cross_page"),
            (133, 1027, "broken_run"),
            (64, 1027, "all_invalid"),
            (8, 4099, "cross_page"),
            (64, 4099, "invalid_tiles"),
            (8, 4099, "random_invalid"),
        ):
            cases.append(Case("store", fmt, block, tokens, pattern))
    return cases


def _toolchain() -> dict[str, str]:
    ptoas = shutil.which("ptoas")
    version = ""
    if ptoas:
        version = subprocess.check_output([ptoas, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    return {
        "tilelang": importlib.metadata.version("tilelang"),
        "ptoas": version,
        "torch_npu": importlib.metadata.version("torch-npu"),
        "pto_flags": os.environ["PTO_FLAGS"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("baseline", "correctness", "perf"), default="baseline")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--op", choices=("load", "store"))
    parser.add_argument("--format", choices=tuple(FORMATS))
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pattern")
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("BENCH_REP", "20")))
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pto-kwargs", default=os.environ.get("PTO_KERNEL_KWARGS", "{}"))
    parser.add_argument("--variant", default=os.environ.get("BENCH_VARIANT", "default"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.op:
        if not args.format or not args.tokens or not args.pattern:
            parser.error("--op requires --format, --tokens, and --pattern")
        cases = [Case(args.op, args.format, args.block_size, args.tokens, args.pattern)]
    else:
        cases = {"baseline": baseline_cases, "correctness": correctness_cases, "perf": perf_cases}[args.suite]()
        if args.format:
            cases = [case for case in cases if case.format == args.format]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.name in wanted]
    if args.list:
        for case in cases:
            print(case.name)
        return
    if not cases:
        raise SystemExit("no cases selected")
    pto_kwargs = json.loads(args.pto_kwargs)
    device = _device()
    toolchain = _toolchain()
    print(f"device={device} cases={len(cases)} variant={args.variant} pto_kwargs={pto_kwargs}", flush=True)
    print(json.dumps(toolchain), flush=True)
    for case in cases:
        case_kwargs = _case_kwargs(case, pto_kwargs)
        print(f"CASE {case.name} pto_kwargs={case_kwargs}", flush=True)
        prepared = prepare(case, case_kwargs)
        prepared.compare()
        row = {
            "name": case.name,
            "op": case.op,
            "format": case.format,
            "block_size": case.block_size,
            "tokens": case.tokens,
            "pattern": case.pattern,
            "correct": True,
            "variant": args.variant,
            "pto_kwargs": case_kwargs,
            "device": device.index,
            "io_bytes": prepared.io_bytes,
            **toolchain,
        }
        if not args.correctness_only:
            if args.detail:
                asc_profile = _time(prepared.asc_fn, device, args.repeat, detail=True)
                matched_profile = _time(prepared.asc_matched_fn, device, args.repeat, detail=True)
                pto_profile = _time(prepared.pto_fn, device, args.repeat, detail=True)
                row["asc_profile"] = asc_profile.__dict__
                row["asc_matched_profile"] = matched_profile.__dict__
                row["pto_profile"] = pto_profile.__dict__
                asc_us = asc_profile.dur_us
                matched_us = matched_profile.dur_us
                pto_us = pto_profile.dur_us
                print(f"ASC original {asc_profile}", flush=True)
                print(f"ASC matched {matched_profile}", flush=True)
                print(f"PTO {pto_profile}", flush=True)
            else:
                asc_us = _time(prepared.asc_fn, device, args.repeat) * 1000.0
                matched_us = _time(prepared.asc_matched_fn, device, args.repeat) * 1000.0
                pto_us = _time(prepared.pto_fn, device, args.repeat) * 1000.0
            row.update(
                asc_us=asc_us,
                asc_matched_us=matched_us,
                pto_us=pto_us,
                ratio=matched_us / pto_us,
                original_ratio=asc_us / pto_us,
                non_vf_ratio=asc_us / matched_us,
                asc_gbs=prepared.io_bytes / asc_us / 1e3,
                asc_matched_gbs=prepared.io_bytes / matched_us / 1e3,
                pto_gbs=prepared.io_bytes / pto_us / 1e3,
            )
            print(
                f"OK asc={asc_us:.2f}us matched={matched_us:.2f}us pto={pto_us:.2f}us "
                f"vf_ratio={matched_us / pto_us:.3f} non_vf={asc_us / matched_us:.3f}",
                flush=True,
            )
        else:
            print("OK correctness", flush=True)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        with args.results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        del prepared
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
