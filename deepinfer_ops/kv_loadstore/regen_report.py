"""Regenerate README.md from the final benchmark JSONL rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "kv_loadstore.jsonl"


def main() -> None:
    latest = {}
    order = []
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kind = "perf" if "asc_us" in row else "correctness"
        key = (kind, row["name"])
        if key not in latest:
            order.append(key)
        latest[key] = row
    perf = [latest[key] for key in order if key[0] == "perf"]
    correctness = [latest[key] for key in order if key[0] == "correctness"]
    if len(perf) != 62 or len(correctness) != 94:
        raise SystemExit(f"incomplete final matrix: {len(correctness)}/94 correctness, {len(perf)}/62 performance")
    counts = Counter(row["format"] for row in correctness)
    misses = [row for row in perf if row["ratio"] < 0.95]

    lines = [
        "# Paged KV load/store on PTO VMI",
        "",
        "Standalone ASC and PTO VMI implementations of DeepInfer paged KV-cache load and store.",
        "The ASC baseline is DeepInfer `a679bb86`; the TileKernels base is `vmi-impl` at `6e20f6b`.",
        "PTO source uses logical VMI operations rather than ASC physical SIMD instructions.",
        "",
        "## Toolchain",
        "",
        "- CANN 9.2.0, torch-npu 2.9.0.post6, Ascend950DT, 72 vector cores.",
        "- [ptoas 0.76](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.76).",
        "- [tile-ai/tilelang `59f3924`](https://github.com/tile-ai/tilelang/tree/59f3924e85ea2f6d3ccec2f00c87044951a783f7), version `0.1.15+cpu.git59f3924e`.",
        "- `PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.",
        "",
        "`ratio = ASC_us / PTO_us`; greater than 1 means PTO is faster. Timing is the on-device msprof mean after five warmup launches, with 20 measured launches. A 256 MiB read-only reduction flushes L2 before each measured launch. The target is ratio >= 0.95.",
        "",
        "## Files and reproduction",
        "",
        "- `load_kv_pto.py`, `store_kv_pto.py`: PTO implementations.",
        "- `load_kv_asc.py`, `store_kv_asc.py`: ASC baselines.",
        "- `kv_quant_pto.py`, `kv_quant_asc.py`: quantization helpers; `common.py`: cache layouts.",
        "- `bench.py`: correctness, profiling, and benchmark harness; `results/kv_loadstore.jsonl`: final raw rows.",
        "- `results/bf16_timing_verification.json`: independent cold/hot event, wall, and hardware-pipe cross-check for the BF16 store comparison.",
        "",
        "Run from the worktree root with the provided CANN environment wrapper:",
        "",
        "```bash",
        "BENCH_DEVICES=1 bash deepinfer_ops/kv_loadstore/env_npu.sh \\",
        "  python deepinfer_ops/kv_loadstore/bench.py --suite correctness --format fp4",
        "BENCH_DEVICES=1 BENCH_REP=20 bash deepinfer_ops/kv_loadstore/env_npu.sh \\",
        "  python deepinfer_ops/kv_loadstore/bench.py --suite perf --format fp4",
        "```",
        "",
        "Use `--op`, `--format`, `--block-size`, `--tokens`, and `--pattern` for one case, `--detail` for AIV pipe profiles, and `--list` to enumerate a suite. NPU execution uses `task-submit` in the measured environment.",
        "",
        "## Coverage and correctness",
        "",
        f"All **{len(correctness)} correctness cases** pass: "
        + ", ".join(f"{name} {counts[name]}" for name in sorted(counts))
        + ".",
        "",
        "The matrix covers BF16, V4.1 FP8/UE8M0 and FP4/E4M3 main caches, FP4/UE8M0 and MXFP8/UE8M0 index caches; block sizes 8/32/64/133 as applicable; token counts 1, 233, 1027, 4099, 67890, and benchmark sizes through 1048576. It includes short/partial/unaligned/merged loads, zero-fill pages, index batching 1 and 4, sequential/random/broken/cross-page/all-invalid stores, padded/strided cache storage, nonzero offsets, 16-byte token alignment, repeated updates, every finite BF16 encoding in directed D512 rows, extreme group magnitudes, and zero inputs.",
        "",
        "Quantized stores and index loads are byte-exact against ASC. BF16 copies are exact. Main-cache dequantization uses the established `rtol=atol=2e-2`, `equal_nan=True` comparison. Guarded stores compare complete backing allocations, including untouched canaries and page gaps.",
        "",
        "## Effective optimizations",
        "",
        "- **Load FP4:** decode 32 scales for four rows with one legal 128-byte load, convert once, and use register `vselr` expansion. This removes four per-row scale scratch stores, barriers, and grouped reloads per 128-value tile.",
        "- **Load FP8:** decode each 16-byte scale row as two legal 8-lane halves, expand in registers, and select scales for 64-value chunks. This replaces pairs of scalar-byte loads, zips, and broadcast construction.",
        "- **Store FP8:** process 64-value halves so grouped maxima and inverse scales stay in native 64-lane registers. No inverse UB spill or reload remains.",
        "- **Store FP4:** exact compact hardware reciprocal is faster than the eight-entry select chain. Four rows share one inverse-store barrier per 128-value chunk; 16-row tiles reduce scheduling overhead while preserving legal 128-value FP4 packing.",
        "- **Store BF16:** staged UB/DMA copies and contiguous page-run coalescing replace element-wise SIMT scatter. Random, invalid, and non-contiguous mappings retain a per-token fallback.",
        "- **Index loads:** single-page tasks use a dedicated loop without unused merge/unit/part control flow. The block-32, 4096-token aligned FP4 case uses four-page batching. The block-64, 4096-token unaligned MXFP8 case uses one pipeline stage. Both non-default configurations are explicit in their result rows.",
        "- **Random main-cache stores:** pipeline depth is tuned separately: FP8 uses four tokens per tile and FP4 uses eight. Sequential quantized stores and index caches retain 16-token tiles. The harness records the effective configuration in every raw and report row.",
        "",
        "Controlled compiler tests established the frontend constraints behind these forms: 16-lane VMI vectors and 32-lane packed-FP4 results are rejected; full-group loads require at least 32 bytes per group; direct grouped broadcast of a computed compact inverse fails VMI layout planning. The delivered paths compile under the standard flags above.",
        "",
        "## BF16 store timing verification",
        "",
        "The vendored ASC BF16 source differs from DeepInfer `a679bb86` only in import paths. Independent NPU events reproduce the large sequential-store ratio; it is an algorithmic difference between direct SIMT global-memory access and staged DMA, not an async-launch timing artifact.",
        "",
        "| timing method | ASC us | PTO us | ratio |",
        "|---|---:|---:|---:|",
        "| msprof, cold cache, mean | 690.94 | 60.84 | 11.357 |",
        "| NPU events, cold cache, median | 690.30 | 60.78 | 11.357 |",
        "| NPU events, hot cache, 50-launch stream | 546.93 | 60.94 | 8.975 |",
        "",
        "Cold event timing places start/end events around each kernel after a 256 MiB L2 flush and synchronizes before reading elapsed times. Hot event timing launches 50 kernels between two events and synchronizes the end event. Synchronized hot wall timing, including host launch/synchronization overhead, gives ASC 563.32 us and PTO 75.62 us. Outputs remain exactly equal after all timing methods.",
        "",
        "Cold event effective bandwidth is 194.8 GB/s for ASC and 2212.5 GB/s for PTO over 134,479,872 logical I/O bytes. Hardware pipe profiles explain the gap: ASC uses 81.28M AIV cycles with 99.88% vector utilization and essentially zero MTE2/MTE3 activity; PTO uses 6.86M cycles with 74.64% MTE2, 18.15% MTE3, and 30.46% scalar utilization. L2 reuse improves ASC hot latency, but the hot ratio is still 8.975.",
        "",
        "## Final performance",
        "",
        f"**{len(perf)} performance cases; {len(perf) - len(misses)} meet or exceed 0.95.**",
        "",
    ]
    if misses:
        lines += ["Cases below target:", ""]
        for row in misses:
            lines.append(f"- `{row['name']}` ratio {row['ratio']:.3f}. See raw row and optimization notes for the measured residual.")
        lines.append("")
    else:
        lines += ["Every measured configuration meets or exceeds the 0.95 target.", ""]
    lines += [
        "| operation | format | block | tokens | pattern | PTO config | ASC us | PTO us | ratio | PTO GB/s |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in perf:
        config = json.dumps(row["pto_kwargs"], sort_keys=True)
        lines.append(
            f"| {row['op']} | {row['format']} | {row['block_size']} | {row['tokens']} | {row['pattern']} | `{config}` "
            f"| {row['asc_us']:.2f} | {row['pto_us']:.2f} | {row['ratio']:.3f} | {row['pto_gbs']:.1f} |"
        )
    lines += [
        "",
        "Small and highly scattered cases include launch, metadata, and per-token DMA overhead; the table reports their measured ratios without treating them as saturated bandwidth cases.",
        "",
    ]
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
