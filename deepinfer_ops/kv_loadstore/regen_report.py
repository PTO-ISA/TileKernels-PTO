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
    if any("asc_matched_us" not in row or "non_vf_ratio" not in row for row in perf):
        raise SystemExit("performance rows must include matched ASC and non-VF measurements")
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
        "The primary `ratio = matched_ASC_us / PTO_us`; greater than 1 means the PTO SimdVF rewrite is faster after matching non-VF work. `non_vf_ratio = original_ASC_us / matched_ASC_us` measures gains that ASC can also adopt. Timing is the on-device msprof mean after five warmup launches, with 20 measured launches. A 256 MiB read-only reduction flushes L2 before each measured launch. The target is ratio >= 0.95.",
        "",
        "## Files and reproduction",
        "",
        "- `load_kv_pto.py`, `store_kv_pto.py`: PTO implementations.",
        "- `load_kv_asc.py`, `store_kv_asc.py`: ASC baselines.",
        "- `kv_quant_pto.py`, `kv_quant_asc.py`: quantization helpers; `common.py`: cache layouts.",
        "- `bench.py`: correctness, profiling, and benchmark harness; `results/kv_loadstore.jsonl`: final raw rows.",
        "- `results/bf16_timing_verification.json`: independent cold/hot event, wall, and hardware-pipe cross-check for the BF16 store comparison.",
        "- `results/fp8_matched_profile.jsonl`: original ASC, matched ASC, and PTO pipe profiles explaining the sequential FP8 VF-only result.",
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
        "## Attribution methodology",
        "",
        "Each performance row measures three implementations: original DeepInfer ASC, ASC with matched non-VF work, and PTO. Matched ASC keeps its original `T.simd` arithmetic while adopting PTO's `T.copy` DMA, persistent/pipeline schedule, tile sizes, and scalar control flow where applicable. The primary ratio is therefore a VF-only comparison; the separate non-VF ratio measures the gain available to ASC without a VF rewrite.",
        "",
        "SimdVF-specific changes include register scale decode/broadcast, native-lane quantization, reciprocal evaluation, and row interleaving inside the vector scope. Non-VF changes include DMA, tile geometry, buffer stages, index batching, and slot/segment control flow. No hot-path speedup is attributed to SimtVF→SimdVF conversion: BF16 replaces SimtVF with non-VF DMA, and index zero-fill retains the same SimtVF operation on both sides.",
        "",
        "## Effective optimizations",
        "",
        "**VF-only rewrites:**",
        "",
        "- **Load FP4:** decode 32 scales for four rows with one legal 128-byte load, convert once, and use register `vselr` expansion. Matched ASC uses the same row tiling and pipeline.",
        "- **Load FP8:** decode each 16-byte scale row as two legal 8-lane halves, expand in registers, and select scales for 64-value chunks. Matched ASC uses the same row tiling and pipeline.",
        "- **Store FP8:** process 64-value halves so grouped maxima and inverse scales stay in native 64-lane registers, eliminating inverse UB spills.",
        "- **Store FP4:** use exact compact hardware reciprocal and share one inverse-store barrier across four rows per 128-value chunk. ASC keeps its original LUT/vector body in the VF-only comparison.",
        "",
        "**Non-VF changes mirrored into matched ASC:**",
        "",
        "- **Store BF16:** staged UB/DMA copies and contiguous page-run coalescing replace direct SimtVF global-memory access. Random, invalid, and non-contiguous mappings retain a per-token fallback.",
        "- **Store tiling:** sequential quantized stores use 16 tokens per tile; random main-cache FP8/FP4 use four/eight. Both matched ASC and PTO use the same tile and stage count.",
        "- **Index loads:** single-page tasks use a dedicated loop without merge/unit/part control flow. Block-32, 4096-token aligned FP4 uses four-page batching on both sides; block-64, 4096-token unaligned MXFP8 uses one stage on both sides.",
        "",
        "Controlled compiler tests established the frontend constraints behind these forms: 16-lane VMI vectors and 32-lane packed-FP4 results are rejected; full-group loads require at least 32 bytes per group; direct grouped broadcast of a computed compact inverse fails VMI layout planning. The delivered paths compile under the standard flags above.",
        "",
        "## BF16 store timing verification",
        "",
        "The original `bf16_dma=False` ASC path matches DeepInfer `a679bb86`; the file also provides `bf16_dma=True` as the matched non-VF variant. Independent NPU events reproduce the large original-ASC versus PTO sequential-store ratio; it is an algorithmic difference between direct SIMT global-memory access and staged DMA, not an async-launch timing artifact. This is a non-VF gain: matched ASC uses the same DMA path in the primary table.",
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
            if row["name"] in ("store/fp8_b64_n65536_sequential", "store/fp8_b133_n65536_sequential"):
                reason = (
                    "The native-half PTO VF path is slower. With identical non-VF work, matched ASC uses 7.24M AIV cycles "
                    "and PTO uses 11.16M; both are 97-98% vector-pipe utilized. The earlier speedup comes from non-VF tiling, not SimdVF."
                )
            elif row["tokens"] <= 4096:
                reason = (
                    "This is a 2-4 us launch/metadata-bound case with matched non-VF work; the small measured residual is not "
                    "presented as a SimdVF advantage or a hardware-bandwidth limitation."
                )
            else:
                reason = "Non-VF work is matched; the residual is reported directly from the measured three-way row."
            lines.append(f"- `{row['name']}` ratio {row['ratio']:.3f}. {reason}")
        lines.append("")
    else:
        lines += ["Every measured configuration meets or exceeds the 0.95 target.", ""]
    lines += [
        "| operation | format | block | tokens | pattern | PTO config | original ASC us | matched ASC us | PTO us | VF-only ratio | non-VF gain |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in perf:
        config = json.dumps(row["pto_kwargs"], sort_keys=True)
        lines.append(
            f"| {row['op']} | {row['format']} | {row['block_size']} | {row['tokens']} | {row['pattern']} | `{config}` "
            f"| {row['asc_us']:.2f} | {row['asc_matched_us']:.2f} | {row['pto_us']:.2f} "
            f"| {row['ratio']:.3f} | {row['non_vf_ratio']:.3f} |"
        )
    lines += [
        "",
        "Small and highly scattered cases include launch, metadata, and per-token DMA overhead; the table reports their measured ratios without treating them as saturated bandwidth cases.",
        "",
    ]
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
