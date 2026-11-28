# Paged KV load/store on PTO VMI

Standalone ASC and PTO VMI implementations of DeepInfer paged KV-cache load and store.
The ASC baseline is DeepInfer `a679bb86`; the TileKernels base is `vmi-impl` at `6e20f6b`.
PTO source uses logical VMI operations rather than ASC physical SIMD instructions.

## Toolchain

- CANN 9.2.0, torch-npu 2.9.0.post6, Ascend950DT, 72 vector cores.
- [ptoas 0.76](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.76).
- [tile-ai/tilelang `59f3924`](https://github.com/tile-ai/tilelang/tree/59f3924e85ea2f6d3ccec2f00c87044951a783f7), version `0.1.15+cpu.git59f3924e`.
- `PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.

The primary `ratio = matched_ASC_us / PTO_us`; greater than 1 means the PTO SimdVF rewrite is faster after matching non-VF work. `non_vf_ratio = original_ASC_us / matched_ASC_us` measures gains that ASC can also adopt. Timing is the on-device msprof mean after five warmup launches, with 20 measured launches. A 256 MiB read-only reduction flushes L2 before each measured launch. The target is ratio >= 0.95.

## Files and reproduction

- `load_kv_pto.py`, `store_kv_pto.py`: PTO implementations.
- `load_kv_asc.py`, `store_kv_asc.py`: ASC baselines.
- `kv_quant_pto.py`, `kv_quant_asc.py`: quantization helpers; `common.py`: cache layouts.
- `bench.py`: correctness, profiling, and benchmark harness; `results/kv_loadstore.jsonl`: final raw rows.
- `results/bf16_timing_verification.json`: independent cold/hot event, wall, and hardware-pipe cross-check for the BF16 store comparison.
- `results/fp8_matched_profile.jsonl`: original ASC, matched ASC, and PTO pipe profiles explaining the sequential FP8 VF-only result.

Run from the worktree root with the provided CANN environment wrapper:

```bash
BENCH_DEVICES=1 bash deepinfer_ops/kv_loadstore/env_npu.sh \
  python deepinfer_ops/kv_loadstore/bench.py --suite correctness --format fp4
BENCH_DEVICES=1 BENCH_REP=20 bash deepinfer_ops/kv_loadstore/env_npu.sh \
  python deepinfer_ops/kv_loadstore/bench.py --suite perf --format fp4
```

Use `--op`, `--format`, `--block-size`, `--tokens`, and `--pattern` for one case, `--detail` for AIV pipe profiles, and `--list` to enumerate a suite. NPU execution uses `task-submit` in the measured environment.

## Coverage and correctness

All **94 correctness cases** pass: bf16 18, fp4 18, fp4_index 20, fp8 18, mxfp8_index 20.

The matrix covers BF16, V4.1 FP8/UE8M0 and FP4/E4M3 main caches, FP4/UE8M0 and MXFP8/UE8M0 index caches; block sizes 8/32/64/133 as applicable; token counts 1, 233, 1027, 4099, 67890, and benchmark sizes through 1048576. It includes short/partial/unaligned/merged loads, zero-fill pages, index batching 1 and 4, sequential/random/broken/cross-page/all-invalid stores, padded/strided cache storage, nonzero offsets, 16-byte token alignment, repeated updates, every finite BF16 encoding in directed D512 rows, extreme group magnitudes, and zero inputs.

Quantized stores and index loads are byte-exact against ASC. BF16 copies are exact. Main-cache dequantization uses the established `rtol=atol=2e-2`, `equal_nan=True` comparison. Guarded stores compare complete backing allocations, including untouched canaries and page gaps.

## Attribution methodology

Each performance row measures three implementations: original DeepInfer ASC, ASC with matched non-VF work, and PTO. Matched ASC keeps its original `T.simd` arithmetic while adopting PTO's `T.copy` DMA, persistent/pipeline schedule, tile sizes, and scalar control flow where applicable. The primary ratio is therefore a VF-only comparison; the separate non-VF ratio measures the gain available to ASC without a VF rewrite.

SimdVF-specific changes include register scale decode/broadcast, native-lane quantization, reciprocal evaluation, and row interleaving inside the vector scope. Non-VF changes include DMA, tile geometry, buffer stages, index batching, and slot/segment control flow. No hot-path speedup is attributed to SimtVF→SimdVF conversion: BF16 replaces SimtVF with non-VF DMA, and index zero-fill retains the same SimtVF operation on both sides.

## Effective optimizations

**VF-only rewrites:**

- **Load FP4:** decode 32 scales for four rows with one legal 128-byte load, convert once, and use register `vselr` expansion. Matched ASC uses the same row tiling and pipeline.
- **Load FP8:** decode each 16-byte scale row as two legal 8-lane halves, expand in registers, and select scales for 64-value chunks. Matched ASC uses the same row tiling and pipeline.
- **Store FP8:** process 64-value halves so grouped maxima and inverse scales stay in native 64-lane registers, eliminating inverse UB spills.
- **Store FP4:** use exact compact hardware reciprocal and share one inverse-store barrier across four rows per 128-value chunk. ASC keeps its original LUT/vector body in the VF-only comparison.

**Non-VF changes mirrored into matched ASC:**

- **Store BF16:** staged UB/DMA copies and contiguous page-run coalescing replace direct SimtVF global-memory access. Random, invalid, and non-contiguous mappings retain a per-token fallback.
- **Store tiling:** sequential quantized stores use 16 tokens per tile; random main-cache FP8/FP4 use four/eight. Both matched ASC and PTO use the same tile and stage count.
- **Index loads:** single-page tasks use a dedicated loop without merge/unit/part control flow. Block-32, 4096-token aligned FP4 uses four-page batching on both sides; block-64, 4096-token unaligned MXFP8 uses one stage on both sides.

Controlled compiler tests established the frontend constraints behind these forms: 16-lane VMI vectors and 32-lane packed-FP4 results are rejected; full-group loads require at least 32 bytes per group; direct grouped broadcast of a computed compact inverse fails VMI layout planning. The delivered paths compile under the standard flags above.

## BF16 store timing verification

The original `bf16_dma=False` ASC path matches DeepInfer `a679bb86`; the file also provides `bf16_dma=True` as the matched non-VF variant. Independent NPU events reproduce the large original-ASC versus PTO sequential-store ratio; it is an algorithmic difference between direct SIMT global-memory access and staged DMA, not an async-launch timing artifact. This is a non-VF gain: matched ASC uses the same DMA path in the primary table.

| timing method | ASC us | PTO us | ratio |
|---|---:|---:|---:|
| msprof, cold cache, mean | 690.94 | 60.84 | 11.357 |
| NPU events, cold cache, median | 690.30 | 60.78 | 11.357 |
| NPU events, hot cache, 50-launch stream | 546.93 | 60.94 | 8.975 |

Cold event timing places start/end events around each kernel after a 256 MiB L2 flush and synchronizes before reading elapsed times. Hot event timing launches 50 kernels between two events and synchronizes the end event. Synchronized hot wall timing, including host launch/synchronization overhead, gives ASC 563.32 us and PTO 75.62 us. Outputs remain exactly equal after all timing methods.

Cold event effective bandwidth is 194.8 GB/s for ASC and 2212.5 GB/s for PTO over 134,479,872 logical I/O bytes. Hardware pipe profiles explain the gap: ASC uses 81.28M AIV cycles with 99.88% vector utilization and essentially zero MTE2/MTE3 activity; PTO uses 6.86M cycles with 74.64% MTE2, 18.15% MTE3, and 30.46% scalar utilization. L2 reuse improves ASC hot latency, but the hot ratio is still 8.975.

## Final performance

**62 performance cases; 56 meet or exceed 0.95.**

Cases below target:

- `store/fp8_b64_n65536_sequential` ratio 0.627. The native-half PTO VF path is slower. With identical non-VF work, matched ASC uses 7.24M AIV cycles and PTO uses 11.16M; both are 97-98% vector-pipe utilized. The earlier speedup comes from non-VF tiling, not SimdVF.
- `store/fp8_b133_n65536_sequential` ratio 0.627. The native-half PTO VF path is slower. With identical non-VF work, matched ASC uses 7.24M AIV cycles and PTO uses 11.16M; both are 97-98% vector-pipe utilized. The earlier speedup comes from non-VF tiling, not SimdVF.
- `load/fp4_index_b64_n4096_aligned` ratio 0.922. This is a 2-4 us launch/metadata-bound case with matched non-VF work; the small measured residual is not presented as a SimdVF advantage or a hardware-bandwidth limitation.
- `load/mxfp8_index_b32_n4096_unaligned` ratio 0.932. This is a 2-4 us launch/metadata-bound case with matched non-VF work; the small measured residual is not presented as a SimdVF advantage or a hardware-bandwidth limitation.
- `load/mxfp8_index_b32_n4096_short` ratio 0.935. This is a 2-4 us launch/metadata-bound case with matched non-VF work; the small measured residual is not presented as a SimdVF advantage or a hardware-bandwidth limitation.
- `load/mxfp8_index_b64_n4096_aligned` ratio 0.916. This is a 2-4 us launch/metadata-bound case with matched non-VF work; the small measured residual is not presented as a SimdVF advantage or a hardware-bandwidth limitation.

| operation | format | block | tokens | pattern | PTO config | original ASC us | matched ASC us | PTO us | VF-only ratio | non-VF gain |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| store | bf16 | 64 | 65536 | sequential | `{}` | 691.23 | 61.22 | 61.48 | 0.996 | 11.290 |
| store | bf16 | 64 | 65536 | random | `{}` | 692.49 | 141.80 | 141.75 | 1.000 | 4.884 |
| load | bf16 | 64 | 65536 | aligned | `{}` | 39.73 | 39.73 | 39.73 | 1.000 | 1.000 |
| store | bf16 | 133 | 65536 | sequential | `{}` | 699.75 | 61.26 | 59.84 | 1.024 | 11.423 |
| store | bf16 | 133 | 65536 | random | `{}` | 701.67 | 141.79 | 141.40 | 1.003 | 4.949 |
| load | bf16 | 133 | 65536 | aligned | `{}` | 39.40 | 39.44 | 39.17 | 1.007 | 0.999 |
| store | fp8 | 64 | 65536 | sequential | `{}` | 115.86 | 62.36 | 99.47 | 0.627 | 1.858 |
| store | fp8 | 64 | 65536 | random | `{"tokens_per_tile": 4}` | 146.82 | 146.65 | 146.82 | 0.999 | 1.001 |
| load | fp8 | 64 | 65536 | aligned | `{}` | 40.78 | 40.76 | 40.16 | 1.015 | 1.000 |
| store | fp8 | 133 | 65536 | sequential | `{}` | 119.03 | 62.34 | 99.49 | 0.627 | 1.909 |
| store | fp8 | 133 | 65536 | random | `{"tokens_per_tile": 4}` | 146.98 | 146.80 | 147.04 | 0.998 | 1.001 |
| load | fp8 | 133 | 65536 | aligned | `{}` | 41.01 | 42.09 | 41.27 | 1.020 | 0.974 |
| store | fp4 | 64 | 65536 | sequential | `{}` | 125.08 | 118.58 | 106.09 | 1.118 | 1.055 |
| store | fp4 | 64 | 65536 | random | `{"tokens_per_tile": 8}` | 133.84 | 133.89 | 133.38 | 1.004 | 1.000 |
| load | fp4 | 64 | 65536 | aligned | `{}` | 29.82 | 30.08 | 28.90 | 1.041 | 0.991 |
| store | fp4 | 133 | 65536 | sequential | `{}` | 124.81 | 118.50 | 105.83 | 1.120 | 1.053 |
| store | fp4 | 133 | 65536 | random | `{"tokens_per_tile": 8}` | 136.86 | 133.59 | 133.62 | 1.000 | 1.024 |
| load | fp4 | 133 | 65536 | aligned | `{}` | 28.36 | 28.84 | 26.90 | 1.072 | 0.983 |
| store | fp4_index | 32 | 65536 | sequential | `{}` | 109.44 | 39.14 | 40.02 | 0.978 | 2.796 |
| store | fp4_index | 32 | 65536 | random | `{}` | 216.21 | 216.03 | 215.84 | 1.001 | 1.001 |
| load | fp4_index | 32 | 4096 | aligned | `{"index_batch_size": 4}` | 2.67 | 2.67 | 2.40 | 1.112 | 1.000 |
| load | fp4_index | 32 | 4096 | unaligned | `{}` | 3.04 | 3.08 | 3.21 | 0.959 | 0.986 |
| load | fp4_index | 32 | 4096 | short | `{}` | 3.55 | 3.57 | 3.71 | 0.962 | 0.996 |
| load | fp4_index | 32 | 65536 | aligned | `{}` | 6.82 | 6.92 | 6.69 | 1.035 | 0.986 |
| load | fp4_index | 32 | 65536 | unaligned | `{}` | 7.08 | 7.35 | 6.82 | 1.077 | 0.964 |
| load | fp4_index | 32 | 65536 | short | `{}` | 19.69 | 19.69 | 19.41 | 1.014 | 1.000 |
| load | fp4_index | 32 | 1048576 | aligned | `{}` | 71.75 | 71.77 | 69.39 | 1.034 | 1.000 |
| load | fp4_index | 32 | 1048576 | unaligned | `{}` | 73.08 | 73.40 | 71.47 | 1.027 | 0.996 |
| load | fp4_index | 32 | 1048576 | short | `{}` | 272.63 | 272.44 | 266.76 | 1.021 | 1.001 |
| store | fp4_index | 64 | 65536 | sequential | `{}` | 109.44 | 39.26 | 40.22 | 0.976 | 2.787 |
| store | fp4_index | 64 | 65536 | random | `{}` | 215.98 | 215.35 | 215.53 | 0.999 | 1.003 |
| load | fp4_index | 64 | 4096 | aligned | `{}` | 2.50 | 2.47 | 2.68 | 0.922 | 1.011 |
| load | fp4_index | 64 | 4096 | unaligned | `{}` | 2.70 | 2.76 | 2.89 | 0.954 | 0.981 |
| load | fp4_index | 64 | 4096 | short | `{}` | 3.07 | 3.07 | 3.17 | 0.967 | 1.000 |
| load | fp4_index | 64 | 65536 | aligned | `{}` | 5.33 | 5.65 | 5.32 | 1.063 | 0.943 |
| load | fp4_index | 64 | 65536 | unaligned | `{}` | 5.69 | 5.58 | 5.59 | 0.998 | 1.020 |
| load | fp4_index | 64 | 65536 | short | `{}` | 11.77 | 11.73 | 11.76 | 0.998 | 1.004 |
| load | fp4_index | 64 | 1048576 | aligned | `{}` | 50.68 | 50.23 | 50.02 | 1.004 | 1.009 |
| load | fp4_index | 64 | 1048576 | unaligned | `{}` | 50.72 | 50.60 | 50.67 | 0.999 | 1.002 |
| load | fp4_index | 64 | 1048576 | short | `{}` | 145.50 | 145.52 | 142.67 | 1.020 | 1.000 |
| store | mxfp8_index | 32 | 65536 | sequential | `{}` | 109.73 | 40.41 | 40.47 | 0.999 | 2.715 |
| store | mxfp8_index | 32 | 65536 | random | `{}` | 215.83 | 216.88 | 216.49 | 1.002 | 0.995 |
| load | mxfp8_index | 32 | 4096 | aligned | `{}` | 2.74 | 2.79 | 2.91 | 0.961 | 0.981 |
| load | mxfp8_index | 32 | 4096 | unaligned | `{}` | 3.06 | 3.03 | 3.25 | 0.932 | 1.010 |
| load | mxfp8_index | 32 | 4096 | short | `{}` | 3.60 | 3.63 | 3.88 | 0.935 | 0.993 |
| load | mxfp8_index | 32 | 65536 | aligned | `{}` | 8.21 | 8.42 | 8.12 | 1.037 | 0.976 |
| load | mxfp8_index | 32 | 65536 | unaligned | `{}` | 8.58 | 8.49 | 8.35 | 1.016 | 1.011 |
| load | mxfp8_index | 32 | 65536 | short | `{}` | 20.46 | 20.22 | 20.15 | 1.004 | 1.012 |
| load | mxfp8_index | 32 | 1048576 | aligned | `{}` | 95.73 | 95.31 | 94.06 | 1.013 | 1.004 |
| load | mxfp8_index | 32 | 1048576 | unaligned | `{}` | 96.07 | 95.68 | 95.03 | 1.007 | 1.004 |
| load | mxfp8_index | 32 | 1048576 | short | `{}` | 281.62 | 281.19 | 277.09 | 1.015 | 1.002 |
| store | mxfp8_index | 64 | 65536 | sequential | `{}` | 109.73 | 40.29 | 40.15 | 1.004 | 2.724 |
| store | mxfp8_index | 64 | 65536 | random | `{}` | 214.75 | 217.19 | 216.36 | 1.004 | 0.989 |
| load | mxfp8_index | 64 | 4096 | aligned | `{}` | 2.54 | 2.50 | 2.73 | 0.916 | 1.016 |
| load | mxfp8_index | 64 | 4096 | unaligned | `{"index_num_stages": 1}` | 2.82 | 2.71 | 2.74 | 0.989 | 1.041 |
| load | mxfp8_index | 64 | 4096 | short | `{}` | 3.02 | 3.14 | 3.24 | 0.968 | 0.963 |
| load | mxfp8_index | 64 | 65536 | aligned | `{}` | 7.36 | 6.96 | 7.28 | 0.957 | 1.056 |
| load | mxfp8_index | 64 | 65536 | unaligned | `{}` | 7.21 | 7.53 | 7.07 | 1.064 | 0.958 |
| load | mxfp8_index | 64 | 65536 | short | `{}` | 12.38 | 12.49 | 12.38 | 1.009 | 0.991 |
| load | mxfp8_index | 64 | 1048576 | aligned | `{}` | 80.94 | 81.08 | 80.85 | 1.003 | 0.998 |
| load | mxfp8_index | 64 | 1048576 | unaligned | `{}` | 81.20 | 81.46 | 81.12 | 1.004 | 0.997 |
| load | mxfp8_index | 64 | 1048576 | short | `{}` | 154.07 | 154.20 | 152.15 | 1.013 | 0.999 |

Small and highly scattered cases include launch, metadata, and per-token DMA overhead; the table reports their measured ratios without treating them as saturated bandwidth cases.
