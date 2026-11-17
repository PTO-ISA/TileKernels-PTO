# Paged KV load/store on PTO VMI

Standalone ASC and PTO VMI implementations of DeepInfer paged KV-cache load and store.
The ASC baseline is DeepInfer `a679bb86`; the TileKernels base is `vmi-impl` at `6e20f6b`.
PTO source uses logical VMI operations rather than ASC physical SIMD instructions.

## Toolchain

- CANN 9.2.0, torch-npu 2.9.0.post6, Ascend950DT, 72 vector cores.
- [ptoas 0.76](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.76).
- [tile-ai/tilelang `59f3924`](https://github.com/tile-ai/tilelang/tree/59f3924e85ea2f6d3ccec2f00c87044951a783f7), version `0.1.15+cpu.git59f3924e`.
- `PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.

`ratio = ASC_us / PTO_us`; greater than 1 means PTO is faster. Timing is the on-device msprof mean after five warmup launches, with 20 measured launches. A 256 MiB read-only reduction flushes L2 before each measured launch. The target is ratio >= 0.95.

## Files and reproduction

- `load_kv_pto.py`, `store_kv_pto.py`: PTO implementations.
- `load_kv_asc.py`, `store_kv_asc.py`: ASC baselines.
- `kv_quant_pto.py`, `kv_quant_asc.py`: quantization helpers; `common.py`: cache layouts.
- `bench.py`: correctness, profiling, and benchmark harness; `results/kv_loadstore.jsonl`: final raw rows.
- `results/bf16_timing_verification.json`: independent cold/hot event, wall, and hardware-pipe cross-check for the BF16 store comparison.

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

## Effective optimizations

- **Load FP4:** decode 32 scales for four rows with one legal 128-byte load, convert once, and use register `vselr` expansion. This removes four per-row scale scratch stores, barriers, and grouped reloads per 128-value tile.
- **Load FP8:** decode each 16-byte scale row as two legal 8-lane halves, expand in registers, and select scales for 64-value chunks. This replaces pairs of scalar-byte loads, zips, and broadcast construction.
- **Store FP8:** process 64-value halves so grouped maxima and inverse scales stay in native 64-lane registers. No inverse UB spill or reload remains.
- **Store FP4:** exact compact hardware reciprocal is faster than the eight-entry select chain. Four rows share one inverse-store barrier per 128-value chunk; 16-row tiles reduce scheduling overhead while preserving legal 128-value FP4 packing.
- **Store BF16:** staged UB/DMA copies and contiguous page-run coalescing replace element-wise SIMT scatter. Random, invalid, and non-contiguous mappings retain a per-token fallback.
- **Index loads:** single-page tasks use a dedicated loop without unused merge/unit/part control flow. The block-32, 4096-token aligned FP4 case uses four-page batching. The block-64, 4096-token unaligned MXFP8 case uses one pipeline stage. Both non-default configurations are explicit in their result rows.
- **Random main-cache stores:** pipeline depth is tuned separately: FP8 uses four tokens per tile and FP4 uses eight. Sequential quantized stores and index caches retain 16-token tiles. The harness records the effective configuration in every raw and report row.

Controlled compiler tests established the frontend constraints behind these forms: 16-lane VMI vectors and 32-lane packed-FP4 results are rejected; full-group loads require at least 32 bytes per group; direct grouped broadcast of a computed compact inverse fails VMI layout planning. The delivered paths compile under the standard flags above.

## BF16 store timing verification

The vendored ASC BF16 source differs from DeepInfer `a679bb86` only in import paths. Independent NPU events reproduce the large sequential-store ratio; it is an algorithmic difference between direct SIMT global-memory access and staged DMA, not an async-launch timing artifact.

| timing method | ASC us | PTO us | ratio |
|---|---:|---:|---:|
| msprof, cold cache, mean | 690.94 | 60.84 | 11.357 |
| NPU events, cold cache, median | 690.30 | 60.78 | 11.357 |
| NPU events, hot cache, 50-launch stream | 546.93 | 60.94 | 8.975 |

Cold event timing places start/end events around each kernel after a 256 MiB L2 flush and synchronizes before reading elapsed times. Hot event timing launches 50 kernels between two events and synchronizes the end event. Synchronized hot wall timing, including host launch/synchronization overhead, gives ASC 563.32 us and PTO 75.62 us. Outputs remain exactly equal after all timing methods.

Cold event effective bandwidth is 194.8 GB/s for ASC and 2212.5 GB/s for PTO over 134,479,872 logical I/O bytes. Hardware pipe profiles explain the gap: ASC uses 81.28M AIV cycles with 99.88% vector utilization and essentially zero MTE2/MTE3 activity; PTO uses 6.86M cycles with 74.64% MTE2, 18.15% MTE3, and 30.46% scalar utilization. L2 reuse improves ASC hot latency, but the hot ratio is still 8.975.

## Final performance

**62 performance cases; 62 meet or exceed 0.95.**

Every measured configuration meets or exceeds the 0.95 target.

| operation | format | block | tokens | pattern | PTO config | ASC us | PTO us | ratio | PTO GB/s |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| store | bf16 | 64 | 65536 | sequential | `{}` | 691.14 | 60.87 | 11.355 | 2209.4 |
| store | bf16 | 64 | 65536 | random | `{}` | 691.99 | 142.08 | 4.870 | 946.5 |
| load | bf16 | 64 | 65536 | aligned | `{}` | 31.25 | 31.32 | 0.998 | 4285.3 |
| store | bf16 | 133 | 65536 | sequential | `{}` | 699.51 | 61.91 | 11.299 | 2172.7 |
| store | bf16 | 133 | 65536 | random | `{}` | 702.28 | 141.55 | 4.961 | 950.3 |
| load | bf16 | 133 | 65536 | aligned | `{}` | 36.56 | 36.57 | 1.000 | 3671.1 |
| store | fp8 | 64 | 65536 | sequential | `{}` | 116.00 | 99.39 | 1.167 | 1026.0 |
| store | fp8 | 64 | 65536 | random | `{"tokens_per_tile": 4}` | 146.80 | 146.98 | 0.999 | 693.8 |
| load | fp8 | 64 | 65536 | aligned | `{}` | 40.58 | 40.02 | 1.014 | 2542.1 |
| store | fp8 | 133 | 65536 | sequential | `{}` | 118.79 | 99.40 | 1.195 | 1026.1 |
| store | fp8 | 133 | 65536 | random | `{"tokens_per_tile": 4}` | 146.47 | 146.76 | 0.998 | 695.0 |
| load | fp8 | 133 | 65536 | aligned | `{}` | 41.22 | 40.88 | 1.008 | 2488.7 |
| store | fp4 | 64 | 65536 | sequential | `{}` | 124.96 | 105.90 | 1.180 | 814.4 |
| store | fp4 | 64 | 65536 | random | `{"tokens_per_tile": 8}` | 132.70 | 133.45 | 0.994 | 646.3 |
| load | fp4 | 64 | 65536 | aligned | `{}` | 30.11 | 29.12 | 1.034 | 2952.8 |
| store | fp4 | 133 | 65536 | sequential | `{}` | 124.66 | 105.86 | 1.178 | 814.8 |
| store | fp4 | 133 | 65536 | random | `{"tokens_per_tile": 8}` | 137.37 | 133.11 | 1.032 | 648.0 |
| load | fp4 | 133 | 65536 | aligned | `{}` | 30.59 | 29.04 | 1.053 | 2961.1 |
| store | fp4_index | 32 | 65536 | sequential | `{}` | 109.43 | 39.99 | 2.736 | 537.5 |
| store | fp4_index | 32 | 65536 | random | `{}` | 216.22 | 216.32 | 1.000 | 99.4 |
| load | fp4_index | 32 | 4096 | aligned | `{"index_batch_size": 4}` | 2.79 | 2.58 | 1.081 | 216.3 |
| load | fp4_index | 32 | 4096 | unaligned | `{}` | 3.16 | 3.21 | 0.985 | 174.3 |
| load | fp4_index | 32 | 4096 | short | `{}` | 3.66 | 3.68 | 0.995 | 152.5 |
| load | fp4_index | 32 | 65536 | aligned | `{}` | 7.00 | 6.58 | 1.063 | 1359.5 |
| load | fp4_index | 32 | 65536 | unaligned | `{}` | 7.04 | 7.10 | 0.990 | 1259.1 |
| load | fp4_index | 32 | 65536 | short | `{}` | 20.31 | 19.45 | 1.044 | 461.2 |
| load | fp4_index | 32 | 1048576 | aligned | `{}` | 71.55 | 69.05 | 1.036 | 2072.9 |
| load | fp4_index | 32 | 1048576 | unaligned | `{}` | 72.35 | 70.42 | 1.027 | 2032.5 |
| load | fp4_index | 32 | 1048576 | short | `{}` | 282.18 | 264.79 | 1.066 | 542.1 |
| store | fp4_index | 64 | 65536 | sequential | `{}` | 109.40 | 40.09 | 2.729 | 536.1 |
| store | fp4_index | 64 | 65536 | random | `{}` | 216.10 | 215.56 | 1.003 | 99.7 |
| load | fp4_index | 64 | 4096 | aligned | `{}` | 2.60 | 2.67 | 0.974 | 209.0 |
| load | fp4_index | 64 | 4096 | unaligned | `{}` | 2.92 | 2.88 | 1.015 | 194.0 |
| load | fp4_index | 64 | 4096 | short | `{}` | 3.18 | 3.22 | 0.989 | 173.8 |
| load | fp4_index | 64 | 65536 | aligned | `{}` | 5.35 | 5.46 | 0.981 | 1635.8 |
| load | fp4_index | 64 | 65536 | unaligned | `{}` | 5.71 | 5.41 | 1.057 | 1651.5 |
| load | fp4_index | 64 | 65536 | short | `{}` | 12.10 | 11.74 | 1.031 | 761.7 |
| load | fp4_index | 64 | 1048576 | aligned | `{}` | 49.19 | 49.67 | 0.990 | 2876.1 |
| load | fp4_index | 64 | 1048576 | unaligned | `{}` | 50.08 | 49.76 | 1.006 | 2870.9 |
| load | fp4_index | 64 | 1048576 | short | `{}` | 149.64 | 141.39 | 1.058 | 1011.9 |
| store | mxfp8_index | 32 | 65536 | sequential | `{}` | 109.71 | 40.48 | 2.710 | 634.6 |
| store | mxfp8_index | 32 | 65536 | random | `{}` | 215.90 | 217.04 | 0.995 | 118.4 |
| load | mxfp8_index | 32 | 4096 | aligned | `{}` | 2.81 | 2.93 | 0.957 | 369.3 |
| load | mxfp8_index | 32 | 4096 | unaligned | `{}` | 3.30 | 3.20 | 1.031 | 338.7 |
| load | mxfp8_index | 32 | 4096 | short | `{}` | 3.92 | 3.75 | 1.047 | 289.7 |
| load | mxfp8_index | 32 | 65536 | aligned | `{}` | 8.57 | 8.04 | 1.066 | 2155.7 |
| load | mxfp8_index | 32 | 65536 | unaligned | `{}` | 8.31 | 8.56 | 0.971 | 2025.1 |
| load | mxfp8_index | 32 | 65536 | short | `{}` | 21.18 | 20.01 | 1.058 | 867.3 |
| load | mxfp8_index | 32 | 1048576 | aligned | `{}` | 91.78 | 90.04 | 1.019 | 3080.2 |
| load | mxfp8_index | 32 | 1048576 | unaligned | `{}` | 92.15 | 90.59 | 1.017 | 3061.4 |
| load | mxfp8_index | 32 | 1048576 | short | `{}` | 289.25 | 273.74 | 1.057 | 1014.7 |
| store | mxfp8_index | 64 | 65536 | sequential | `{}` | 109.92 | 40.30 | 2.727 | 637.4 |
| store | mxfp8_index | 64 | 65536 | random | `{}` | 214.83 | 217.43 | 0.988 | 118.2 |
| load | mxfp8_index | 64 | 4096 | aligned | `{}` | 2.74 | 2.67 | 1.028 | 406.1 |
| load | mxfp8_index | 64 | 4096 | unaligned | `{"index_num_stages": 1}` | 2.80 | 2.71 | 1.031 | 398.7 |
| load | mxfp8_index | 64 | 4096 | short | `{}` | 3.14 | 3.19 | 0.983 | 339.3 |
| load | mxfp8_index | 64 | 65536 | aligned | `{}` | 7.36 | 6.93 | 1.063 | 2499.8 |
| load | mxfp8_index | 64 | 65536 | unaligned | `{}` | 7.14 | 7.33 | 0.974 | 2361.5 |
| load | mxfp8_index | 64 | 65536 | short | `{}` | 12.65 | 12.43 | 1.018 | 1394.5 |
| load | mxfp8_index | 64 | 1048576 | aligned | `{}` | 76.41 | 76.49 | 0.999 | 3622.6 |
| load | mxfp8_index | 64 | 1048576 | unaligned | `{}` | 76.98 | 77.48 | 0.993 | 3576.1 |
| load | mxfp8_index | 64 | 1048576 | short | `{}` | 155.86 | 148.31 | 1.051 | 1869.7 |

Small and highly scattered cases include launch, metadata, and per-token DMA overhead; the table reports their measured ratios without treating them as saturated bandwidth cases.
