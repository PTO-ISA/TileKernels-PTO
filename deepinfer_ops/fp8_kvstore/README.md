# Sequential FP8 `store_kv`: matched ASC/PTO reproducer

This standalone experiment isolates the slow VF-only PTO quantization path in sequential FP8 `store_kv`. Both implementations use the same slot classification, 16-token tiles, three-stage pipeline, contiguous 528-byte writes, and `T.copy` DMA in `store_common.py`; only the vector quantization differs. It is based on TileKernels-PTO [`vmi-impl` at `6e20f6b`](https://github.com/PTO-ISA/TileKernels-PTO/commit/6e20f6b). The ASC vector path is ported from DeepInfer [`store_kv.py` at `a679bb86`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv.py).

## Reproduce

Use Python 3.11 on AArch64, CANN 9.2.0, torch-npu 2.9.0.post6, and an Ascend A5 device. Install the public [PTOAS 0.76 wheel](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.76) (the wheel used for these results has SHA-256 `137155df42f899c6c01ce0e08debffcabad6b38bb60c478ba22ee44b69a50316`) and TileLang [`59f3924e`](https://github.com/tile-ai/tilelang/commit/59f3924e85ea2f6d3ccec2f00c87044951a783f7):

```bash
python -m pip install \
  https://github.com/hw-native-sys/PTOAS/releases/download/v0.76/ptoas-0.76-cp311-cp311-manylinux_2_34_aarch64.whl

git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git checkout 59f3924e85ea2f6d3ccec2f00c87044951a783f7
USE_CUDA=0 USE_ROCM=0 python -m pip install .
```

The wrapper discovers conda with `conda info --base`; set `CONDA_SH` to override that path, and set `CANN_SET_ENV` to the CANN 9.2.0 `set_env.sh`. The wrapper supplies the required PTOAS flags, including `--enable-vecscope-mem-bar=false`. Run from this repository root:

```bash
BENCH_DEVICES=0 BENCH_REP=20 TILELANG_CACHE_DIR="$PWD/.cache-fp8-native" \
  bash deepinfer_ops/fp8_kvstore/env_npu.sh \
  python deepinfer_ops/fp8_kvstore/bench.py --variant native_half --detail

BENCH_DEVICES=0 TILELANG_CACHE_DIR="$PWD/.cache-fp8-lowering" \
  bash deepinfer_ops/fp8_kvstore/env_npu.sh \
  python deepinfer_ops/fp8_kvstore/dump_lowering.py --variant all
```

Use a fresh `TILELANG_CACHE_DIR` for each experiment. `bench.py` checks every production PTO result byte-for-byte against ASC before timing, then reports medians from 20 profiled launches after five warmups. It also exports the actual PTOAS and TileLang versions and all PTOAS flags into each JSONL row. Ratio is `ASC time / PTO time`; greater than 1 means PTO is faster. `scale_only` and `pack_only` are output-changing ablations, not valid kernels. `direct_brc` is production-valid only when its reported `correct` field is true.

## Isolated shape and memory traffic

The input is 65,536 contiguous BF16 tokens of dimension 512. Slot mapping is `0..65535`, cache block size is 64, and each output token contains 512 FP8 bytes plus 16 UE8M0 scales. Each group of 32 values has one independent scale. The measured traffic model is 101,974,016 bytes: 67,108,864-byte input, 34,603,008-byte output, and 262,144-byte slot mapping. Hardware counters come from TileLang's `msprof_detail` backend on device 1; no host-side asynchronous launch interval is used as kernel time.

## Results

| variant | valid output | ASC µs | PTO µs | ASC/PTO | PTO AIV cycles |
| --- | :---: | ---: | ---: | ---: | ---: |
| `native_half` | yes | 61.59 | 99.27 | **0.620** | 11,135,268 |
| `native_half_hoisted` | yes | 61.44 | 99.43 | 0.618 | 11,140,988 |
| `compact_ub` | yes | 61.57 | 170.09 | 0.362 | 19,912,083 |
| `direct_brc` | compiler rejection | — | — | — | — |
| `scale_only` ablation | no | 61.43 | 77.03 | 0.797 | 8,907,408 |
| `pack_only` ablation | no | 61.49 | 52.01 | 1.182 | 5,902,842 |

ASC is stable at 61.4–61.6 µs across the controls. Native PTO reaches 1,027 GB/s in the traffic model versus 1,656 GB/s for ASC, and spends 11.14M AIV cycles versus ASC's 7.09M. This is a vector-side gap, not a host-launch artifact. Native PTO's AIV vector utilization is 98.1%; its MTE2 utilization falls from ASC's 53.4% to 30.0% as the longer vector path dominates execution.

Hoisting masks and broadcast constants out of the loop has no measurable benefit. The pack-only ablation removes scale computation and writes zero scales; it is 47.26 µs faster than native PTO and faster than full ASC, but it is not a valid optimization or evidence of a VF performance advantage. Scale-only retains native scale generation but replaces quantized values with zero, removing the scale-dependent multiply. Its 77.03 µs result shows substantial standalone scale cost; the remaining 22.24 µs between scale-only and the full path includes the dependent multiply and conversion chain, so the two ablations must not be interpreted as additive instruction costs.

## Compiler target

`native_half` first reduces each pair of FP32 lanes, broadcasts the pair result to 64 lanes, and then runs clamp, multiply, exponent extraction, and inverse construction on all 64 lanes. This happens eight times per token. ASC instead converts even and odd BF16 values into two FP32 vectors and derives four scales in one 64-lane scale sequence, four times per token. The extra scale sequence is the main controlled difference; the surrounding kernel is shared.

`direct_brc` expresses the intended compact alternative: reduce 128 values to four maxima, run scale arithmetic on four lanes, and broadcast each inverse to 32 lanes. PTOAS 0.76 rejects the register-only `V.vbrc(inverse, size=128, group=4)`:

```text
VMI-UNSUPPORTED: pto.vmi.group_broadcast ... requires full result physical chunks; found padding lane in physical chunk
```

`compact_ub` replaces that broadcast with a four-float UB store, an explicit `VST_VLD` barrier, and a grouped load/broadcast. It is a correctness-valid control, not an optimization: the round-trip costs more than the arithmetic it saves. The requested PTOAS fix is to support this computed compact-vector grouped broadcast directly in registers, or select an equivalent `vselr`/shuffle lowering without UB traffic or an inserted barrier. The committed `direct_brc.pto` is the exact failing compiler input. This recommendation follows from the compiler rejection and the slower UB control; it does not assume a particular physical instruction sequence.

The emitted VPTO supports this attribution. Native-half contains eight each of `vmaxs`, `vmuls`, `vadds`, `vshrs`, and `vshls`, plus 16 scale-path `vsub` operations, in the unrolled token body. Compact-UB halves each of those counts to four and eight, but adds four `VST_VLD` barriers and layout/materialization operations including eight `vpack` and 12 `vor`; it takes 19.91M AIV cycles. Native-half and native-half-hoisted have no `pto.mem_bar` lines in their VPTO, so the 0.620× native result is not an implicit vecscope-barrier regression. `--enable-vecscope-mem-bar=false` was active for every compile and run.

To rerun only the compiler failure from the committed IR, without TileLang compilation or an NPU:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 \
  --enable-vecscope-mem-bar=false --emit-vpto \
  deepinfer_ops/fp8_kvstore/results/lowering/direct_brc.pto \
  -o /tmp/direct_brc.vpto
```

PTOAS 0.76 exits nonzero at `direct_brc.pto:1379`: `%707 = pto.vmi.vbrc %706 {group = 4 : i64} : !pto.vmi.vreg<4xf32> -> !pto.vmi.vreg<128xf32>`. Lines 1350–1381 contain the complete reduce, four-lane scale calculation, broadcast, multiply, and FP8 conversion sequence.

## Files

- `quant_asc.py`: physical-SIMD ASC quantization, including even/odd conversion, grouped maxima, `vselr` inverse-scale selection, and packed stores.
- `quant_pto.py`: VMI implementations and controlled ablations.
- `store_common.py`: unchanged non-VF implementation shared by ASC and PTO.
- `bench.py`: byte-exactness checks, hardware profiling, and raw JSONL output.
- `dump_lowering.py`: PTODSL, PTO, and VPTO export with operation summaries.
- `results/bench.jsonl`: final measured rows.
- `results/lowering/`: compiler outputs corresponding to the committed sources.
