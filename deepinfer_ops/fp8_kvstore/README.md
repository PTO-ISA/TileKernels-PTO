# Sequential FP8 `store_kv`: matched ASC/PTO reproducer

This standalone experiment isolates the slow VF-only PTO quantization path in sequential FP8 `store_kv`. Both implementations use the same slot classification, 16-token tiles, three-stage pipeline, contiguous 528-byte writes, and `T.copy` DMA in `store_common.py`; only the vector quantization differs. It is based on TileKernels-PTO [`vmi-impl` at `6e20f6b`](https://github.com/PTO-ISA/TileKernels-PTO/commit/6e20f6b). The ASC vector path is ported from DeepInfer [`store_kv_cache.py` at `a679bb86`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py).

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
| `direct_brc_grouped` | compiler rejection | — | — | — | — |
| `direct_brc_packed` | compiler rejection | — | — | — | — |
| `scale_only` ablation | no | 61.43 | 77.03 | 0.797 | 8,907,408 |
| `pack_only` ablation | no | 61.49 | 52.01 | 1.182 | 5,902,842 |

ASC is stable at 61.4–61.6 µs across the controls. Native PTO reaches 1,027 GB/s in the traffic model versus 1,656 GB/s for ASC, and spends 11.14M AIV cycles versus ASC's 7.09M. This is a vector-side gap, not a host-launch artifact. Native PTO's AIV vector utilization is 98.1%; its MTE2 utilization falls from ASC's 53.4% to 30.0% as the longer vector path dominates execution.

Hoisting masks and broadcast constants out of the loop has no measurable benefit. The pack-only ablation removes scale computation and writes zero scales; it is 47.26 µs faster than native PTO and faster than full ASC, but it is not a valid optimization or evidence of a VF performance advantage. Scale-only retains native scale generation but replaces quantized values with zero, removing the scale-dependent multiply. Its 77.03 µs result shows substantial standalone scale cost; the remaining 22.24 µs between scale-only and the full path includes the dependent multiply and conversion chain, so the two ablations must not be interpreted as additive instruction costs.

## Compiler target

### Native PTO: eight 64-lane scale sequences per token

The following is shortened from `quant_pto.py`; the loop bounds, vector widths, and operations are unchanged:

```python
for p in T.unroll(4, explicit=True):
    for half in T.unroll(2, explicit=True):
        offset = p * 128 + half * 64
        values = V.vcvt(V.vload(x[h, offset], size=64), "float32")
        mask = V.create_mask(64, size=64)

        # 64 FP32 values -> two maxima, one for each 32-value group.
        amax = V.vcmax(V.vabs(values, mask), mask, group=2)

        # Repeat each maximum over its 32 lanes before computing scales.
        amax = V.vbrc(amax, size=64, group=2)
        encoded_dense, inverse = compute_scale64(
            amax,
            V.create_mask(64, size=64),
            V.vbrc(T.uint32(1), size=64),
            V.vbrc(T.uint32(254), size=64),
        )
        packed = V.vcvt(
            V.vmul(values, inverse, mask),
            "float8_e4m3fn", rounding="R", saturate="SAT",
        )
```

Each `half` iteration handles 64 values and two scales. The nested `4 × 2` loops therefore invoke `compute_scale64` eight times per token. Despite only two distinct maxima entering the helper, every instruction below operates on 64 lanes:

```python
clamped = V.vmaxs(amax, T.float32(FP8_AMAX_MARGIN), mask)
bits = V.vinterpret_cast(
    V.vmuls(clamped, T.float32(1.0 / FLOAT8E4M3_MAX), mask), "uint32"
)
exponent = V.vadds(
    V.vshrs(V.vsub(bits, one, mask), 23, mask), T.uint32(1), mask
)
inverse = V.vinterpret_cast(
    V.vshls(V.vsub(bias, exponent, mask), 23, mask), "float32"
)
```

### ASC: four 64-lane scale sequences per token

The corresponding ASC excerpt from `quant_asc.py` is:

```python
for p in T.unroll(4, explicit=True):
    raw = S.vld(x[h, p * 128])  # 128 BF16 values = four 32-value groups
    even = S.vcvt(raw, T.float32, part=0)
    odd = S.vcvt(raw, T.float32, part=1)

    maxima16 = S.vcgmax(S.vmax(S.vabs(even), S.vabs(odd)))
    even_maxima, odd_maxima = S.vdintlv(maxima16, maxima16)
    amax = S.vmax(even_maxima, odd_maxima)  # four distinct group maxima

    encoded, inverse = compute_scale(amax)  # one 64-lane scale sequence
    inverse = S.vselr(inverse, S.vshrs(lane, 4))

    q0 = S.vcvt(S.vmul(even, inverse), T.float8_e4m3fn, part=0)
    q1 = S.vcvt(S.vmul(odd, inverse), T.float8_e4m3fn, part=1)
    packed = S.vor(
        T.reinterpret(q0, "uint32x64"), T.reinterpret(q1, "uint32x64")
    )
```

ASC's `compute_scale` uses the same clamp and exponent formula as the PTO helper, with physical-SIMD operations such as `S.vmaxs`, `S.vmuls`, and `S.vshls`. The difference is the data organization around that helper: even/odd conversion plus `vcgmax`/`vdintlv` lets one 64-lane call produce four scales. There are four `p` iterations, hence four scale sequences per token rather than eight. `vselr` then selects the appropriate inverse for each lane without first computing two separate dense 64-lane half-vectors. This doubled scale-sequence count is the main controlled difference; the surrounding DMA and pipeline code is shared.

### Intended compact PTO form

`direct_brc` removes the redundant dense scale work by keeping the scale computation at four lanes:

```python
values = V.vcvt(V.vload(x[h, p * 128], size=128), "float32")
mask = V.create_mask(128, size=128)
amax = V.vcmax(V.vabs(values, mask), mask, group=4)  # 128 -> 4 maxima
mask4 = V.create_mask(4, size=4)
encoded, inverse = compute_scale4(
    amax, mask4, V.vbrc(T.uint32(1), size=4), V.vbrc(T.uint32(254), size=4)
)
broadcast = V.vbrc(inverse, size=128, group=4)       # 4 -> 128 lanes
packed = V.vcvt(
    V.vmul(values, broadcast, mask),
    "float8_e4m3fn", rounding="R", saturate="SAT",
)
```

PTOAS 0.76 rejects the register-only `V.vbrc(inverse, size=128, group=4)`:

```text
VMI-UNSUPPORTED: pto.vmi.group_broadcast ... requires full result physical chunks; found padding lane in physical chunk
```

`compact_ub` replaces that broadcast with a four-float UB store, an explicit `VST_VLD` barrier, and a grouped load/broadcast. It is a correctness-valid control, not an optimization: the round-trip costs more than the arithmetic it saves. The requested PTOAS fix is to support this computed compact-vector grouped broadcast directly in registers, or select an equivalent `vselr`/shuffle lowering without UB traffic or an inserted barrier. The committed `direct_brc.pto` is the exact failing compiler input. This recommendation follows from the compiler rejection and the slower UB control; it does not assume a particular physical instruction sequence.

Related quant kernels provide two relevant patterns. [`per_block_cast_vmi.py`](https://github.com/PTO-ISA/TileKernels-PTO/blob/6e20f6b/tile_kernels/quant/per_block_cast_vmi.py#L110-L145) uses `create_mask(1, size=4, group=4)`, computed four-lane scales, direct grouped broadcast, and a `vunzip`-based UE8M0 packing path. `direct_brc_grouped` applies the grouped mask alone; `direct_brc_packed` additionally applies the packed scale store. Both controls still fail at the same computed `4xf32` to `128xf32` broadcast with the same padding-lane diagnostic. These results rule out either change as a sufficient frontend workaround; they do not show that the per-block kernel itself is broken.

[`per_token_cast_vmi.py`](https://github.com/PTO-ISA/TileKernels-PTO/blob/6e20f6b/tile_kernels/quant/per_token_cast_vmi.py#L223-L246) instead reduces all groups, computes scales together, and then reloads grouped inverse scales. A batched two-phase `store_kv` variant could use this structure to replace four independent compact-UB round trips with one scale phase and two phase boundaries. This is a candidate for future measurement, not a claimed performance improvement.

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
