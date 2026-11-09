# Gather on PTO VMI

Port of DeepInfer `gather.py` at `a679bb86ce36e883a98a97733492b686e5b9a08e`:

https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc/gather.py

The TileKernels-PTO base is `vmi-impl` at `f4e8fc8`:

https://github.com/PTO-ISA/TileKernels-PTO/commit/f4e8fc8

The TileLang frontend is branch `pto-dev` at `479d5cd1acfbd31bcd750b728e992409d5cbb175`:

https://github.com/PTO-ISA/tilelang/commit/479d5cd1acfbd31bcd750b728e992409d5cbb175

This directory is the gather kernel only. `asc.py` is the ASC baseline. `pto.py` is the VMI port. `bench.py` rebuilds the int8 view, checks the outputs, and appends msprof rows to `results/gather.jsonl`.

## Shape

`m = n = 131072`. The logical tensor is float32 with `D = 128` and is viewed as int8, so the kernel sees `D = 512` and `x_stride = 512`. The index is int32. `index[::4] = -1` and `index[1::4] = n`, so half the rows are out of range and must stay zero. The fill is on the critical path. 72 vector cores, Ascend950DT. `ratio = ASC_us / PTO_us`. Timing is the msprof median after 5 warmup launches and `BENCH_REP=10`, on device 0.

## ASC baseline

`zero_fill="parallel"` is the published `T.Parallel` sugar. `zero_fill="simd"` is the instruction that sugar lowers to: `pset`, `vdup`, and one `vsts` per 256-byte chunk. The two match and take the same time.

| fill | us | vs parallel |
| --- | ---: | ---: |
| parallel | 44.67 | 1 |
| simd | 44.33 | 0.992 |

## PTO, release ptoas 0.72

https://github.com/hw-native-sys/PTOAS/releases/tag/v0.72

Both styles use one `V.vbrc` and one unmasked `V.vstore` over 256-byte chunks. That is the VMI spelling of the ASC `vsts` loop. The indexed `T.copy` is unchanged.

`style="single"` keeps one `T.alloc_shared` plus `annotate_buffer_versions`. The store address is `(version & 1) * 125952 + chunk * 256`. The alignment proof accepts the product, so the store is one aligned `pto.vsts`. The vecscope carried-dependence proof does not. `(version & 1) * 125952` is not affine in the induction variable, the pair is left unknown, and ptoas inserts `pto.mem_bar "VST_VST"` after every store. Measured 44.67 / 65.57 µs, ratio 0.681.

`style="double"` is the source workaround. Two fixed UB slots are chosen outside the vecscope, and each is opted out of versioning. Each `vsts` address is a constant base plus `chunk * 256`, so the barrier is not inserted. The slot select is the remaining extra scalar work. Measured 44.67 / 48.00 µs, ratio 0.931. Correct against the ASC parallel output.

| style | ASC us | PTO us | ratio | ptoas |
| --- | ---: | ---: | ---: | --- |
| single | 44.67 | 65.57 | 0.681 | 0.72 |
| double | 44.67 | 48.00 | 0.931 | 0.72 |

## Single buffer after the ptoas patch

`style="patched"` is the same source as `single`: one `T.alloc_shared`. Release 0.72 inserts `pto.mem_bar "VST_VST"` because `(version & 1) * 125952` is not affine in the chunk induction variable. The patched compiler treats a value defined outside those induction variables as an exact invariant symbol. The existing mul-by-constant rule makes `(version & 1) * 125952` one symbol. When both accesses of a carried pair use that symbol at the same coefficient, and the address cannot wrap the index width, the symbol cancels and overlap is tested on `chunk * 256` alone. A mismatched coefficient, or an address that can wrap, still emits the barrier.

On that compiler the fill lowers to one `pto.vsts` per chunk and no `VST_VST`. Paired with a fresh ASC timing, the kernel is 44.80 / 48.82 µs, ratio 0.918, and it matches the ASC parallel output. The two-slot spelling is the workaround that release 0.72 can compile today. The patch is:

https://github.com/hw-native-sys/PTOAS/pull/TODO

| style | ASC us | PTO us | ratio | ptoas |
| --- | ---: | ---: | ---: | --- |
| patched | 44.80 | 48.82 | 0.918 | 0.72 + invariant-symbol cancellation |
