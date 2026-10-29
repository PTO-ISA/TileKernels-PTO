# PTO style against every DeepInfer SimdVF

Each `with T.SimdVF():` line under `deepinfer/model_executor/layers/tilelang_ops/asc` is one of: cleaner, about equal, more verbose, or not ported. `T.SimtVF` is out of scope. A region is not "ported" because another file uses `vsel` or `vcmp`.

VMI vectors are logical. `part`, `UNPK`, `PK`, and `PAT_*` stay on the ASC side. PTO uses `size=`, a mask, `group=`, and `vcvt`.

## Cleaner

Grouped reduction. ASC stores a max from each half of a register. PTO reduces the row once.

```python
# ASC: four half-register maxima
S.vsts(amax_ub[0], S.vcmax(v0, low), dist="ONEPT_B32")
S.vsts(amax_ub[1], S.vcmax(v0, high), dist="ONEPT_B32")
# PTO
amax = V.vcmax(V.vabs(values, mask), mask, group=4)
```

One convert. ASC unpacks or splits `part=0` and `part=1`. PTO converts the logical vector.

```python
# ASC
S.vcvt(S.vld(ub[0], dist="UNPK_B16"), T.float32)
# PTO
V.vcvt(V.vload(ub[0], size=128), "float32")
```

One loop instead of copied accumulators. `dflash` dot used to name `acc0`..`acc3` and four stores. Both sides now index one local with `T.unroll`.

```python
acc = V.alloc_local((ROWS,), V.vreg(VEC, T.float32))
for row in T.unroll(ROWS, explicit=True):
    acc[row] = V.vmula(acc[row], w_v, x_v, mask)
    V.vstore(V.vcadd(acc[row], mask, reassoc=True), sum_ub[row])
```

| source line | driver | why it is shorter |
| --- | --- | --- |
| `norm.py` 235 | `norm/bf16` | `vld2` / `vcgmax` / `vintlv` becomes `vcmax(..., group=8)` and one `vcvt` to FP8 |
| `norm.py` 277 | `norm/quant` | four half-register `vcmax` stores become one grouped `vcmax` |
| `norm.py` 374 | `norm/unpack` | two `UNPK_B16` loads become one `vcvt` |
| `store_kv_cache.py` 152 | `store_kv_cache` | same grouped `vcmax` as quant |
| `store_kv_cache.py` 184 | `store_kv_cache/rows` | the second row is `T.unroll`, not a second copy of the body |
| `dflash.py` 57 | `dflash/dot` | four named accumulators become one unrolled local |
| `dflash.py` 137 | `dflash/mean` | one 128-lane `vcvt`, not two 64-lane iterations |
| `candidate_blocks.py` 198 | `candidate_blocks/max` | `vcgmax` plus `PAT_VL8` becomes `vcmax(..., group=8)` |
| `image_embed_ops.py` 342, bf16 arm | `image_embed_ops/pool_bf16` | even/odd `part` pack becomes one `vcvt` to bfloat16 |
| `fused_transform.py` 477, quant flag | `fused_transform/quant` | one grouped `vcmax` and one `vcvt` to FP8 |
| `load_paged_kv_cache.py` 258, FP4 arm | `load_paged_kv_cache/fp4` | `UNPK4` plus `part=0` becomes `vinterpret_cast` and one `vcvt` |

The FP4 check is the zero codepoint: packed bytes 0 and scale 1 produce zeros. Torch CPU has no e2m1 reference for a non-zero nibble. The convert compiled and simulated on both targets.

## About equal

`size=` and the mask are the extra tokens. The arithmetic is the same load, op, and store.

| source line | driver |
| --- | --- |
| `structure_output.py` 82 | `structure_output` |
| `swa_index.py` 41 | `swa_index` |
| `kv_index_cat.py` 60 | `kv_index_cat` |
| `gather.py` 56 | `gather` (PTO fills with `vbrc`; ASC fills in `T.Parallel`) |
| `schedule_mtp_varlen.py` 85 | `schedule_mtp_varlen` |
| `image_embed_ops.py` 125 | `image_embed_ops/copy` |
| `image_embed_ops.py` 320 | `image_embed_ops/pool` zero of the window |
| `image_embed_ops.py` 342, fp32 arm | `image_embed_ops/pool` |
| `image_embed_ops.py` 375 | `image_embed_ops/scatter` |
| `get_kv_index.py` 52, block table | `get_kv_index/blocks` |
| `get_kv_index.py` 52, identity | `get_kv_index/identity` |
| `get_kv_index.py` 214 | `get_kv_index/dflash` |
| `norm.py` 382 and 420 | `norm/layout` index and gather |
| `norm.py` 154 | `norm/rms` is the naive estimate. `norm/rstd` inlines `compensated_div_add` and `compensated_rsqrt` from `rstd_1ulp.py` into that same VF. It is not a second `SimdVF` |
| `load_paged_kv_cache.py` 258, FP8 arm | `load_paged_kv_cache` |
| `load_paged_kv_cache.py` 326 | `load_paged_kv_cache` zero fill of a missing page |
| `random_sampling.py` 224, 232, 235, 252 | `random_sampling` zero, fold, and both scans |
| `sampler.py` 181, 247, 633 | `sampler` |
| `compress.py` 116, 147, 374 | `compress` prefill and decode stay two regions |
| `candidate_blocks.py` 56, 64, 79, 90, 127, 258, 328 | `candidate_blocks` main kernel |
| `fused_sample.py` 214, 223, 261, 625 | `fused_sample/lse` |
| `fused_sample.py` 531, 580, 592, 667 | `fused_sample/cdf` |
| `fused_transform.py` 477, norm flag | `fused_transform` |
| `fused_transform.py` 477, RoPE flag | `fused_transform/rope` |
| `fused_transform.py` 477, Hadamard flag | `fused_transform/hadamard` |
| `fused_transform.py` 477, FP4 reciprocal LUT | `fused_transform/lut`, one row of 512 |

`candidate_blocks.py` lines 90 and 127 are two launches of the same 64-lane scan, one per core group and one per row tile. They stay two regions.

## More verbose

VMI `pmode` only zeroes inactive lanes. A merging add is a full `vadd` and then `vsel`, so inactive lanes keep the accumulator.

```python
# ASC
count = S.vadd(count, one_i, candidate, "MODE_MERGING")
# PTO
count = V.vsel(candidate, V.vadd(count, one_i, full), count)
```

There is no predicate memory op. ASC `pld` / `pst` in the mask round-trip did not lower, so both drivers store the predicate as 0/1 and reload it. That is the region that runs.

`pto.vmi.vdiv` is f16/f32 only. The non-power-of-two arm converts to f32, divides, and truncates back. That ratio is not a fair ISA win. The power-of-two `dflash` arm uses `vand` with 7 because the block size is 8, which is a different instruction, so that 1.21x is not a fair ISA win either.

Float-to-int `vcvt` has rounding `R`, `A`, `H`, and `Z`. It has no ceiling mode. Non-negative ceiling is the toward-zero convert, plus one when a fraction remains.

| source line | driver | extra |
| --- | --- | --- |
| `sum_smaller_probs.py` 58 | `sum_smaller_probs` | merging add is `vadd` plus `vsel` |
| `fused_sample.py` 35 | `fused_sample/search` | three threshold sums, each a merging add |
| `fused_sample.py` 282 and 292 | `fused_sample/filter` | temperature softmax, then the `2^-32` cutoff |
| `fused_sample.py` 327, 347, 391 | `fused_sample/quartile` | bit-pattern quartiles and the pick |
| `fused_sample.py` 446 | `fused_sample/quartile` | ceiling convert spelled out, because `rounding="C"` is rejected |
| `fused_sample.py` 469, 484, 522 | `fused_sample/mask` | 0/1 mask store and reload |
| `fused_sample.py` 550 and 566 | `fused_sample/mask` | log-mass and the filtered logprob, mask reloaded as 0/1 |
| `get_kv_index.py` 52, non-power-of-two | `get_kv_index/mod` | `vcvt` / `vdiv` / truncating `vcvt` in place of integer `vdiv` |

## Not ported

None of the 64 regions were dropped. Each one has a single-core driver that compiled and matched its torch reference under `msprof op simulator`.

## Ratios that are not ISA wins

`candidate_blocks` main is 1627 ASC cycles against 802 PTO cycles (2.03x). That kernel launches the same 64-lane scan seven times on ASC and four times on PTO. The launches were not restacked to move the number.

`sampler` is 709 against 312 (2.27x) on the same 64-lane scatter. The gap is larger than the extra ASC index reload. The algorithms match; the cycle gap is not explained by a shorter PTO body.

`get_kv_index/dflash` and `get_kv_index/mod` are the two division cases above. Do not read either ratio as the same instruction running faster.
