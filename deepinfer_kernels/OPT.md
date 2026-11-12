# PTO perf-miss candidates

History of spellings tried on the rows with ratio < 0.95. The report table is the latest kept measurement. A row stays in the report note only for the spelling that was kept.

Ratio is ASC us / PTO us. Kept means the spelling is in the tree. Dropped spellings are not.

Device runs used `BENCH_REP=10`. The logical VMI form is what stayed when a longer form did not cross 0.95.

## Membar-off rebench

All rows below 0.95 were rerun with `--enable-vecscope-mem-bar=false` in `PTO_FLAGS` (ptoas 0.72, out of box). The flag stops ptoas from inserting a `pto.mem_bar "VST_VST"` after every versioned-buffer store; no kernel here carries a real UB hazard that needs it, and every rebenched row stayed correct against ASC. Most kept spellings now measure at or above 0.95: `gather/large_d128` 0.924, `kv_index/paged_n1024_b128` 0.961, `sampler/remap_draft_topk` 0.991, `candidate/restore_indices` 1.018, `kv_index_cat` 1.003, `candidate/block_max` 1.002, `bitmask` 0.967, `swa` 1.013, `dflash/input_store` 1.161, `norm/bf16` 0.999, `image/batch_downsample_concat` 1.002, `image/batch_downsample_pool` 0.978, `fused_transform/norm_rope` 0.986. The quant rows improved but stay below: `load_kv/v41_fp4` 0.454, `load_kv/v41_fp8` 0.811, `store_kv/v41_fp4` 0.598, `store_kv/v41_fp8` 0.705. `norm/bf16_residual` 0.761, `fused_transform/norm_fp8` 0.835, `cp_mirror` 0.828, and `schedule` 0.895 did not move; their gap is not barrier cost. The per-entry ratios below predate this rebench.

## Gather

- `kv_index/paged_n1024_b128`. Full-mask `vgather` after clamping out-of-range lanes to block 0. Kept. Device 0: ASC 14.32 µs, PTO 53.67 µs, ratio 0.267 (was 55.78 µs / 0.253). `T.vmi.vgather` on 64 int32 lanes already lowers to one `vgather2_bc` with no trailing `vsel`. The kept function adds one clamp `vsel` and is still shorter than pasting `vgather2`. The 4× gap is the serial 128-vector carry.
- `sampler/remap_draft_topk`. Same full-mask gather, 16-vector lookup `T.unroll`. Kept. ASC 10.21 µs, PTO 24.79 µs, ratio 0.412 (was 25.60 µs / 0.402). Still shorter than the ASC `vgather2` body.
- `candidate/restore_indices`. Same full-mask gather, unrolled across the token block. Kept. ASC 2.42 µs, PTO 3.42 µs, ratio 0.707 (was 3.48 µs / 0.674). Still shorter than the ASC twin.
- `kv_index_cat/q16384_w128_k1024`. `T.serial` matching ASC was slower: ASC 43.88 µs, PTO 83.31 µs, ratio 0.527. Dropped. The explicit unroll stays at 0.686.
- `candidate/block_max`. `vcmax` on bf16 does not compile (`vcmax` rejects bf16). Dropped. The f32 convert plus `vcmax(group=8)` stays at 0.317. An even/odd split would need `part=`, which this spelling does not have.

## Quant

- `store_kv/v41_fp4`. Eight native-lane `vsel` replace the 64-lane `vselr` spill, and the row's four inverses share one grouped reload. Kept. ASC 10.03 µs, PTO 28.12 µs, ratio 0.357 (was 32.89 µs / 0.305). The eight `vsel` are shorter than the spill. A grouped `vbrc` of a computed inverse is still illegal, so the slot store remains.
- `store_kv/v41_fp8`. The same row-wide inverse batch was slower: ASC 9.62 µs, PTO 24.41 µs, ratio 0.394 (was 20.96 µs / 0.457). Dropped. FP8 keeps the per-chunk spill and grouped reload.
- `load_kv/v41_fp4`. Storing the four group scales and reloading them after one barrier returned NaNs, with both a 2-D slot buffer and a flat buffer. Dropped. The per-group store, barrier, and grouped `vbrc` stays at 0.329.
- `load_kv/v41_fp8`. No spelling change. The scale path is still `vzip` plus `vbrc`. Ratio 0.429.

## Reduction

- `norm/bf16`. Two rows in one VF, one `vcadd` each, one `rstd` on the stored square-sums. ASC 270.71 µs, PTO 1063.45 µs, ratio 0.255 (was 923.15 µs / 0.294). Dropped. Plain bf16 stays one row per VF.
- `norm/bf16_residual`. Same two-row VF. Kept. ASC 584.44 µs, PTO 775.50 µs, ratio 0.754 (was 1426.38 µs / 0.414). The square-sum is a one-lane store, which is the `norm_forward_vmi` form, and the compensated `rstd` is one call rather than a second layout dimension. It does not reach 0.95.
- `fused_transform/norm_fp8_h8_d128_n1024`. The 8 grouped maxima are stored with an 8-lane mask and reloaded. A 64-lane mask was rejected (`vstore` requires the mask lane count to match the 8-lane result). Kept. ASC 13.47 µs, PTO 21.48 µs, ratio 0.627 (was 28.31 µs / 0.481). Shorter than the per-lane `vcadd`/`vsel` extract.
- `fused_transform/norm_rope_h8_d128_n4096`. No spelling change. Both sides use the same VF, and the source has no extra integer divide. Ratio 0.899.

## Broadcast

- `bitmask/b512_v129280_ftp`. One 64-word load plus `vselr` covers 32 logit vectors. The tail stays scalar `vload`+`vbrc`. Kept. ASC 157.57 µs, PTO 256.08 µs, ratio 0.615 (was 322.31 µs / 0.492). `T.vmi.vload` has no `dist_mode`, so `BRC_B32` is not a user spelling. The hoist does not reach 0.95, and `dist_mode` stays out.
- `swa/n8192_w4096`. The four pivot broadcasts were already outside the column loop. Rebench: ASC 34.91 µs, PTO 83.39 µs, ratio 0.419. No edit. `dist_mode` stays out for the same reason.

## Medium

- `gather/large_d128`. Replacing the in-VF zero with a copy of one prefilled zero row: ASC 45.15 µs, PTO 7626.23 µs, ratio 0.006. Dropped.
- `gather/large_d128` ASC `zero_fill="simd"`. Explicit `pset` + `vdup` + one `vsts` loop versus the `T.Parallel` sugar: 44.61 µs versus 44.66 µs, ratio 1.001. Same result. The default published body stays the sugar.
- `gather/large_d128` PTO. One unmasked `vstore` over 256-byte chunks, `vbrc` once. Kept. ASC 44.56 µs, PTO 65.06 µs, ratio 0.685 (was 74.33 µs / 0.604). The double `T.serial` masked store stays out.
- `image/batch_downsample_concat`. Hoisted scatter ramp and unrolled the vector loop. Kept. ASC 126.49 µs, PTO 191.88 µs, ratio 0.659 (was 193.92 µs / 0.653). Both sides already `vscatter`. The report line that said ASC uses copies was wrong.
- `dflash/input_store`. `T.unroll(explicit=True)` over 64 chunks: ASC 3.65 µs, PTO 5.87 µs, ratio 0.621 (was 5.67 µs / 0.660). Dropped. The serial loop stays.
- `cp_mirror`, `compress/decode`, `schedule`. No spelling change. Constant shifts stay. `vbrc(vcadd)` stays. Compress divides in float on both sides. Ratios 0.830, 0.879, 0.901.

## Near the bar

Rebenched with the same body. None crossed 0.95. No instruction was added.

- `image/batch_downsample_pool`. ASC 56.80 µs, PTO 61.64 µs, ratio 0.922 (was 0.920). Whole-vector `vcvt`. `part=` stays out.
- `sampler/get_accept_result`. ASC 8.87 µs, PTO 9.74 µs, ratio 0.911 (was 0.926). Identical SIMT scan. The residual is launch and measurement.
- `store_kv/bf16`. ASC 98.94 µs, PTO 106.55 µs, ratio 0.929 (was 0.930). Identical copy schedule.
- `flash_mla/b256_k1024_full`. ASC 7.10 µs, PTO 7.50 µs, ratio 0.947 (was 0.941). Identical SIMT body.
