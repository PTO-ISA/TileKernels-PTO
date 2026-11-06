# PTO perf-miss candidates

History of spellings tried on the rows with ratio < 0.95. The report table is the latest kept measurement. A row stays in the report note only for the spelling that was kept.

Ratio is ASC us / PTO us. Kept means the spelling is in the tree. Dropped spellings are not.

## Gather

- `kv_index/paged_n1024_b128`. Full-mask `vgather` after clamping out-of-range lanes to block 0. Kept. Device 0, BENCH_REP=10: ASC 14.32 µs, PTO 53.67 µs, ratio 0.267 (was 55.78 µs / 0.253). `T.vmi.vgather` on 64 int32 lanes already lowers to one `vgather2_bc` with no trailing `vsel`. The kept function adds one clamp `vsel` and is still shorter than pasting `vgather2`. The 4× gap is the serial 128-vector carry, not the gather opcode.
- `sampler/remap_draft_topk`. Same full-mask gather, 16-vector lookup `T.unroll`. Kept. ASC 10.21 µs, PTO 24.79 µs, ratio 0.412 (was 25.60 µs / 0.402). Still shorter than the ASC `vgather2` body.
- `candidate/restore_indices`. Same full-mask gather, unrolled across the token block. Kept. ASC 2.42 µs, PTO 3.42 µs, ratio 0.707 (was 3.48 µs / 0.674). Still shorter than the ASC twin.
- `kv_index_cat/q16384_w128_k1024`. `T.serial` matching ASC was slower: ASC 43.88 µs, PTO 83.31 µs, ratio 0.527. Dropped. The explicit unroll stays. Newest kept row is 0.686.
- `candidate/block_max`. `vcmax` on bf16 does not compile (`vcmax` rejects bf16). Dropped. The f32 convert plus `vcmax(group=8)` stays at 0.317. An even/odd split would need `part=`, which the ASC twin has and this spelling does not.

## Quant

- `store_kv/v41_fp4`. Reciprocal lookup is eight `vsel`s on the native lane count. The 64-lane `vselr` spill is gone. Pending device.
- `load_kv/v41_fp4`. The four group scales are stored, then one `VST_VLD` barrier, then broadcast. Pending device.

## Reduction

- `norm/bf16`, `norm/bf16_residual`. The tile's two rows share one VF. Each row is a serial `vcadd` and the rstd is not a second layout dimension. Pending device.
- `candidate/block_max`. `vcmax` on the bf16 vector, group 8, one maximum per 16-wide block, then convert the 8 results. Pending device.
- `fused_transform/norm_fp8_h8_d128_n1024`. The 8 grouped maxima are parked with one zero-fill and an 8-lane store, replacing the per-lane `vcadd`/`vsel` extract. Pending device.

## Broadcast

- `bitmask/b512_v129280_ftp`. A 64-word load plus `vselr` covers 32 logit vectors. The scalar `vload`+`vbrc` remains only on the tail. Pending device.
- `swa/n8192_w4096`. The four pivot broadcasts are already outside the column loop. `T.vmi.vload` has no `dist_mode`, so a `BRC_B32` load is not a user spelling. No further edit unless the rebench is still under 0.95, and then the note is the missing broadcast-load fold.

## Medium

- `gather/large_d128`. The tile is no longer zeroed up front. Out-of-range rows copy one prefilled zero row. Pending device.
- `image/batch_downsample_concat`. The scatter index ramp is hoisted and the vector loop is unrolled. Pending device.
- `dflash/input_store`. The 64 chunks are `T.unroll(explicit=True)`. Pending device.
- `cp_mirror`, `compress/decode`, `fused_transform/norm_rope`, `schedule`. No spelling change. Constant shifts stay. `vbrc(vcadd)` stays. Rebench only.

## Near the bar

Rebench before any edit: `image/batch_downsample_pool`, `sampler/get_accept_result`, `store_kv/bf16`, `flash_mla/b256_k1024_full`. Identical bodies. If the median stays under 0.95, the note is launch/measurement, not a new instruction.
