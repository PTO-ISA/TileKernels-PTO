# ASC vs PTO on-device report

Kernels are the DeepInfer [`tilelang_ops/asc`](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc) set
(DeepInfer `a679bb8`), launched on all vector cores.
Toolchain: CANN 9.2.0, torch-npu 2.9.0.post6, ptoas 0.72, tilelang 0.1.15 (`pto-dev` `479d5cd1`).
`PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144`.
Device: Ascend950DT. `ratio = ASC_us / PTO_us`. A row is a perf miss when ratio < 0.95.
Timing is `do_bench(..., backend='msprof')` median over `BENCH_REP` repeats after 5 warmup launches.

## Shapes

Each row is one case from DeepInfer's `@pytest.mark.benchmark` lists
(DeepInfer `a679bb8`), not a serving trace and not every pytest combination.
The sizes that carry the comparison are DeepSeek-class prefill shapes:
RMSNorm is 32768 tokens by hidden 7168, the logit kernels use vocab 129280,
and gather uses 131072 rows.

Source lists:

- Norm, including `32768 × 7168`: [norm.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/norm.py#L472)
- KV cache benchmark length, 65536 tokens: [cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L388)
- Gather `(n, d, m)`: [gather.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/gather.py#L98)
- Candidate blocks, `max_model_len = 1 << 20`: [candidate_blocks.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py#L184)
- Fused transform, captured 550B Q cases: [fused_transform.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/fused_transform.py#L673)
- Sampling batch 256 or 4096 and vocab 129280 or 1024: [random_sampling.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L101)
- Sampler vocab 129280: [sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L634)
- Image embeds: [image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L240)
- Remaining kernels use the matching module under [tilelang_ops/](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops).

## Smaller than the DeepInfer list

Five rows use a smaller size than the published benchmark. The original size
still needs a run wherever the note is marked TODO.

- **KV load and store.** DeepInfer times every layout at 65536 tokens
  ([cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L388)).
  This report uses 8192 (bf16), 4096 (fp8 and fp4), and 1024 (mxfp8 index).
  A 65536-token cache fits under the 512 MiB tensor cap, so the cut is a harness choice.
  TODO: rerun each layout at 65536 tokens.
- **Candidate block-max.** The test allocates `max_model_len = 1 << 20`
  ([candidate_blocks.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py#L184)).
  At 256 rows that buffer is 512 MiB of bf16, which is the harness cap.
  The case sets the kernel width to the valid length, 65536, the smallest `valid_len` in that list.
  Tasks per row are `ceil(max_model_len / 16384)`, so this width is 1 task per row and `1 << 20` is 64.
  TODO: rerun `max_model_len = 1 << 20` at a row count that stays under 512 MiB (one row is 2 MiB).
- **Fused sample.** The sampling tests use vocab 129280 and 1024
  ([random_sampling.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L101)).
  Random sampling in this report has both. Fused sample has only 1024.
  The kernel accepts 129280: it requires a multiple of 64 and chunks a vocab of 8192 or more by 8192.
  TODO: add the vocab-129280 fused-sample row, including the reject specialization.
- **Fused transform.** The captured cases are 64 heads by 512 and 32 heads by 128, with up to 32768 tokens
  ([fused_transform.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/fused_transform.py#L673)).
  These rows are 8 heads by 128 at 4096 and 1024 tokens.
  The KV-cache store does not run: the kernel allows `store_cache` only when `num_heads == 1`, and these rows use 8 heads.
  TODO: run a captured head shape (64×512 or 32×128) and a `num_heads == 1` row that stores the KV cache.
- **Image pad and downsample.** `n_vit = 54` is the benchmark grid side, about 54×54 patches per image.
  The rgb benchmark builds 50 images at that grid
  ([image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L386)).
  Downsample is `(8, 24)`, `(32, 54)`, and `(8, 96)`
  ([image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L617)).
  This report uses 8 images at `n_vit = 54`.
  TODO: rgb at 50 images, and downsample at `(32, 54)` and `(8, 96)`.

Kernels that finish in a few microseconds are launch-bound.
Their ratio is not a saturated prefill measurement.

| family | case | correct | ASC us | PTO us | ratio | ASC GB/s | PTO GB/s | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| bitmask | b512_v129280_ftp | yes | 157.57 | 256.08 | 0.615 | 3413.16 | 2100.18 | perf A 64-word load plus vselr covers 32 logit vectors. The tail is still vload plus vbrc, and T.vmi.vload has no dist_mode. has_ftp, step 0, one row per request logits |
| block_tables | n256_q8_b4096 | yes | 11.39 | 10.75 | 1.060 | 3315.78 | 3513.27 | full max_num_blocks rows block_tables_out |
| candidate | block_max | yes | 16.56 | 52.29 | 0.317 | 2152.71 | 681.87 | perf Block max converts 128 bf16 lanes to f32, then one grouped vcmax. vcmax rejects bf16, and the ASC even/odd split is a part= convert. rows=256 valid_len=65536 block=16 block_scores,valid_blocks |
| candidate | compact_lens | yes | 7.82 | 7.93 | 0.986 | 67.14 | 66.22 | rows=128 topk_blocks=1024 block=16 compact_lens |
| candidate | restore_indices | yes | 2.42 | 3.42 | 0.707 | 461.43 | 326.31 | perf Restore gathers block ids with a full-mask vgather unrolled across the token block. ASC uses vgather2. rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1 compact_indices |
| compress | decode | yes | 8.87 | 10.08 | 0.879 | 8382.34 | 7369.44 | perf Both sides divide the softmax sum in float. The tile is unchanged. ratio=2 dim=512 seqs=64 nextn=6 kv_compressed,state_cache |
| compress | prefill | yes | 201.24 | 207.49 | 0.970 | 1545.60 | 1499.06 | ratio=2 dim=512 has_index=0 seqs=8 qlen=8192 kv_compressed,state_cache |
| cp_mirror | prepare_b4096_c1_2 | yes | 93.38 | 112.50 | 0.830 | 20.09 | 16.67 | perf PTO has no dynamic shift, so the binary search is straight-line constant shifts. TILE is unchanged. cp_size 8, cp_rank 7, ratios (1, 2) cu_lens,mirror_cp_reverse_index,mirror_cp_index,ks,ke |
| dflash | confidence_head | yes | 13.82 | 13.71 | 1.008 | 2430.02 | 2449.69 | N=4097 n_embed=4096 markov=0 out |
| dflash | input_store | yes | 3.74 | 5.67 | 0.660 | 1400.27 | 924.25 | perf 64 serial f32 iterations. Unrolling them was slower, and T.unroll(..., unroll_factor=) is rejected on Ascend. n=128 n_embed=4096 hc_mult=4 out |
| dflash_copy | src257_dst512 | yes | 3.26 | 2.98 | 1.097 | 21.97 | 24.10 | has_driver_inputs first_token_positions,last_valid_offset,golden_tokens,sample_positions,temperatures,top_ps,top_ks,seq_seeds,mtp_info_write_slots |
| dflash_input | n1024_blk5_swa512 | yes | 11.27 | 10.58 | 1.066 | 1077.00 | 1147.80 | valid_token_mask on, gate_force_random off (None) newx_slot_mapping,positions,input_ids,valid_token_mask,kv_index |
| flash_mla | b256_k1024_full | yes | 7.10 | 7.50 | 0.947 | 885.74 | 839.19 | perf Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement. in-place topk; full context; block_size 16 topk_idxs |
| fused_sample | b256_v1024 | no |  |  |  |  |  | ptoas rejects pto.vmi.extui unless both types are unsigned. The 0/1 prefix is built as uint32 in the source; task submission entered maintenance before that spelling was recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_sample | b256_v1024_reject | no |  |  |  |  |  | Same extui rejection as the non-reject row. The unsigned prefix spelling was not recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_transform | norm_fp8_h8_d128_n1024 | yes | 13.47 | 21.48 | 0.627 | 236.06 | 147.98 | perf The eight grouped maxima are parked with one 8-lane store and reloaded. The per-lane vcadd extract is gone. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 quant,sf |
| fused_transform | norm_rope_h8_d128_n4096 | yes | 21.97 | 24.45 | 0.899 | 812.03 | 729.74 | perf Same VF as ASC. No extra vcvt or vdiv turned up, so the tile is unchanged. bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 bf16 |
| gather | large_d128 | yes | 44.92 | 74.33 | 0.604 | 2999.59 | 1812.70 | perf The zero fill stays a 256-lane vbrc+vstore inside the VF. Copying one prefilled zero row was about 100x slower. float32 viewed as int8, D=512; some OOB indexes Y |
| gather | small_d4 | yes | 6.49 | 6.64 | 0.977 | 727.30 | 710.28 | float32 viewed as int8, D=16; some OOB indexes Y |
| gather_engram | n1024_nextn5 | yes | 8.47 | 8.88 | 0.953 | 70.10 | 66.83 | max_ngram_size 4, old_ratio 0.9, vocab 100000 out_ngram_token_ids |
| gather_ids | n1024_nextn7 | yes | 4.20 | 4.18 | 1.006 | 25.52 | 25.68 | old_ratio 0.9 out_input_ids |
| image | batch_downsample_concat | yes | 126.49 | 191.88 | 0.659 | 755.43 | 497.97 | perf The scatter index ramp is hoisted and the vector loop is unrolled. Both sides already scatter. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_pool | yes | 56.80 | 61.64 | 0.922 | 934.53 | 861.26 | perf Pool uses a whole-vector vcvt. The body matches ASC aside from part=, and the rebench stays under 0.95. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | replace_embeds | yes | 11.53 | 11.21 | 1.028 | 3457.74 | 3555.99 | tokens=4096 hidden=4096 input_embeds |
| image | rgb_pad | yes | 665.80 | 639.30 | 1.041 | 54.47 | 56.73 | patch=14 imgs=8 n_vit=54 padded=1 output |
| kv_index | paged_n1024_b128 | yes | 14.32 | 53.67 | 0.267 | 2361.65 | 630.24 | perf Paged gather is a full-mask vgather, which lowers to one vgather2_bc. The remaining gap is the serial 128-vector carry. index_length 8192, block_size 128, in-bounds tables kv_index |
| kv_index | swa_n1024 | yes | 5.31 | 5.26 | 1.008 | 100.35 | 101.15 | index_length 128, block_size 128, no block tables kv_index |
| kv_index_cat | q16384_w128_k1024 | yes | 43.58 | 63.56 | 0.686 | 3464.75 | 2375.79 | perf The two-vector add/select stays explicitly unrolled. A serial loop matching ASC was slower, so the tile is unchanged. largest _KV_INDEX_CAT_CASES row out |
| load_kv | bf16 | yes | 5.94 | 5.92 | 1.003 | 2823.32 | 2832.76 | block=64 tokens=8192 pattern=aligned layout=BF16 KV. kv |
| load_kv | mxfp8_index | yes | 2.35 | 2.21 | 1.060 | 115.38 | 122.31 | block=64 tokens=1024 pattern=aligned layout=V41, index kv. kv,kv_sf |
| load_kv | v41_fp4 | yes | 4.83 | 14.68 | 0.329 | 1113.32 | 366.15 | perf Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=4096 pattern=aligned layout=V41, FP4 main kv. kv |
| load_kv | v41_fp8 | yes | 5.90 | 13.75 | 0.429 | 1076.92 | 462.33 | perf Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=4096 pattern=aligned layout=V41, main/swa kv. kv |
| mtp | v1024_index_conf | yes | 8.54 | 8.68 | 0.984 | 987.12 | 971.76 | req 128, nextn 4, index and confidence, slot 0 skipped token_ids_pool,logprobs_pool,logprobs_index_pool,confidence_pool |
| mtp | v129280_no_index | yes | 133.19 | 132.92 | 1.002 | 3991.34 | 3999.59 | req 128, nextn 4, confidence on, unique slots, slot 0 skipped token_ids_pool,logprobs_pool,confidence_pool |
| norm | bf16 | yes | 271.31 | 923.15 | 0.294 | 3462.98 | 1017.75 | perf One row per VF. Sharing the tile's two rows through a square-sum spill was slower on this row. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=False fp8=False y |
| norm | bf16_residual | yes | 584.44 | 775.50 | 0.754 | 3215.16 | 2423.05 | perf The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False y,residual_out |
| norm | fp8_npc32_packed | no |  |  |  |  |  | Measured max abs 32 on y. The grouped amax is a 128-lane vcmax; an unmasked store writes past the 4 live lanes. A 4-of-128 mask is in the source and was not re-run. RuntimeError: norm/fp8_npc32_packed/y: max_abs 32.0 |
| sample | b256_v1024 | no |  |  |  |  |  | bisheng exits 139 (SIGSEGV) compiling the kernel, with the scheduler on or off. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sample | b256_v129280 | no |  |  |  |  |  | Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sampler | expand_sampling_params | yes | 3.91 | 4.10 | 0.953 | 10.65 | 10.15 | req 256, max_steps 7 temperatures_q,top_ps_q,top_ks_q,sample_positions_q,seq_seeds_q |
| sampler | finalize_verification | yes | 10.29 | 9.18 | 1.122 | 80.62 | 90.43 | req 256, max_steps 3, topk 64, confidence and ssp last_valid_offset,num_generated_tokens,num_verified_tokens,batch_confidence_error,emitted_stop_token,sampled_token_ids,sampled_token_logprobs,sampled_sum_smaller_probs,sampled_top_logprobs,sampled_top_token_ids,verified_token_ids,verified_token_logprobs,verified_sum_smaller_probs,verified_top_logprobs,verified_top_token_ids |
| sampler | get_accept_result | yes | 8.87 | 9.74 | 0.911 | 0.81 | 0.74 | perf Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement. total_q 256, has_golden, request length 16 token_ids,effective_accept,is_final |
| sampler | map_lookup | yes | 3.49 | 3.65 | 0.956 | 600.70 | 574.34 | by_key float32, rows 256, force_topk 1024 out |
| sampler | prepare_verification_metadata | yes | 2.97 | 2.66 | 1.115 | 4.32 | 4.81 | req 256, max_steps 7 row_to_req,local_step,is_verify |
| sampler | remap_draft_topk | yes | 10.21 | 24.79 | 0.412 | 411.01 | 169.39 | perf Remap gathers with a full-mask vgather over an unrolled 16-vector lookup. ASC uses vgather2. total_q 256, force_topk 1024, vocab 129280, pool 43136 aligned_logprobs,draft_local_ids,identity_rows |
| schedule | n1024_nextn7 | yes | 48.17 | 53.48 | 0.901 | 1.37 | 1.23 | perf The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input cum_rates,seqlens_q |
| seqlens | n1024_nextn3 | yes | 2.90 | 3.04 | 0.954 | 7.09 | 6.76 | half the rows remapped; in-place seqlens_k seqlens_k |
| state_cache | load_to_cache | yes | 4.55 | 4.61 | 0.986 | 267.01 | 263.29 | same shape as select_slice, opposite direction state_cache |
| state_cache | select_slice | yes | 3.65 | 3.21 | 1.137 | 332.93 | 378.39 | feature 584, cache 258, tensor 130, layers 8 state_tensor |
| store_kv | bf16 | yes | 98.94 | 106.55 | 0.929 | 169.90 | 157.76 | perf Same copy schedule. The rebench stays under 0.95 on a 100us kernel, so the residual is launch and measurement. block=64 tokens=8192 layout=BF16 KV. kv_cache |
| store_kv | v41_fp4 | yes | 10.03 | 28.12 | 0.357 | 537.35 | 191.66 | perf The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=4096 layout=V41, FP4 main kv. kv_cache |
| store_kv | v41_fp8 | yes | 9.59 | 20.96 | 0.457 | 664.85 | 304.01 | perf Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=4096 layout=V41, main/swa kv. kv_cache |
| sum_probs | b512_v129280 | yes | 87.02 | 85.53 | 1.017 | 3042.68 | 3095.59 | normalized logprobs, vocab 129280 output |
| swa | n8192_w4096 | yes | 34.91 | 83.39 | 0.419 | 3848.14 | 1611.01 | perf The four pivot broadcasts already sit outside the column loop. T.vmi.vload has no dist_mode, so each pivot stays vload plus vbrc. lens clipped to width output |
| v4_meta | n1024_q8 | yes | 17.72 | 17.77 | 0.997 | 31.03 | 30.95 | ca=sparse (1, 2), dense empty, block 128 positions,valid_token_mask,gate_force_random,fix_routing_mask,context_lens,klen,kv_ca_index_offset_len,kv_swa_index_offset_len,slot_mapping_store_swa,query_indices,state_block_idx_out,positions_c,slot_mapping_store_ca |
| yoco | n256_len500 | yes | 25.15 | 22.19 | 1.133 | 25.77 | 29.21 | cp_size 4, cp_rank 0, keep=seqlen cu_lens,yoco_contrib_index,yoco_gathered_reverse_index |

Rows: 56. Correct: 51. Perf misses (ratio < 0.95): 25.

## Perf misses

- `bitmask/b512_v129280_ftp` ratio 0.615. A 64-word load plus vselr covers 32 logit vectors. The tail is still vload plus vbrc, and T.vmi.vload has no dist_mode. has_ftp, step 0, one row per request
- `candidate/block_max` ratio 0.317. Block max converts 128 bf16 lanes to f32, then one grouped vcmax. vcmax rejects bf16, and the ASC even/odd split is a part= convert. rows=256 valid_len=65536 block=16
- `candidate/restore_indices` ratio 0.707. Restore gathers block ids with a full-mask vgather unrolled across the token block. ASC uses vgather2. rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1
- `compress/decode` ratio 0.879. Both sides divide the softmax sum in float. The tile is unchanged. ratio=2 dim=512 seqs=64 nextn=6
- `cp_mirror/prepare_b4096_c1_2` ratio 0.830. PTO has no dynamic shift, so the binary search is straight-line constant shifts. TILE is unchanged. cp_size 8, cp_rank 7, ratios (1, 2)
- `dflash/input_store` ratio 0.660. 64 serial f32 iterations. Unrolling them was slower, and T.unroll(..., unroll_factor=) is rejected on Ascend. n=128 n_embed=4096 hc_mult=4
- `flash_mla/b256_k1024_full` ratio 0.947. Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement. in-place topk; full context; block_size 16
- `fused_transform/norm_fp8_h8_d128_n1024` ratio 0.627. The eight grouped maxima are parked with one 8-lane store and reloaded. The per-lane vcadd extract is gone. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `fused_transform/norm_rope_h8_d128_n4096` ratio 0.899. Same VF as ASC. No extra vcvt or vdiv turned up, so the tile is unchanged. bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `gather/large_d128` ratio 0.604. The zero fill stays a 256-lane vbrc+vstore inside the VF. Copying one prefilled zero row was about 100x slower. float32 viewed as int8, D=512; some OOB indexes
- `image/batch_downsample_concat` ratio 0.659. The scatter index ramp is hoisted and the vector loop is unrolled. Both sides already scatter. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=0
- `image/batch_downsample_pool` ratio 0.922. Pool uses a whole-vector vcvt. The body matches ASC aside from part=, and the rebench stays under 0.95. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=1
- `kv_index/paged_n1024_b128` ratio 0.267. Paged gather is a full-mask vgather, which lowers to one vgather2_bc. The remaining gap is the serial 128-vector carry. index_length 8192, block_size 128, in-bounds tables
- `kv_index_cat/q16384_w128_k1024` ratio 0.686. The two-vector add/select stays explicitly unrolled. A serial loop matching ASC was slower, so the tile is unchanged. largest _KV_INDEX_CAT_CASES row
- `load_kv/v41_fp4` ratio 0.329. Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=4096 pattern=aligned layout=V41, FP4 main kv.
- `load_kv/v41_fp8` ratio 0.429. Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=4096 pattern=aligned layout=V41, main/swa kv.
- `norm/bf16` ratio 0.294. One row per VF. Sharing the tile's two rows through a square-sum spill was slower on this row. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=False fp8=False
- `norm/bf16_residual` ratio 0.754. The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False
- `sampler/get_accept_result` ratio 0.911. Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement. total_q 256, has_golden, request length 16
- `sampler/remap_draft_topk` ratio 0.412. Remap gathers with a full-mask vgather over an unrolled 16-vector lookup. ASC uses vgather2. total_q 256, force_topk 1024, vocab 129280, pool 43136
- `schedule/n1024_nextn7` ratio 0.901. The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input
- `store_kv/bf16` ratio 0.929. Same copy schedule. The rebench stays under 0.95 on a 100us kernel, so the residual is launch and measurement. block=64 tokens=8192 layout=BF16 KV.
- `store_kv/v41_fp4` ratio 0.357. The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=4096 layout=V41, FP4 main kv.
- `store_kv/v41_fp8` ratio 0.457. Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=4096 layout=V41, main/swa kv.
- `swa/n8192_w4096` ratio 0.419. The four pivot broadcasts already sit outside the column loop. T.vmi.vload has no dist_mode, so each pivot stays vload plus vbrc. lens clipped to width

## Failures

- `fused_sample/b256_v1024`: ptoas rejects pto.vmi.extui unless both types are unsigned. The 0/1 prefix is built as uint32 in the source; task submission entered maintenance before that spelling was recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `fused_sample/b256_v1024_reject`: Same extui rejection as the non-reject row. The unsigned prefix spelling was not recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `norm/fp8_npc32_packed`: Measured max abs 32 on y. The grouped amax is a 128-lane vcmax; an unmasked store writes past the 4 live lanes. A 4-of-128 mask is in the source and was not re-run. RuntimeError: norm/fp8_npc32_packed/y: max_abs 32.0
- `sample/b256_v1024`: bisheng exits 139 (SIGSEGV) compiling the kernel, with the scheduler on or off. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
- `sample/b256_v129280`: Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
