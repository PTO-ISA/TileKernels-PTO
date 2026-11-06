# ASC vs PTO on-device report

Worktree `vllm-kernels-ondevice` from TileKernels-PTO `vmi-impl`.
Kernels are the DeepInfer `tilelang_ops/asc` set in package `deepinfer_kernels`, launched on all vector cores.
This report is `deepinfer_kernels/REPORT.md`.
Toolchain: CANN 9.2.0 (`/usr/local/Ascend/canns/9.2.0/cann-9.2.0/set_env.sh`),
conda `cann91_dev` (`torch-npu==2.9.0.post6`), ptoas 0.72, tilelang `0.1.15+cpu.git479d5cd1`
(`pto-dev` checkout `479d5cd1`). `PTO_FLAGS=--pto-backend=vpto --pto-level=level3`
`--vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144`.
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
| bitmask | b512_v129280_ftp | yes | 158.43 | 322.31 | 0.492 | 3394.70 | 1668.60 | perf PTO broadcasts the mask with vload+vbrc and walks the left bits serially; ASC uses BRC_B32. has_ftp, step 0, one row per request logits |
| block_tables | n256_q8_b4096 | yes | 11.39 | 10.75 | 1.060 | 3315.78 | 3513.27 | full max_num_blocks rows block_tables_out |
| candidate | block_max | yes | 16.56 | 52.29 | 0.317 | 2152.71 | 681.87 | perf Block max is a grouped vcmax over bf16 tiles. ASC reduces the same tile with packed vmax. rows=256 valid_len=65536 block=16 block_scores,valid_blocks |
| candidate | compact_lens | yes | 7.82 | 7.93 | 0.986 | 67.14 | 66.22 | rows=128 topk_blocks=1024 block=16 compact_lens |
| candidate | restore_indices | yes | 2.34 | 3.48 | 0.674 | 475.68 | 320.64 | perf Restore gathers block ids with vgather; ASC uses vgather2. rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1 compact_indices |
| compress | decode | yes | 8.87 | 10.08 | 0.879 | 8382.34 | 7369.44 | perf Softmax sum is a float vdiv. ASC divides integers. The tile is unchanged. ratio=2 dim=512 seqs=64 nextn=6 kv_compressed,state_cache |
| compress | prefill | yes | 201.24 | 207.49 | 0.970 | 1545.60 | 1499.06 | ratio=2 dim=512 has_index=0 seqs=8 qlen=8192 kv_compressed,state_cache |
| cp_mirror | prepare_b4096_c1_2 | yes | 93.38 | 112.50 | 0.830 | 20.09 | 16.67 | perf PTO has no dynamic shift, so the binary search is straight-line constant shifts. TILE is unchanged. cp_size 8, cp_rank 7, ratios (1, 2) cu_lens,mirror_cp_reverse_index,mirror_cp_index,ks,ke |
| dflash | confidence_head | yes | 13.82 | 13.71 | 1.008 | 2430.02 | 2449.69 | N=4097 n_embed=4096 markov=0 out |
| dflash | input_store | yes | 3.74 | 5.67 | 0.660 | 1400.27 | 924.25 | perf 64 serial f32 iterations. T.unroll(..., unroll_factor=) is rejected on Ascend. n=128 n_embed=4096 hc_mult=4 out |
| dflash_copy | src257_dst512 | yes | 3.26 | 2.98 | 1.097 | 21.97 | 24.10 | has_driver_inputs first_token_positions,last_valid_offset,golden_tokens,sample_positions,temperatures,top_ps,top_ks,seq_seeds,mtp_info_write_slots |
| dflash_input | n1024_blk5_swa512 | yes | 11.27 | 10.58 | 1.066 | 1077.00 | 1147.80 | valid_token_mask on, gate_force_random off (None) newx_slot_mapping,positions,input_ids,valid_token_mask,kv_index |
| flash_mla | b256_k1024_full | yes | 7.02 | 7.46 | 0.941 | 895.89 | 843.17 | perf Same SIMT body. The two medians are a few tenths of a microsecond apart on a 7us kernel. in-place topk; full context; block_size 16 topk_idxs |
| fused_sample | b256_v1024 | no |  |  |  |  |  | ptoas rejects pto.vmi.extui unless both types are unsigned. The 0/1 prefix is built as uint32 in the source; task submission entered maintenance before that spelling was recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_sample | b256_v1024_reject | no |  |  |  |  |  | Same extui rejection as the non-reject row. The unsigned prefix spelling was not recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_transform | norm_fp8_h8_d128_n1024 | yes | 13.62 | 28.31 | 0.481 | 233.38 | 112.29 | perf FP8 scale packing is scalar vcvt/vselr per head. The rope-only row is the same schedule without that cast. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 quant,sf |
| fused_transform | norm_rope_h8_d128_n4096 | yes | 21.97 | 24.45 | 0.899 | 812.03 | 729.74 | perf Same multi-core schedule. The extra time is the vmi lowering of the norm and rope VFs. bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 bf16 |
| gather | large_d128 | yes | 44.92 | 74.33 | 0.604 | 2999.59 | 1812.70 | perf PTO rejects T.Parallel inside SimdVF, so the zero fill is a 256-lane vbrc+vstore. float32 viewed as int8, D=512; some OOB indexes Y |
| gather | small_d4 | yes | 6.49 | 6.64 | 0.977 | 727.30 | 710.28 | float32 viewed as int8, D=16; some OOB indexes Y |
| gather_engram | n1024_nextn5 | yes | 8.47 | 8.88 | 0.953 | 70.10 | 66.83 | max_ngram_size 4, old_ratio 0.9, vocab 100000 out_ngram_token_ids |
| gather_ids | n1024_nextn7 | yes | 4.20 | 4.18 | 1.006 | 25.52 | 25.68 | old_ratio 0.9 out_input_ids |
| image | batch_downsample_concat | yes | 126.62 | 194.05 | 0.653 | 754.63 | 492.42 | perf Concat is a per-element VF. ASC moves the same tile with copies. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_pool | yes | 56.60 | 61.55 | 0.920 | 937.90 | 862.46 | perf Pool reduction stays in the VF. ASC is closer to the copy bandwidth of the same tile. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | replace_embeds | yes | 11.53 | 11.21 | 1.028 | 3457.74 | 3555.99 | tokens=4096 hidden=4096 input_embeds |
| image | rgb_pad | yes | 665.80 | 639.30 | 1.041 | 54.47 | 56.73 | patch=14 imgs=8 n_vit=54 padded=1 output |
| kv_index | paged_n1024_b128 | yes | 14.13 | 55.78 | 0.253 | 2393.59 | 606.40 | perf Paged gather is vgather. ASC uses vgather2. The SWA path, which is copies, matches. index_length 8192, block_size 128, in-bounds tables kv_index |
| kv_index | swa_n1024 | yes | 5.31 | 5.26 | 1.008 | 100.35 | 101.15 | index_length 128, block_size 128, no block tables kv_index |
| kv_index_cat | q16384_w128_k1024 | yes | 43.58 | 63.56 | 0.686 | 3464.75 | 2375.79 | perf Explicitly unrolled VF on a bandwidth-bound index cat. The tile is unchanged. largest _KV_INDEX_CAT_CASES row out |
| load_kv | bf16 | yes | 5.94 | 5.92 | 1.003 | 2823.32 | 2832.76 | block=64 tokens=8192 pattern=aligned layout=BF16 KV. kv |
| load_kv | mxfp8_index | yes | 2.35 | 2.21 | 1.060 | 115.38 | 122.31 | block=64 tokens=1024 pattern=aligned layout=V41, index kv. kv,kv_sf |
| load_kv | v41_fp4 | yes | 4.83 | 14.68 | 0.329 | 1113.32 | 366.15 | perf Dequant is a 64-lane vselr table plus a grouped scale broadcast. ASC uses packed SIMD. block=64 tokens=4096 pattern=aligned layout=V41, FP4 main kv. kv |
| load_kv | v41_fp8 | yes | 5.90 | 13.75 | 0.429 | 1076.92 | 462.33 | perf Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=4096 pattern=aligned layout=V41, main/swa kv. kv |
| mtp | v1024_index_conf | yes | 8.54 | 8.68 | 0.984 | 987.12 | 971.76 | req 128, nextn 4, index and confidence, slot 0 skipped token_ids_pool,logprobs_pool,logprobs_index_pool,confidence_pool |
| mtp | v129280_no_index | yes | 133.19 | 132.92 | 1.002 | 3991.34 | 3999.59 | req 128, nextn 4, confidence on, unique slots, slot 0 skipped token_ids_pool,logprobs_pool,confidence_pool |
| norm | bf16 | yes | 271.31 | 923.15 | 0.294 | 3462.98 | 1017.75 | perf One row per VF. Multi-row plus the compensated rstd exceeds the layout search. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=False fp8=False y |
| norm | bf16_residual | yes | 590.18 | 1426.38 | 0.414 | 3183.88 | 1317.36 | perf One row per VF, same layout-search limit as the plain bf16 row. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=True fp8=False y,residual_out |
| norm | fp8_npc32_packed | no |  |  |  |  |  | Measured max abs 32 on y. The grouped amax is a 128-lane vcmax; an unmasked store writes past the 4 live lanes. A 4-of-128 mask is in the source and was not re-run. RuntimeError: norm/fp8_npc32_packed/y: max_abs 32.0 |
| sample | b256_v1024 | no |  |  |  |  |  | bisheng exits 139 (SIGSEGV) compiling the kernel, with the scheduler on or off. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sample | b256_v129280 | no |  |  |  |  |  | Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sampler | expand_sampling_params | yes | 3.91 | 4.10 | 0.953 | 10.65 | 10.15 | req 256, max_steps 7 temperatures_q,top_ps_q,top_ks_q,sample_positions_q,seq_seeds_q |
| sampler | finalize_verification | yes | 10.29 | 9.18 | 1.122 | 80.62 | 90.43 | req 256, max_steps 3, topk 64, confidence and ssp last_valid_offset,num_generated_tokens,num_verified_tokens,batch_confidence_error,emitted_stop_token,sampled_token_ids,sampled_token_logprobs,sampled_sum_smaller_probs,sampled_top_logprobs,sampled_top_token_ids,verified_token_ids,verified_token_logprobs,verified_sum_smaller_probs,verified_top_logprobs,verified_top_token_ids |
| sampler | get_accept_result | yes | 9.03 | 9.75 | 0.926 | 0.80 | 0.74 | perf Same SIMT accept scan. The medians differ by under a microsecond. total_q 256, has_golden, request length 16 token_ids,effective_accept,is_final |
| sampler | map_lookup | yes | 3.49 | 3.65 | 0.956 | 600.70 | 574.34 | by_key float32, rows 256, force_topk 1024 out |
| sampler | prepare_verification_metadata | yes | 2.97 | 2.66 | 1.115 | 4.32 | 4.81 | req 256, max_steps 7 row_to_req,local_step,is_verify |
| sampler | remap_draft_topk | yes | 10.30 | 25.60 | 0.402 | 407.77 | 164.00 | perf Remap gathers with vgather. ASC uses vgather2. total_q 256, force_topk 1024, vocab 129280, pool 43136 aligned_logprobs,draft_local_ids,identity_rows |
| schedule | n1024_nextn7 | yes | 48.17 | 53.48 | 0.901 | 1.37 | 1.23 | perf The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input cum_rates,seqlens_q |
| seqlens | n1024_nextn3 | yes | 2.90 | 3.04 | 0.954 | 7.09 | 6.76 | half the rows remapped; in-place seqlens_k seqlens_k |
| state_cache | load_to_cache | yes | 4.55 | 4.61 | 0.986 | 267.01 | 263.29 | same shape as select_slice, opposite direction state_cache |
| state_cache | select_slice | yes | 3.65 | 3.21 | 1.137 | 332.93 | 378.39 | feature 584, cache 258, tensor 130, layers 8 state_tensor |
| store_kv | bf16 | yes | 99.49 | 107.01 | 0.930 | 168.96 | 157.09 | perf Same copy schedule. The medians differ by about 7us on a 100us kernel. block=64 tokens=8192 layout=BF16 KV. kv_cache |
| store_kv | v41_fp4 | yes | 10.05 | 32.89 | 0.305 | 536.59 | 163.88 | perf Quant spills the inverse, widens the LUT to 64 lanes for vselr, then stores with a narrow mask. block=64 tokens=4096 layout=V41, FP4 main kv. kv_cache |
| store_kv | v41_fp8 | yes | 9.59 | 20.96 | 0.457 | 664.85 | 304.01 | perf Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=4096 layout=V41, main/swa kv. kv_cache |
| sum_probs | b512_v129280 | yes | 87.02 | 85.53 | 1.017 | 3042.68 | 3095.59 | normalized logprobs, vocab 129280 output |
| swa | n8192_w4096 | yes | 34.87 | 83.31 | 0.419 | 3852.38 | 1612.56 | perf ASC is at memory bandwidth. The PTO index math stays in the VF, so the same tile is slower. lens clipped to width output |
| v4_meta | n1024_q8 | yes | 17.72 | 17.77 | 0.997 | 31.03 | 30.95 | ca=sparse (1, 2), dense empty, block 128 positions,valid_token_mask,gate_force_random,fix_routing_mask,context_lens,klen,kv_ca_index_offset_len,kv_swa_index_offset_len,slot_mapping_store_swa,query_indices,state_block_idx_out,positions_c,slot_mapping_store_ca |
| yoco | n256_len500 | yes | 25.15 | 22.19 | 1.133 | 25.77 | 29.21 | cp_size 4, cp_rank 0, keep=seqlen cu_lens,yoco_contrib_index,yoco_gathered_reverse_index |

Rows: 56. Correct: 51. Perf misses (ratio < 0.95): 25.

## Perf misses

- `bitmask/b512_v129280_ftp` ratio 0.492. PTO broadcasts the mask with vload+vbrc and walks the left bits serially; ASC uses BRC_B32. has_ftp, step 0, one row per request
- `candidate/block_max` ratio 0.317. Block max is a grouped vcmax over bf16 tiles. ASC reduces the same tile with packed vmax. rows=256 valid_len=65536 block=16
- `candidate/restore_indices` ratio 0.674. Restore gathers block ids with vgather; ASC uses vgather2. rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1
- `compress/decode` ratio 0.879. Softmax sum is a float vdiv. ASC divides integers. The tile is unchanged. ratio=2 dim=512 seqs=64 nextn=6
- `cp_mirror/prepare_b4096_c1_2` ratio 0.830. PTO has no dynamic shift, so the binary search is straight-line constant shifts. TILE is unchanged. cp_size 8, cp_rank 7, ratios (1, 2)
- `dflash/input_store` ratio 0.660. 64 serial f32 iterations. T.unroll(..., unroll_factor=) is rejected on Ascend. n=128 n_embed=4096 hc_mult=4
- `flash_mla/b256_k1024_full` ratio 0.941. Same SIMT body. The two medians are a few tenths of a microsecond apart on a 7us kernel. in-place topk; full context; block_size 16
- `fused_transform/norm_fp8_h8_d128_n1024` ratio 0.481. FP8 scale packing is scalar vcvt/vselr per head. The rope-only row is the same schedule without that cast. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `fused_transform/norm_rope_h8_d128_n4096` ratio 0.899. Same multi-core schedule. The extra time is the vmi lowering of the norm and rope VFs. bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `gather/large_d128` ratio 0.604. PTO rejects T.Parallel inside SimdVF, so the zero fill is a 256-lane vbrc+vstore. float32 viewed as int8, D=512; some OOB indexes
- `image/batch_downsample_concat` ratio 0.653. Concat is a per-element VF. ASC moves the same tile with copies. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=0
- `image/batch_downsample_pool` ratio 0.920. Pool reduction stays in the VF. ASC is closer to the copy bandwidth of the same tile. imgs=8 n_vit=54 dim=1024 ratio=3 downsample=1
- `kv_index/paged_n1024_b128` ratio 0.253. Paged gather is vgather. ASC uses vgather2. The SWA path, which is copies, matches. index_length 8192, block_size 128, in-bounds tables
- `kv_index_cat/q16384_w128_k1024` ratio 0.686. Explicitly unrolled VF on a bandwidth-bound index cat. The tile is unchanged. largest _KV_INDEX_CAT_CASES row
- `load_kv/v41_fp4` ratio 0.329. Dequant is a 64-lane vselr table plus a grouped scale broadcast. ASC uses packed SIMD. block=64 tokens=4096 pattern=aligned layout=V41, FP4 main kv.
- `load_kv/v41_fp8` ratio 0.429. Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=4096 pattern=aligned layout=V41, main/swa kv.
- `norm/bf16` ratio 0.294. One row per VF. Multi-row plus the compensated rstd exceeds the layout search. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=False fp8=False
- `norm/bf16_residual` ratio 0.414. One row per VF, same layout-search limit as the plain bf16 row. Hidden 7168 is unchanged. tokens=32768 hidden=7168 residual=True fp8=False
- `sampler/get_accept_result` ratio 0.926. Same SIMT accept scan. The medians differ by under a microsecond. total_q 256, has_golden, request length 16
- `sampler/remap_draft_topk` ratio 0.402. Remap gathers with vgather. ASC uses vgather2. total_q 256, force_topk 1024, vocab 129280, pool 43136
- `schedule/n1024_nextn7` ratio 0.901. The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input
- `store_kv/bf16` ratio 0.930. Same copy schedule. The medians differ by about 7us on a 100us kernel. block=64 tokens=8192 layout=BF16 KV.
- `store_kv/v41_fp4` ratio 0.305. Quant spills the inverse, widens the LUT to 64 lanes for vselr, then stores with a narrow mask. block=64 tokens=4096 layout=V41, FP4 main kv.
- `store_kv/v41_fp8` ratio 0.457. Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=4096 layout=V41, main/swa kv.
- `swa/n8192_w4096` ratio 0.419. ASC is at memory bandwidth. The PTO index math stays in the VF, so the same tile is slower. lens clipped to width

## Failures

- `fused_sample/b256_v1024`: ptoas rejects pto.vmi.extui unless both types are unsigned. The 0/1 prefix is built as uint32 in the source; task submission entered maintenance before that spelling was recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `fused_sample/b256_v1024_reject`: Same extui rejection as the non-reject row. The unsigned prefix spelling was not recompiled. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `norm/fp8_npc32_packed`: Measured max abs 32 on y. The grouped amax is a 128-lane vcmax; an unmasked store writes past the 4 live lanes. A 4-of-128 mask is in the source and was not re-run. RuntimeError: norm/fp8_npc32_packed/y: max_abs 32.0
- `sample/b256_v1024`: bisheng exits 139 (SIGSEGV) compiling the kernel, with the scheduler on or off. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
- `sample/b256_v129280`: Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
