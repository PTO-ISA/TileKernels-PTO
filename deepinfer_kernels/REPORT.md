# ASC vs PTO on-device report

Kernels are the DeepInfer [`tilelang_ops/asc`](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc) set
(DeepInfer `a679bb8`), launched on all vector cores.
Toolchain: CANN 9.2.0, torch-npu 2.9.0.post6, ptoas 0.76, tilelang 0.1.15 (`pto-dev` `479d5cd1`).
`PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.
Device: Ascend950DT. `ratio = ASC_us / PTO_us`. A row is a perf miss when ratio < 0.95.
Timing is `do_bench(..., backend='msprof')` median over `BENCH_REP` repeats after 5 warmup launches.

## Shapes

Each row is one case from DeepInfer's `@pytest.mark.benchmark` lists
(DeepInfer `a679bb8`), not a serving trace and not every pytest combination.
The sizes that carry the comparison are DeepSeek-class prefill shapes:
RMSNorm is 32768 tokens by hidden 7168, the logit kernels use vocab 129280,
gather uses 131072 rows, KV load and store run every layout at 65536 tokens,
and candidate block-max scans the full `1 << 20` model length at 256 rows.

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

Kernels that finish in a few microseconds are launch-bound.
Their ratio is not a saturated prefill measurement.

| family | case | correct | ASC us | PTO us | ratio | ASC GB/s | PTO GB/s | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| bitmask | b512_v129280_ftp | yes | 158.11 | 163.12 | 0.969 | 3401.57 | 3297.07 | has_ftp, step 0, one row per request logits |
| block_tables | n256_q8_b4096 | yes | 10.95 | 11.35 | 0.965 | 3447.67 | 3326.24 | full max_num_blocks rows block_tables_out |
| candidate | block_max | yes | 180.69 | 180.65 | 1.000 | 3156.96 | 3157.58 | rows=256 valid_len=1048576 max_model_len=1048576 block=16 block_scores,valid_blocks |
| candidate | compact_lens | yes | 7.94 | 7.86 | 1.010 | 66.17 | 66.85 | rows=128 topk_blocks=1024 block=16 compact_lens |
| candidate | restore_indices | yes | 2.40 | 2.45 | 0.979 | 464.17 | 454.51 | rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1 compact_indices |
| compress | decode | yes | 8.81 | 9.31 | 0.946 | 8439.69 | 7986.14 | perf Both sides divide the softmax sum in float. The tile is unchanged; the residual on this 9 us row is launch and measurement. ratio=2 dim=512 seqs=64 nextn=6 kv_compressed,state_cache |
| compress | prefill | yes | 200.95 | 207.34 | 0.969 | 1547.80 | 1500.09 | ratio=2 dim=512 has_index=0 seqs=8 qlen=8192 kv_compressed,state_cache |
| cp_mirror | prepare_b4096_c1_2 | yes | 93.27 | 112.59 | 0.828 | 20.11 | 16.66 | perf PTO has no dynamic shift, so the binary search is straight-line constant shifts. cp_size 8, cp_rank 7, ratios (1, 2) cu_lens,mirror_cp_reverse_index,mirror_cp_index,ks,ke |
| dflash | confidence_head | yes | 13.54 | 13.08 | 1.036 | 2479.89 | 2568.69 | N=4097 n_embed=4096 markov=0 out |
| dflash | input_store | yes | 3.82 | 3.38 | 1.130 | 1373.18 | 1551.91 | n=128 n_embed=4096 hc_mult=4 out |
| dflash_copy | src257_dst512 | yes | 3.37 | 3.06 | 1.101 | 21.28 | 23.43 | has_driver_inputs first_token_positions,last_valid_offset,golden_tokens,sample_positions,temperatures,top_ps,top_ks,seq_seeds,mtp_info_write_slots |
| dflash_input | n1024_blk5_swa512 | yes | 11.46 | 10.33 | 1.110 | 1059.36 | 1175.83 | valid_token_mask on, gate_force_random off (None) newx_slot_mapping,positions,input_ids,valid_token_mask,kv_index |
| flash_mla | b256_k1024_full | yes | 7.00 | 7.40 | 0.945 | 898.89 | 849.82 | perf Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement. in-place topk; full context; block_size 16 topk_idxs |
| fused_sample | b256_v1024 | no |  |  |  |  |  | ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_sample | b256_v1024_reject | no |  |  |  |  |  | Same extui rejection as the non-reject row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_sample | b256_v129280 | no |  |  |  |  |  | Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_sample | b256_v129280_reject | no |  |  |  |  |  | Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel']. |
| fused_transform | norm_fp8_h8_d128_n1024 | yes | 13.66 | 16.03 | 0.852 | 232.63 | 198.29 | perf The eight grouped maxima are parked with one 8-lane store and reloaded. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 quant,sf |
| fused_transform | norm_rope_h8_d128_n4096 | yes | 22.13 | 22.25 | 0.994 | 806.43 | 801.99 | bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 bf16 |
| fused_transform | rope_h64_d512_n32768 | yes | 1325.96 | 1329.80 | 0.997 | 3240.03 | 3230.67 | captured 550B Q: rope only, rotary 64 offset 448, num_heads=64 bf16 |
| fused_transform | store_mxfp8_h1_d128_n32768 | yes | 32.51 | 43.35 | 0.750 | 391.14 | 293.28 | perf The store_cache arm writes each token through the paged slot_mapping; the quantized store into the MXFP8 index cache is 11 us behind on a 33 us row. norm + e4m3 ue8m0 group 32 into MXFP8 index cache, num_heads=1 block 128, contiguous slots, slot 0 invalid kv_cache |
| gather | large_d128 | yes | 44.48 | 48.81 | 0.911 | 3029.20 | 2760.67 | perf The residual on this 45 us row is launch and measurement. float32 viewed as int8, D=512; some OOB indexes Y |
| gather | small_d4 | yes | 6.51 | 6.67 | 0.975 | 725.32 | 707.24 | float32 viewed as int8, D=16; some OOB indexes Y |
| gather_engram | n1024_nextn5 | yes | 8.38 | 8.94 | 0.937 | 70.84 | 66.40 | perf Same gather body; the residual on this 9 us row is launch and measurement. max_ngram_size 4, old_ratio 0.9, vocab 100000 out_ngram_token_ids |
| gather_ids | n1024_nextn7 | yes | 4.18 | 4.10 | 1.020 | 25.66 | 26.16 | old_ratio 0.9 out_input_ids |
| image | batch_downsample_concat_i32_v54 | yes | 497.04 | 495.92 | 1.002 | 768.96 | 770.70 | imgs=32 n_vit=54 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_concat_i8_v24 | yes | 29.54 | 29.37 | 1.006 | 639.00 | 642.66 | imgs=8 n_vit=24 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_concat_i8_v96 | yes | 391.13 | 390.91 | 1.001 | 772.10 | 772.53 | imgs=8 n_vit=96 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_pool_i32_v54 | yes | 220.46 | 226.62 | 0.973 | 963.17 | 936.95 | imgs=32 n_vit=54 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | batch_downsample_pool_i8_v24 | yes | 14.24 | 14.32 | 0.994 | 736.26 | 732.17 | imgs=8 n_vit=24 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | batch_downsample_pool_i8_v96 | yes | 174.83 | 179.70 | 0.973 | 959.64 | 933.61 | imgs=8 n_vit=96 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | replace_embeds | yes | 11.59 | 10.92 | 1.061 | 3440.40 | 3650.09 | tokens=4096 hidden=4096 input_embeds |
| image | rgb_pad | yes | 673.06 | 647.25 | 1.040 | 332.25 | 345.49 | patch=14 imgs=50 n_vit=54 padded=1 output |
| kv_index | paged_n1024_b128 | yes | 14.18 | 14.67 | 0.967 | 2385.48 | 2306.15 | index_length 8192, block_size 128, in-bounds tables kv_index |
| kv_index | swa_n1024 | yes | 5.44 | 5.38 | 1.012 | 97.92 | 99.06 | index_length 128, block_size 128, no block tables kv_index |
| kv_index_cat | q16384_w128_k1024 | yes | 43.45 | 43.52 | 0.998 | 3475.40 | 3469.71 | largest _KV_INDEX_CAT_CASES row out |
| load_kv | bf16 | yes | 39.94 | 39.81 | 1.003 | 3360.99 | 3371.75 | block=64 tokens=65536 pattern=aligned layout=BF16 KV. kv |
| load_kv | mxfp8_index | yes | 9.35 | 9.43 | 0.992 | 1852.16 | 1836.93 | block=64 tokens=65536 pattern=aligned layout=V41, index kv. kv,kv_sf |
| load_kv | v41_fp4 | yes | 30.20 | 113.26 | 0.267 | 2847.27 | 759.30 | perf Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=65536 pattern=aligned layout=V41, FP4 main kv. kv |
| load_kv | v41_fp8 | yes | 40.72 | 61.95 | 0.657 | 2498.29 | 1642.04 | perf Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=65536 pattern=aligned layout=V41, main/swa kv. kv |
| mtp | v1024_index_conf | yes | 8.73 | 8.61 | 1.015 | 965.17 | 979.66 | req 128, nextn 4, index and confidence, slot 0 skipped token_ids_pool,logprobs_pool,logprobs_index_pool,confidence_pool |
| mtp | v129280_no_index | yes | 133.32 | 132.96 | 1.003 | 3987.40 | 3998.11 | req 128, nextn 4, confidence on, unique slots, slot 0 skipped token_ids_pool,logprobs_pool,confidence_pool |
| norm | bf16 | yes | 271.66 | 273.92 | 0.992 | 3458.45 | 3429.94 | tokens=32768 hidden=7168 residual=False fp8=False y |
| norm | bf16_residual | yes | 586.90 | 778.54 | 0.754 | 3201.69 | 2413.56 | perf The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False y,residual_out |
| norm | fp8_npc32_packed | no |  |  |  |  |  | The 4-of-128 amax mask spelling fails PTO codegen legality: store destination misaligned, byte remainder 16 vs required 32. InternalError: [VerifyPTOCodegenLegality] kernel=_norm_kernel, store=#4: destination is misaligned: byte remainder=T.int64(16), required alignment=32 |
| sample | b256_v1024 | no |  |  |  |  |  | bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off. Not a membar interaction. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sample | b256_v129280 | no |  |  |  |  |  | Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel']. |
| sampler | expand_sampling_params | yes | 4.13 | 4.17 | 0.991 | 10.07 | 9.98 | req 256, max_steps 7 temperatures_q,top_ps_q,top_ks_q,sample_positions_q,seq_seeds_q |
| sampler | finalize_verification | yes | 10.26 | 9.22 | 1.112 | 80.91 | 89.98 | req 256, max_steps 3, topk 64, confidence and ssp last_valid_offset,num_generated_tokens,num_verified_tokens,batch_confidence_error,emitted_stop_token,sampled_token_ids,sampled_token_logprobs,sampled_sum_smaller_probs,sampled_top_logprobs,sampled_top_token_ids,verified_token_ids,verified_token_logprobs,verified_sum_smaller_probs,verified_top_logprobs,verified_top_token_ids |
| sampler | get_accept_result | yes | 9.01 | 9.77 | 0.922 | 0.80 | 0.74 | perf Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement. total_q 256, has_golden, request length 16 token_ids,effective_accept,is_final |
| sampler | map_lookup | yes | 3.27 | 3.50 | 0.935 | 641.83 | 600.01 | perf 3.5 us kernel, so the residual is launch and measurement. by_key float32, rows 256, force_topk 1024 out |
| sampler | prepare_verification_metadata | yes | 2.61 | 2.55 | 1.025 | 4.90 | 5.02 | req 256, max_steps 7 row_to_req,local_step,is_verify |
| sampler | remap_draft_topk | yes | 10.62 | 10.23 | 1.038 | 395.43 | 410.39 | total_q 256, force_topk 1024, vocab 129280, pool 43136 aligned_logprobs,draft_local_ids,identity_rows |
| schedule | n1024_nextn7 | yes | 47.90 | 53.10 | 0.902 | 1.38 | 1.24 | perf The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input cum_rates,seqlens_q |
| seqlens | n1024_nextn3 | yes | 2.91 | 2.97 | 0.981 | 7.06 | 6.92 | half the rows remapped; in-place seqlens_k seqlens_k |
| state_cache | load_to_cache | yes | 4.56 | 4.57 | 0.998 | 266.34 | 265.92 | same shape as select_slice, opposite direction state_cache |
| state_cache | select_slice | yes | 3.72 | 3.11 | 1.196 | 326.20 | 390.18 | feature 584, cache 258, tensor 130, layers 8 state_tensor |
| store_kv | bf16 | yes | 689.83 | 743.91 | 0.927 | 194.95 | 180.77 | perf Same copy schedule. The gap holds at the full 65536-token benchmark size. block=64 tokens=65536 layout=BF16 KV. kv_cache |
| store_kv | v41_fp4 | yes | 125.40 | 229.62 | 0.546 | 687.77 | 375.60 | perf The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=65536 layout=V41, FP4 main kv. kv_cache |
| store_kv | v41_fp8 | yes | 117.12 | 174.52 | 0.671 | 870.70 | 584.29 | perf Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=65536 layout=V41, main/swa kv. kv_cache |
| sum_probs | b512_v129280 | yes | 86.34 | 85.92 | 1.005 | 3066.60 | 3081.40 | normalized logprobs, vocab 129280 output |
| swa | n8192_w4096 | yes | 34.79 | 34.75 | 1.001 | 3861.80 | 3866.66 | lens clipped to width output |
| v4_meta | n1024_q8 | yes | 17.20 | 17.43 | 0.987 | 31.97 | 31.55 | ca=sparse (1, 2), dense empty, block 128 positions,valid_token_mask,gate_force_random,fix_routing_mask,context_lens,klen,kv_ca_index_offset_len,kv_swa_index_offset_len,slot_mapping_store_swa,query_indices,state_block_idx_out,positions_c,slot_mapping_store_ca |
| yoco | n256_len500 | yes | 25.47 | 22.36 | 1.139 | 25.45 | 28.99 | cp_size 4, cp_rank 0, keep=seqlen cu_lens,yoco_contrib_index,yoco_gathered_reverse_index |

Rows: 64. Correct: 57. Perf misses (ratio < 0.95): 16.

## Perf misses

- `compress/decode` ratio 0.946. Both sides divide the softmax sum in float. The tile is unchanged; the residual on this 9 us row is launch and measurement. ratio=2 dim=512 seqs=64 nextn=6
- `cp_mirror/prepare_b4096_c1_2` ratio 0.828. PTO has no dynamic shift, so the binary search is straight-line constant shifts. cp_size 8, cp_rank 7, ratios (1, 2)
- `flash_mla/b256_k1024_full` ratio 0.945. Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement. in-place topk; full context; block_size 16
- `fused_transform/norm_fp8_h8_d128_n1024` ratio 0.852. The eight grouped maxima are parked with one 8-lane store and reloaded. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `fused_transform/store_mxfp8_h1_d128_n32768` ratio 0.750. The store_cache arm writes each token through the paged slot_mapping; the quantized store into the MXFP8 index cache is 11 us behind on a 33 us row. norm + e4m3 ue8m0 group 32 into MXFP8 index cache, num_heads=1 block 128, contiguous slots, slot 0 invalid
- `gather/large_d128` ratio 0.911. The residual on this 45 us row is launch and measurement. float32 viewed as int8, D=512; some OOB indexes
- `gather_engram/n1024_nextn5` ratio 0.937. Same gather body; the residual on this 9 us row is launch and measurement. max_ngram_size 4, old_ratio 0.9, vocab 100000
- `load_kv/v41_fp4` ratio 0.267. Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=65536 pattern=aligned layout=V41, FP4 main kv.
- `load_kv/v41_fp8` ratio 0.657. Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=65536 pattern=aligned layout=V41, main/swa kv.
- `norm/bf16_residual` ratio 0.754. The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False
- `sampler/get_accept_result` ratio 0.922. Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement. total_q 256, has_golden, request length 16
- `sampler/map_lookup` ratio 0.935. 3.5 us kernel, so the residual is launch and measurement. by_key float32, rows 256, force_topk 1024
- `schedule/n1024_nextn7` ratio 0.902. The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input
- `store_kv/bf16` ratio 0.927. Same copy schedule. The gap holds at the full 65536-token benchmark size. block=64 tokens=65536 layout=BF16 KV.
- `store_kv/v41_fp4` ratio 0.546. The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=65536 layout=V41, FP4 main kv.
- `store_kv/v41_fp8` ratio 0.671. Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=65536 layout=V41, main/swa kv.

## Failures

- `fused_sample/b256_v1024`: ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `fused_sample/b256_v1024_reject`: Same extui rejection as the non-reject row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `fused_sample/b256_v129280`: Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `fused_sample/b256_v129280_reject`: Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed for kernels ['_fused_sample_kernel'].
- `norm/fp8_npc32_packed`: The 4-of-128 amax mask spelling fails PTO codegen legality: store destination misaligned, byte remainder 16 vs required 32. InternalError: [VerifyPTOCodegenLegality] kernel=_norm_kernel, store=#4: destination is misaligned: byte remainder=T.int64(16), required alignment=32
- `sample/b256_v1024`: bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off. Not a membar interaction. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
- `sample/b256_v129280`: Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed for kernels ['_random_sampling_kernel'].
