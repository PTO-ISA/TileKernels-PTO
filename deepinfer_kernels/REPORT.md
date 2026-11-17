# ASC vs PTO on-device report

Kernels are the DeepInfer [`tilelang_ops/asc`](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc) set
(DeepInfer `a679bb8`), launched on all vector cores.
Toolchain: CANN 9.2.0, torch-npu 2.9.0.post6, ptoas 0.76, tilelang 0.1.15+cpu.git59f3924e
([tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/59f3924e85ea2f6d3ccec2f00c87044951a783f7) `59f3924`).
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
| bitmask | b512_v129280_ftp | yes | 159.08 | 163.34 | 0.974 | 3380.80 | 3292.49 | has_ftp, step 0, one row per request logits |
| block_tables | n256_q8_b4096 | yes | 11.34 | 10.57 | 1.073 | 3328.26 | 3570.79 | full max_num_blocks rows block_tables_out |
| candidate | block_max | yes | 181.14 | 181.42 | 0.998 | 3149.02 | 3144.16 | rows=256 valid_len=1048576 max_model_len=1048576 block=16 block_scores,valid_blocks |
| candidate | compact_lens | yes | 7.69 | 7.90 | 0.973 | 68.35 | 66.53 | rows=128 topk_blocks=1024 block=16 compact_lens |
| candidate | restore_indices | yes | 2.35 | 2.41 | 0.975 | 475.13 | 463.31 | rows=128 topk_tokens=1024 topk_blocks=128 block=16 offset=1 compact_indices |
| compress | decode | yes | 8.70 | 8.70 | 1.000 | 8538.40 | 8542.08 | ratio=2 dim=512 seqs=64 nextn=6 kv_compressed,state_cache |
| compress | prefill | yes | 200.72 | 207.07 | 0.969 | 1549.58 | 1502.09 | ratio=2 dim=512 has_index=0 seqs=8 qlen=8192 kv_compressed,state_cache |
| cp_mirror | prepare_b4096_c1_2 | yes | 93.47 | 113.22 | 0.826 | 20.07 | 16.57 | perf PTO has no dynamic shift, so the binary search is straight-line constant shifts. cp_size 8, cp_rank 7, ratios (1, 2) cu_lens,mirror_cp_reverse_index,mirror_cp_index,ks,ke |
| dflash | confidence_head | yes | 13.17 | 12.86 | 1.025 | 2549.31 | 2612.67 | N=4097 n_embed=4096 markov=0 out |
| dflash | input_store | yes | 3.54 | 3.42 | 1.035 | 1482.34 | 1534.94 | n=128 n_embed=4096 hc_mult=4 out |
| dflash_copy | src257_dst512 | yes | 3.22 | 3.16 | 1.021 | 22.25 | 22.71 | has_driver_inputs first_token_positions,last_valid_offset,golden_tokens,sample_positions,temperatures,top_ps,top_ks,seq_seeds,mtp_info_write_slots |
| dflash_input | n1024_blk5_swa512 | yes | 11.42 | 10.44 | 1.094 | 1063.27 | 1163.10 | valid_token_mask on, gate_force_random off (None) newx_slot_mapping,positions,input_ids,valid_token_mask,kv_index |
| flash_mla | b256_k1024_full | yes | 6.99 | 6.97 | 1.002 | 900.24 | 902.41 | in-place topk; full context; block_size 16 topk_idxs |
| fused_sample | b256_v1024 | no |  |  |  |  |  | ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected. RuntimeError: PTO lowering failed. |
| fused_sample | b256_v1024_reject | no |  |  |  |  |  | Same extui rejection as the non-reject row. RuntimeError: PTO lowering failed. |
| fused_sample | b256_v129280 | no |  |  |  |  |  | Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed. |
| fused_sample | b256_v129280_reject | no |  |  |  |  |  | Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed. |
| fused_transform | norm_fp8_h8_d128_n1024 | yes | 13.38 | 16.02 | 0.836 | 237.51 | 198.45 | perf The eight grouped maxima are parked with one 8-lane store and reloaded. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 quant,sf |
| fused_transform | norm_rope_h8_d128_n4096 | yes | 22.21 | 22.53 | 0.986 | 803.37 | 791.86 | bf16 norm+rope, rotary 64; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8 bf16 |
| fused_transform | rope_h64_d512_n32768 | yes | 1327.20 | 1338.52 | 0.992 | 3236.99 | 3209.62 | captured 550B Q: rope only, rotary 64 offset 448, num_heads=64 bf16 |
| fused_transform | store_mxfp8_h1_d128_n32768 | yes | 32.20 | 43.93 | 0.733 | 394.80 | 289.43 | perf The store_cache arm writes each token through the paged slot_mapping; the quantized store into the MXFP8 index cache is 11 us behind on a 33 us row. norm + e4m3 ue8m0 group 32 into MXFP8 index cache, num_heads=1 block 128, contiguous slots, slot 0 invalid kv_cache |
| gather | large_d128 | yes | 44.46 | 48.19 | 0.923 | 3030.86 | 2796.08 | perf The residual on this 45 us row is launch and measurement. float32 viewed as int8, D=512; some OOB indexes Y |
| gather | small_d4 | yes | 7.05 | 6.51 | 1.083 | 669.38 | 725.10 | float32 viewed as int8, D=16; some OOB indexes Y |
| gather_engram | n1024_nextn5 | yes | 8.63 | 9.11 | 0.947 | 68.82 | 65.14 | perf Same gather body; the residual on this 9 us row is launch and measurement. max_ngram_size 4, old_ratio 0.9, vocab 100000 out_ngram_token_ids |
| gather_ids | n1024_nextn7 | yes | 4.23 | 4.18 | 1.010 | 25.39 | 25.64 | old_ratio 0.9 out_input_ids |
| image | batch_downsample_concat_i32_v54 | yes | 498.17 | 496.66 | 1.003 | 767.22 | 769.55 | imgs=32 n_vit=54 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_concat_i8_v24 | yes | 29.67 | 29.54 | 1.004 | 636.25 | 638.86 | imgs=8 n_vit=24 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_concat_i8_v96 | yes | 392.55 | 392.28 | 1.001 | 769.31 | 769.84 | imgs=8 n_vit=96 dim=1024 ratio=3 downsample=0 images_embeds |
| image | batch_downsample_pool_i32_v54 | yes | 222.74 | 229.25 | 0.972 | 953.31 | 926.21 | imgs=32 n_vit=54 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | batch_downsample_pool_i8_v24 | yes | 14.34 | 14.68 | 0.977 | 731.00 | 714.09 | imgs=8 n_vit=24 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | batch_downsample_pool_i8_v96 | yes | 177.03 | 181.62 | 0.975 | 947.72 | 923.78 | imgs=8 n_vit=96 dim=1024 ratio=3 downsample=1 images_embeds_sum |
| image | replace_embeds | yes | 11.57 | 11.08 | 1.045 | 3444.61 | 3598.13 | tokens=4096 hidden=4096 input_embeds |
| image | rgb_pad | yes | 672.34 | 646.72 | 1.040 | 332.60 | 345.78 | patch=14 imgs=50 n_vit=54 padded=1 output |
| kv_index | paged_n1024_b128 | yes | 13.96 | 14.45 | 0.966 | 2422.93 | 2341.51 | index_length 8192, block_size 128, in-bounds tables kv_index |
| kv_index | swa_n1024 | yes | 5.25 | 5.18 | 1.014 | 101.41 | 102.81 | index_length 128, block_size 128, no block tables kv_index |
| kv_index_cat | q16384_w128_k1024 | yes | 43.65 | 43.64 | 1.000 | 3459.20 | 3459.76 | largest _KV_INDEX_CAT_CASES row out |
| load_kv | bf16 | yes | 35.89 | 35.95 | 0.998 | 3740.35 | 3733.80 | block=64 tokens=65536 pattern=aligned layout=BF16 KV. kv |
| load_kv | mxfp8_index | yes | 8.06 | 7.95 | 1.014 | 2148.61 | 2178.54 | block=64 tokens=65536 pattern=aligned layout=V41, index kv. kv,kv_sf |
| load_kv | v41_fp4 | yes | 27.89 | 113.44 | 0.246 | 3083.99 | 758.08 | perf Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=65536 pattern=aligned layout=V41, FP4 main kv. kv |
| load_kv | v41_fp8 | yes | 40.60 | 62.05 | 0.654 | 2505.50 | 1639.36 | perf Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=65536 pattern=aligned layout=V41, main/swa kv. kv |
| mtp | v1024_index_conf | yes | 8.71 | 8.49 | 1.026 | 967.34 | 992.74 | req 128, nextn 4, index and confidence, slot 0 skipped token_ids_pool,logprobs_pool,logprobs_index_pool,confidence_pool |
| mtp | v129280_no_index | yes | 133.35 | 133.05 | 1.002 | 3986.61 | 3995.55 | req 128, nextn 4, confidence on, unique slots, slot 0 skipped token_ids_pool,logprobs_pool,confidence_pool |
| norm | bf16 | yes | 273.89 | 274.07 | 0.999 | 3430.40 | 3428.04 | tokens=32768 hidden=7168 residual=False fp8=False y |
| norm | bf16_residual | yes | 588.35 | 782.56 | 0.752 | 3193.78 | 2401.18 | perf The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False y,residual_out |
| norm | fp8_npc32_packed | no |  |  |  |  |  | The 4-of-128 amax mask spelling fails PTO codegen legality: store destination misaligned, byte remainder 16 vs required 32. InternalError: [VerifyPTOCodegenLegality] kernel=_norm_kernel, store=#4: destination is misaligned: byte remainder=T.int64(16), required alignment=32 |
| sample | b256_v1024 | no |  |  |  |  |  | bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off. Not a membar interaction. RuntimeError: PTO lowering failed. |
| sample | b256_v129280 | no |  |  |  |  |  | Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed. |
| sampler | expand_sampling_params | yes | 4.15 | 4.22 | 0.984 | 10.03 | 9.87 | req 256, max_steps 7 temperatures_q,top_ps_q,top_ks_q,sample_positions_q,seq_seeds_q |
| sampler | finalize_verification | yes | 9.66 | 8.93 | 1.083 | 85.88 | 92.97 | req 256, max_steps 3, topk 64, confidence and ssp last_valid_offset,num_generated_tokens,num_verified_tokens,batch_confidence_error,emitted_stop_token,sampled_token_ids,sampled_token_logprobs,sampled_sum_smaller_probs,sampled_top_logprobs,sampled_top_token_ids,verified_token_ids,verified_token_logprobs,verified_sum_smaller_probs,verified_top_logprobs,verified_top_token_ids |
| sampler | get_accept_result | yes | 8.97 | 9.03 | 0.993 | 0.81 | 0.80 | total_q 256, has_golden, request length 16 token_ids,effective_accept,is_final |
| sampler | map_lookup | yes | 3.64 | 3.37 | 1.083 | 575.98 | 623.58 | by_key float32, rows 256, force_topk 1024 out |
| sampler | prepare_verification_metadata | yes | 2.82 | 2.78 | 1.012 | 4.54 | 4.60 | req 256, max_steps 7 row_to_req,local_step,is_verify |
| sampler | remap_draft_topk | yes | 9.84 | 9.90 | 0.994 | 426.69 | 424.24 | total_q 256, force_topk 1024, vocab 129280, pool 43136 aligned_logprobs,draft_local_ids,identity_rows |
| schedule | n1024_nextn7 | yes | 47.89 | 53.30 | 0.899 | 1.38 | 1.24 | perf The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input cum_rates,seqlens_q |
| seqlens | n1024_nextn3 | yes | 2.78 | 2.80 | 0.992 | 7.39 | 7.33 | half the rows remapped; in-place seqlens_k seqlens_k |
| state_cache | load_to_cache | yes | 4.53 | 4.88 | 0.928 | 268.37 | 249.10 | perf 4.5 us kernel, so the residual is launch and measurement. same shape as select_slice, opposite direction state_cache |
| state_cache | select_slice | yes | 3.66 | 3.60 | 1.018 | 331.55 | 337.55 | feature 584, cache 258, tensor 130, layers 8 state_tensor |
| store_kv | bf16 | yes | 690.51 | 743.34 | 0.929 | 194.75 | 180.91 | perf Same copy schedule. The gap holds at the full 65536-token benchmark size. block=64 tokens=65536 layout=BF16 KV. kv_cache |
| store_kv | v41_fp4 | yes | 125.47 | 229.56 | 0.547 | 687.36 | 375.70 | perf The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=65536 layout=V41, FP4 main kv. kv_cache |
| store_kv | v41_fp8 | yes | 115.47 | 174.43 | 0.662 | 883.10 | 584.60 | perf Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=65536 layout=V41, main/swa kv. kv_cache |
| sum_probs | b512_v129280 | yes | 84.31 | 84.00 | 1.004 | 3140.51 | 3152.20 | normalized logprobs, vocab 129280 output |
| swa | n8192_w4096 | yes | 34.63 | 34.70 | 0.998 | 3879.25 | 3871.82 | lens clipped to width output |
| v4_meta | n1024_q8 | yes | 17.20 | 17.88 | 0.962 | 31.98 | 30.76 | ca=sparse (1, 2), dense empty, block 128 positions,valid_token_mask,gate_force_random,fix_routing_mask,context_lens,klen,kv_ca_index_offset_len,kv_swa_index_offset_len,slot_mapping_store_swa,query_indices,state_block_idx_out,positions_c,slot_mapping_store_ca |
| yoco | n256_len500 | yes | 25.42 | 22.44 | 1.133 | 25.50 | 28.89 | cp_size 4, cp_rank 0, keep=seqlen cu_lens,yoco_contrib_index,yoco_gathered_reverse_index |

Rows: 64. Correct: 57. Perf misses (ratio < 0.95): 13.

## Perf misses

- `cp_mirror/prepare_b4096_c1_2` ratio 0.826. PTO has no dynamic shift, so the binary search is straight-line constant shifts. cp_size 8, cp_rank 7, ratios (1, 2)
- `fused_transform/norm_fp8_h8_d128_n1024` ratio 0.836. The eight grouped maxima are parked with one 8-lane store and reloaded. norm + e4m3 ue8m0 group 32; KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8
- `fused_transform/store_mxfp8_h1_d128_n32768` ratio 0.733. The store_cache arm writes each token through the paged slot_mapping; the quantized store into the MXFP8 index cache is 11 us behind on a 33 us row. norm + e4m3 ue8m0 group 32 into MXFP8 index cache, num_heads=1 block 128, contiguous slots, slot 0 invalid
- `gather/large_d128` ratio 0.923. The residual on this 45 us row is launch and measurement. float32 viewed as int8, D=512; some OOB indexes
- `gather_engram/n1024_nextn5` ratio 0.947. Same gather body; the residual on this 9 us row is launch and measurement. max_ngram_size 4, old_ratio 0.9, vocab 100000
- `load_kv/v41_fp4` ratio 0.246. Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs. block=64 tokens=65536 pattern=aligned layout=V41, FP4 main kv.
- `load_kv/v41_fp8` ratio 0.654. Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load. block=64 tokens=65536 pattern=aligned layout=V41, main/swa kv.
- `norm/bf16_residual` ratio 0.752. The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. tokens=32768 hidden=7168 residual=True fp8=False
- `schedule/n1024_nextn7` ratio 0.899. The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg. bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input
- `state_cache/load_to_cache` ratio 0.928. 4.5 us kernel, so the residual is launch and measurement. same shape as select_slice, opposite direction
- `store_kv/bf16` ratio 0.929. Same copy schedule. The gap holds at the full 65536-token benchmark size. block=64 tokens=65536 layout=BF16 KV.
- `store_kv/v41_fp4` ratio 0.547. The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. block=64 tokens=65536 layout=V41, FP4 main kv.
- `store_kv/v41_fp8` ratio 0.662. Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input. block=64 tokens=65536 layout=V41, main/swa kv.

## Failures

- `fused_sample/b256_v1024`: ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected. RuntimeError: PTO lowering failed.
- `fused_sample/b256_v1024_reject`: Same extui rejection as the non-reject row. RuntimeError: PTO lowering failed.
- `fused_sample/b256_v129280`: Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed.
- `fused_sample/b256_v129280_reject`: Same extui rejection as the vocab-1024 row. RuntimeError: PTO lowering failed.
- `norm/fp8_npc32_packed`: The 4-of-128 amax mask spelling fails PTO codegen legality: store destination misaligned, byte remainder 16 vs required 32. InternalError: [VerifyPTOCodegenLegality] kernel=_norm_kernel, store=#4: destination is misaligned: byte remainder=T.int64(16), required alignment=32
- `sample/b256_v1024`: bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off. Not a membar interaction. RuntimeError: PTO lowering failed.
- `sample/b256_v129280`: Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch. RuntimeError: PTO lowering failed.
