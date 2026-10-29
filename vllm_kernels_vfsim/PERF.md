# PTO:ASC simulated VF cycles

Ratio = ASC VF cycles / PTO VF cycles. Above 1.0 means the PTO kernel spent fewer simulated vector-function cycles.

Cycles are the sum of the `cycles` column on `VF` / `VF_SIMD` rows in `core0.veccore0_instr_exe.csv`. `call_count` is not a multiplier. A case that fails its numeric check has no ratio.

- `OK`: ratio in [0.8, 1.1]
- `WIN`: PTO faster than 1.1x ASC
- `WARN`: PTO slower than 0.8x ASC
- `FAIL`: numeric check failed

| case | ASC cycles | PTO cycles | PTO:ASC | flag | ASC vs PTO |
| --- | ---: | ---: | ---: | --- | --- |
| candidate_blocks | 1627.0 | 802.0 | 2.029x | WIN | match |
| compress | 762.0 | 706.0 | 1.079x | OK | match |
| dflash/dot | 91.0 | 91.0 | 1.000x | OK | match |
| dflash/mean | 66.0 | 71.0 | 0.930x | OK | match |
| fused_sample | 134.0 | 136.0 | 0.985x | OK | match |
| fused_transform | 168.0 | 141.0 | 1.191x | WIN | match |
| gather | 458.0 | 467.0 | 0.981x | OK | match |
| get_kv_index/blocks | 114.0 | 114.0 | 1.000x | OK | match |
| get_kv_index/identity | 462.0 | 471.0 | 0.981x | OK | match |
| get_kv_index/dflash | 574.0 | 475.0 | 1.208x | WIN | match |
| image_embed_ops/copy | 50.0 | 50.0 | 1.000x | OK | match |
| image_embed_ops/pool | 526.0 | 536.0 | 0.981x | OK | match |
| image_embed_ops/scatter | 88.0 | 88.0 | 1.000x | OK | match |
| kv_index_cat | 62.0 | 62.0 | 1.000x | OK | match |
| load_paged_kv_cache | 596.0 | 601.0 | 0.992x | OK | match |
| norm/rms | 162.0 | 129.0 | 1.256x | WIN | match |
| norm/unpack | 54.0 | 58.0 | 0.931x | OK | match |
| norm/quant | 181.0 | 126.0 | 1.437x | WIN | no-dump |
| norm/layout | 78.0 | 78.0 | 1.000x | OK | no-dump |
| random_sampling | 1449.0 | 1402.0 | 1.034x | OK | match |
| sampler | 709.0 | 312.0 | 2.272x | WIN | match |
| schedule_mtp_varlen | 246.0 | 225.0 | 1.093x | OK | match |
| store_kv_cache | 200.0 | 134.0 | 1.493x | WIN | no-dump |
| structure_output | 78.0 | 78.0 | 1.000x | OK | match |
| sum_smaller_probs | 92.0 | 91.0 | 1.011x | OK | match |
| swa_index | 119.0 | 119.0 | 1.000x | OK | match |
| candidate_blocks/map | 260.0 | 260.0 | 1.000x | OK | match |
| candidate_blocks/max | 88.0 | 88.0 | 1.000x | OK | match |
| fused_sample/search | 112.0 | 108.0 | 1.037x | OK | match |
| fused_sample/filter | 596.0 | 602.0 | 0.990x | OK | match |
| fused_sample/quartile | 701.0 | 681.0 | 1.029x | OK | match |
| fused_sample/mask | 780.0 | 698.0 | 1.117x | WIN | match |
| fused_sample/cdf | 1462.0 | 1415.0 | 1.033x | OK | match |
| fused_transform/rope | 61.0 | 61.0 | 1.000x | OK | match |
| fused_transform/hadamard | 167.0 | 166.0 | 1.006x | OK | match |
| fused_transform/quant | 182.0 | 126.0 | 1.444x | WIN | match |
| fused_transform/lut | 122.0 | 127.0 | 0.961x | OK | match |
| get_kv_index/mod | 572.0 | 499.0 | 1.146x | WIN | match |
| image_embed_ops/pool_bf16 | 73.0 | 81.0 | 0.901x | OK | match |
| load_paged_kv_cache/fp4 | 58.0 | 58.0 | 1.000x | OK | match |
| norm/rstd | 175.0 | 152.0 | 1.151x | WIN | match |
| norm/bf16 | 708.0 | 122.0 | 5.803x | WIN | match |
| store_kv_cache/rows | 707.0 | 150.0 | 4.713x | WIN | match |

Wins: 13. Warnings: 0. Failures: 0.

New and edited variants were remeasured on `Ascend950PR_9599`. Untouched rows keep their previous cycles.

`candidate_blocks` (the main kernel) is 2.03x because ASC launches the same 64-lane scan seven times and PTO four times. `get_kv_index/dflash` uses `vand` where ASC divides by 8. `get_kv_index/mod` uses a float divide where ASC uses an integer divide. Those three ratios are not fair ISA comparisons. `sampler` matches the same 64-lane scatter; its 2.27x gap is larger than the extra ASC index reload.
