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

Wins: 7. Warnings: 0. Failures: 0.


