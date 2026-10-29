# DeepInfer SimdVF, ASC and PTO, under the CPU simulator

Single-core ports of every `with T.SimdVF():` region in
[deepinfer/model_executor/layers/tilelang_ops/asc](https://github.com/deepseek-ai/DeepInfer/tree/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc).
`T.SimtVF` is not ported.

Each case is a `T.Kernel(1)` driver with a small shape. `T.copy` only moves that tile into unified buffer and back. The ASC file keeps the original vector ops. The PTO file (`target="pto"`, `vmi as V`) is the same schedule with the VMI calling convention: `size=`, `create_mask`, mask-first `vsel`, and `vload` plus `vbrc` instead of a broadcast load.

There is no NPU in this environment. `python cases/<name>/run.py asc|pto` re-executes under `msprof op simulator` (`Ascend950PR_9599`) and checks the result against a torch-CPU reference. PTO is also compared with ASC when `python run_all.py` loads both dumps.

## Coverage

The port is complete for every `with T.SimdVF():` in DeepInfer at commit `553741557a57df44a46fb3ab72b6aafeaf945337`.

| | count |
| --- | ---: |
| `with T.SimdVF():` statements | 64 |
| simulated kernels in `PERF.md` | 43 |

A kernel is one `T.Kernel(1)` launch. Its cycle number is the sum of every vector function inside that launch, so several statements share one row. Nine further rows are a second compile-time arm of a statement already counted (a flag, a dtype, or the compensated reciprocal inlined into `norm.py` line 154). `rstd_1ulp.py` has no `SimdVF` of its own. `T.SimtVF` is outside both totals.

64 statements − 30 that share a kernel with another statement + 9 extra arms = 43 kernels.

| kernel | statements |
| --- | ---: |
| `candidate_blocks` | 7 |
| `candidate_blocks/map` | 1 |
| `candidate_blocks/max` | 1 |
| `compress` | 3 |
| `dflash/dot` | 1 |
| `dflash/mean` | 1 |
| `fused_sample` | 4 |
| `fused_sample/search` | 1 |
| `fused_sample/filter` | 2 |
| `fused_sample/quartile` | 4 |
| `fused_sample/mask` | 5 |
| `fused_sample/cdf` | 4 |
| `fused_transform` | 1 |
| `gather` | 1 |
| `get_kv_index/blocks` | 1 |
| `get_kv_index/dflash` | 1 |
| `image_embed_ops/copy` | 1 |
| `image_embed_ops/pool` | 2 |
| `image_embed_ops/scatter` | 1 |
| `kv_index_cat` | 1 |
| `load_paged_kv_cache` | 2 |
| `norm/rms` | 1 |
| `norm/unpack` | 1 |
| `norm/quant` | 1 |
| `norm/layout` | 2 |
| `norm/bf16` | 1 |
| `random_sampling` | 4 |
| `sampler` | 3 |
| `schedule_mtp_varlen` | 1 |
| `store_kv_cache` | 1 |
| `store_kv_cache/rows` | 1 |
| `structure_output` | 1 |
| `sum_smaller_probs` | 1 |
| `swa_index` | 1 |
| **34 kernels** | **64** |

The other 9 kernels repeat a statement that is already in that sum:

| kernel | same statement |
| --- | --- |
| `fused_transform/rope`, `/hadamard`, `/quant`, `/lut` | `fused_transform.py:477` |
| `get_kv_index/identity`, `/mod` | `get_kv_index.py:52` |
| `image_embed_ops/pool_bf16` | `image_embed_ops.py:342` |
| `load_paged_kv_cache/fp4` | `load_paged_kv_cache.py:258` |
| `norm/rstd` | `norm.py:154` |

34 + 9 = 43 kernels. Each of the 64 statements has an ASC driver and a PTO driver, and both matched a torch-CPU reference under the simulator.

## Source map

Line numbers are for DeepInfer commit `553741557a57df44a46fb3ab72b6aafeaf945337`.

| case | source lines |
| --- | --- |
| `kv_index_cat` | [kv_index_cat.py:60](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/kv_index_cat.py#L60) |
| `structure_output` | [structure_output.py:82](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/structure_output.py#L82) |
| `swa_index` | [swa_index.py:41](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/swa_index.py#L41) |
| `gather` | [gather.py:56](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/gather.py#L56) |
| `sum_smaller_probs` | [sum_smaller_probs.py:58](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/sum_smaller_probs.py#L58) |
| `schedule_mtp_varlen` | [schedule_mtp_varlen.py:85](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/schedule_mtp_varlen.py#L85) |
| `compress` | [compress.py:116](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L116), [:147](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L147), [:374](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L374) |
| `dflash` | [dflash.py:57](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L57), [:137](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L137) |
| `get_kv_index` | [get_kv_index.py:52](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L52), [:214](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L214) |
| `image_embed_ops` | [image_embed_ops.py:125](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L125), [:320](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L320), [:342](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L342), [:375](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py#L375) |
| `norm` | [norm.py:154](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L154), [:235](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L235), [:277](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L277), [:374](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L374), [:382](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L382), [:420](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L420) |
| `random_sampling` | [random_sampling.py:224](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L224), [:232](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L232), [:235](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L235), [:252](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L252) |
| `candidate_blocks` | [candidate_blocks.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py) lines 56, 64, 79, 90, 127, 149, 198, 258, 328 |
| `sampler` | [sampler.py:181](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L181), [:247](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L247), [:633](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py#L633) |
| `load_paged_kv_cache` | [load_paged_kv_cache.py:258](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py#L258), [:326](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py#L326) |
| `store_kv_cache` | [store_kv_cache.py:152](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py#L152), [:184](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py#L184) and [kv_quant.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/kv_quant.py) |
| `fused_sample` | [fused_sample.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py) (20 regions; see that case README) |
| `fused_transform` | [fused_transform.py:477](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477) |

Files in that directory with no `T.SimdVF` are omitted (`T.SimtVF` only, or no vector function).

## Run

```bash
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"
python cases/kv_index_cat/run.py pto
python run_all.py
```

`run_all.py` writes `PERF.md` and `perf/vf_cycles.json`. Build the image from this directory with `docker build -f docker/Dockerfile .` The toolchain pin is CANN 9.2.0-beta.2, torch 2.9.0 CPU, torch-npu 2.9.0.post6, ptoas 0.70, tilelang `8e10a859`.

VF cycles are the sum of the `cycles` column on `instr` `VF` or `VF_SIMD` in `core0.veccore0_instr_exe.csv`. Do not multiply by `call_count`. The ratio is ASC cycles / PTO cycles.
