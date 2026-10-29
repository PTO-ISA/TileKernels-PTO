# store_kv_cache

Sources:

- [`store_kv_cache.py` line 152](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py#L152) scalar-row quant
- [`store_kv_cache.py` line 184](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py#L184) the same chain on several rows
- helpers in [`kv_quant.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/kv_quant.py)

Sim shape: one row, 128 bfloat16, four groups of 32, FP8 e4m3 scales `amax/448`. The batched VF is this chain replicated across rows; one row is what the simulator runs. PTO takes the group maximum in one `vcmax`.
