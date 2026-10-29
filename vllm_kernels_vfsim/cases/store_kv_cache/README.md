# store_kv_cache

Sources in [`store_kv_cache.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/store_kv_cache.py) and [`kv_quant.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/kv_quant.py). One core, 128 bf16, four groups of 32, FP8 scale `amax/448`.

| variant | line | shape |
| --- | --- | --- |
| `row` | 152 | one row. PTO takes the group maximum in one `vcmax` |
| `rows` | 184 | two rows, the same quant chain. PTO indexes the row accumulators with `T.unroll` |
