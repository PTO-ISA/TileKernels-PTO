# load_paged_kv_cache

Sources in [`load_paged_kv_cache.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py):

- line 258: FP8 e4m3 times a UE8M0 scale. Low 32 lanes use scale byte 0 shifted by 23; high 32 lanes use byte 1 shifted by 15. FP4 cast is the other compile-time arm and is not in this small shape.
- line 326: a missing page stores zeros

Sim shape: 64 elements, one core.
