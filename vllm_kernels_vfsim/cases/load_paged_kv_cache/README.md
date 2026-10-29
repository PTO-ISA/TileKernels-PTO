# load_paged_kv_cache

Sources in [`load_paged_kv_cache.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/load_paged_kv_cache.py). One core.

| variant | line | arm |
| --- | --- | --- |
| `fp8` | 258 and 326 | FP8 e4m3 times a UE8M0 scale, and a missing page stores zeros. 64 elements |
| `fp4` | 258 | one page, 128 logical fp4 values. `vcvt` from packed `float4_e2m1fn` compiled. The check is the zero codepoint |

PTO FP4 is one `vinterpret_cast` and one `vcvt`, not `UNPK4` plus `part`.
