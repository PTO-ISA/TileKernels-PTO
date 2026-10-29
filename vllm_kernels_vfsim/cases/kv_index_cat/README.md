# kv_index_cat

Source: [`kv_index_cat.py` line 60](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/kv_index_cat.py#L60)

One `T.SimdVF`: `out = index + offset`, except lanes equal to `-1` stay `-1`.

Sim shape: 2 rows x 128 int32, one core. The production kernel is multi-core and persistent; this driver only copies a tile into UB, runs the VF, and copies it back.
