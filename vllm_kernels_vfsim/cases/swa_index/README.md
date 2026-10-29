# swa_index

Source: [`swa_index.py` line 41](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/swa_index.py#L41)

One `T.SimdVF`. Two `[offset, length]` segments expand into a width-128 index row. Columns past both segments are `-1`.

Sim shape: 2 rows, width 128, one core.
