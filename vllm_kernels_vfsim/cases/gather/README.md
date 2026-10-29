# gather

Source: [`gather.py` line 56](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/gather.py#L56)

The only `T.SimdVF` in this file is the large-row zero fill (`T.Parallel` stores). Indexed row copies are `T.copy`, kept here only so the fill is observable. Out-of-range indices stay zero.

Sim shape: 4 source rows, 4 queries, 128 int8 columns, UB pad 256. One core.

PTO replaces the scalar fill with one `vbrc` + `vstore` per row.
