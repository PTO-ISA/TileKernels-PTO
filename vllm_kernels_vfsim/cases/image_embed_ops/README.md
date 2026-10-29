# image_embed_ops

Sources in [`image_embed_ops.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py). One core.

| variant | line | shape |
| --- | --- | --- |
| `copy` | 125 | 128 bf16, UB row copy |
| `pool` | 320 and 342, fp32 arm | `D=2`, dim 64, zero then mean |
| `pool_bf16` | 342, bf16 arm | `D=2`, dim 128. ASC keeps the even/odd `part` pack. PTO stores one `vcvt` to bfloat16 |
| `scatter` | 375 | transpose `[D*D, dim]` to `[dim, D, D]` |
