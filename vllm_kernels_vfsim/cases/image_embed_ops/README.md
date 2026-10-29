# image_embed_ops

Sources, all under [`image_embed_ops.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/image_embed_ops.py):

- line 125, variant `copy`: bfloat16 UB row copy, 128 lanes
- line 320, variant `pool`: zero a `D*D` by `dim` window before the mean
- line 342, variant `pool`: sum the window in fp32 and scale by `1/(vh*vw)`
- line 375, variant `scatter`: transpose `[D*D, dim]` to `[dim, D, D]`

Sim shape: hidden 128, `D=2`, dim 64, fp32 for the pool and scatter. One core.
