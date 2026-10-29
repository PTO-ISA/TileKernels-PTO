# fused_transform

Source: [`fused_transform.py` line 477](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477)

One `T.SimdVF` with compile-time flags. This shape enables RMSNorm on one head of 128 bfloat16 values and leaves RoPE, Hadamard, and FP8 quant off. The quant flag is the same per-group scale as [`norm.py` line 277](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py#L277). PTO converts the whole head in one `vcvt` instead of two `UNPK_B16` loads.

Sim shape: 128 bfloat16, one core.
