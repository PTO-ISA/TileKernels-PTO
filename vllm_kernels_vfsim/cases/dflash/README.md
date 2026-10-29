# dflash

Sources:

- [`dflash.py` line 57](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L57) dot of 4 bfloat16 rows with one weight, fp32 horizontal sum. PTO keeps the four accumulators in one `T.unroll`, not four named registers. The sigmoid that follows is `T.SimtVF` and is not ported.
- [`dflash.py` line 137](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/dflash.py#L137) mean over 2 heads.

Sim shape: dim 128, one core. Variants: `dot`, `mean`.
