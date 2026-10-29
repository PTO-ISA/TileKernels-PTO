# structure_output

Source: [`structure_output.py` line 82](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/structure_output.py#L82)

One `T.SimdVF`: each vocab lane tests one bit of the bitmask. A clear bit replaces the logit with `-inf`.

Sim shape: vocab 128 (two 64-lane vectors, four int32 mask words), one core, one row.
