# random_sampling

Sources in [`random_sampling.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py):

- line 224 zero the slot accumulators
- line 232 fold `prob * 2^31` into `col % 64`
- line 235 inclusive scan of the slot sums
- line 252 inclusive scan of that CDF again (the second-level scan)

`T.SimtVF` random draw and binary search are out of scope. Sim shape: vocab 128, 64 slots, one core.
