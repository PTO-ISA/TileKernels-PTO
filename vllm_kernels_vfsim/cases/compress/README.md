# compress

Sources:

- [`compress.py` line 116](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L116) prefill weighted sum
- [`compress.py` line 147](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L147) invalid block stores zeros
- [`compress.py` line 374](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/compress.py#L374) decode window, score and latent packed on axis 1

Sim shape: window 2, dim 64, one core. Output rows are the prefill result, a zero row, and the decode result.
