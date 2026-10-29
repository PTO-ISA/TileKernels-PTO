# sum_smaller_probs

Source: [`sum_smaller_probs.py` line 58](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sum_smaller_probs.py#L58)

One `T.SimdVF`. Sum `exp(logprob)` over lanes strictly below the sampled token's logprob, and over equal logprobs only at indices at or before the sample.

Sim shape: vocab 128, sample index 70, one core. The production kernel walks 8192-wide chunks; this driver is one chunk.
