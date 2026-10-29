# fused_transform

Source: [`fused_transform.py` line 477](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_transform.py#L477). One `T.SimdVF` with compile-time flags. Each flag is a one-head driver, `head_dim=128` except the LUT.

| variant | flag | shape |
| --- | --- | --- |
| `norm` | RMSNorm only | 128 bf16. PTO converts the head in one `vcvt` |
| `rope` | RoPE only | 64 fp32 pairs. PTO is `vmul` / `vadd` on the logical vector |
| `hadamard` | Hadamard only | one 64-lane butterfly, six stages, written once |
| `quant` | FP8 quant only | per-32 scale, one `vcvt` to FP8 |
| `lut` | FP4 reciprocal LUT | the `head_dim == 512` table, one row of 512 |

The merged-store assert that requires `head_dim == 512` is the LUT arm. It compiled and matched the integer reciprocal on both targets.
