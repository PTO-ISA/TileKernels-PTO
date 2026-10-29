# norm

Sources in [`norm.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py) and the helper inlined from [`rstd_1ulp.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/rstd_1ulp.py). One core.

| variant | lines | what runs |
| --- | --- | --- |
| `rms` | 154 | naive sum of squares, `vsqrt` / `vdiv`, weight. Hidden 128 |
| `rstd` | 154 | the same VF with `compensated_div_add` and `compensated_rsqrt` inlined. Not a second `SimdVF` |
| `bf16` | 235 | one row, 256 bf16 channels. ASC keeps `vld2` / `vcgmax`. PTO is `vcmax(..., group=8)` and one `vcvt` to FP8 |
| `quant` | 277 | fp32 per-32 amax and FP8. PTO uses one `vcmax(..., group=4)` |
| `unpack` | 374 | bfloat16 weight expanded to fp32 |
| `layout` | 382, 420 | transpose index, then gather |
