# norm

Sources in [`norm.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/norm.py) and the inlined helper [`rstd_1ulp.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/rstd_1ulp.py):

- line 154, variant `rms`: sum of squares, reciprocal square root, weight. Hidden 128, one row, fp32. The simulated rstd is the direct `vsqrt`/`vdiv` estimate; production wraps that estimate in the compensated residual from `rstd_1ulp.py`.
- line 374, variant `unpack`: bfloat16 weight expanded to fp32
- line 277, variant `quant`: per-32 amax, scale `amax/448`, FP8 e4m3. PTO uses one `vcmax(..., group=4)`
- lines 382 and 420, variant `layout`: transpose indices `row = lane % 8`, `pack = lane >> 3`, then gather

Line 235 is the bfloat16 fast amax. Its ASC body exists because 32 bfloat16 values are not one register group; the PTO form of that reduction is the grouped `vcmax` in `quant`.

Sim shape: hidden 128, one core.
