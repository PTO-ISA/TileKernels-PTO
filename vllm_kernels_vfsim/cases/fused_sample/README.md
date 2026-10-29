# fused_sample

Source: [`fused_sample.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py). Twenty `T.SimdVF` regions. Each group below is its own driver. Vocab 128, one core.

| variant | lines | what runs |
| --- | --- | --- |
| `lse` | 214, 223, 261, 625 | online softmax and argmax |
| `search` | 35 | three-threshold mass and count |
| `filter` | 282, 292 | temperature softmax and the `2^-32` cutoff |
| `quartile` | 327, 347, 391, 446 | bound init, bit-pattern quartiles, pick, keep count |
| `mask` | 469, 484, 522, 550, 566 | 0/1 keep mask, log-mass, filtered and plain logprob |
| `cdf` | 531, 580, 592, 667 | zero, fixed-point fold, both inclusive scans |

ASC `pld` / `pst` did not lower. Both mask drivers store the predicate as 0/1 and reload it. PTO ceiling-rounds the keep count with a toward-zero convert plus one, because VMI has no `rounding="C"`.
