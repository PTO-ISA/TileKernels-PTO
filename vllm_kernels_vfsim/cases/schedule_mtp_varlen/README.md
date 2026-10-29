# schedule_mtp_varlen

Source: [`schedule_mtp_varlen.py` line 85](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/schedule_mtp_varlen.py#L85)

One `T.SimdVF`. Each of `nextn` cumulative rates is ranked against every other rate (strictly greater, or equal and a smaller flat index). The sequence length is one base token plus the number of ranks that fit in `extra_budget`.

The `T.SimtVF` prefix product that builds those rates is out of scope. Sim shape: 2 sequences, `nextn=4`, rates padded to 64 with `-inf`, one core.
