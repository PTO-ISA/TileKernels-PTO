# get_kv_index

Sources in [`get_kv_index.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py). The `T.SimtVF` above line 214 is out of scope. One core, 64 lanes.

| variant | line | arm |
| --- | --- | --- |
| `blocks` | 52 | power-of-two block table |
| `identity` | 52 | `block_size == 0` |
| `mod` | 52 | `block_size` 6. ASC keeps integer `vdiv`. PTO converts to f32, divides, and truncates. That ratio is not a fair ISA win |
| `dflash` | 214 | SWA column to slot. Block size 8, so PTO uses `vand` with 7 instead of a divide. Not a fair ISA win |
