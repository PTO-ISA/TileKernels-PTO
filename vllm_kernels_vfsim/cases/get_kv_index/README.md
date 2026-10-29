# get_kv_index

Sources:

- [`get_kv_index.py` line 52](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L52) paged index. Variants `blocks` (power-of-two block table gather) and `identity` (`block_size == 0`) are the two compile-time arms that this small shape can execute. The non-power-of-two remainder arm is the same div/mul pattern as `dflash`.
- [`get_kv_index.py` line 214](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/get_kv_index.py#L214) SWA column to slot, variant `dflash`. The `T.SimtVF` above it is out of scope.

Sim shape: 64 lanes, block size 16, one core.
