# sampler

Sources in [`sampler.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/sampler.py):

- line 181: select the value whose key equals the query, then `vcmax` (`-inf` if none)
- line 247: fill a pool with `-inf`, scatter values through indices, gather them back. The production pool is 43136 wide; this shape is 64 lanes
- line 633: if the row was accepted, copy the vector, otherwise store zeros

Sim shape: 64 lanes, one core. Other `T.SimtVF` regions in this file are out of scope.
