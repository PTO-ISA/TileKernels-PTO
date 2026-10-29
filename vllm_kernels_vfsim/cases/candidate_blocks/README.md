# candidate_blocks

Sources in [`candidate_blocks.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py):

- lines 56, 64, 79: zero, `ceil(len / 16384)` per lane, horizontal sum
- lines 90 and 127: Hillis-Steele prefix of those counts (the per-core scan and the per-row tile prefix are the same vector scan)
- line 149: task-to-row binary search is the same `vselr` prefix walk; this shape checks the prefix it searches
- line 198: the bf16 block max is a horizontal max of the same kind as `vcmax` in `norm` quant
- line 258: compact token contribution `min(block, length - id * block)`
- line 328: restore `block_id * block + index % block`, or `-1`

Sim shape: 64 lanes, 4 candidate blocks, block size 16, one core. One kernel runs the count, prefix, restore, and compact VFs.
