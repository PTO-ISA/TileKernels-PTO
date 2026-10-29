# candidate_blocks

Sources in [`candidate_blocks.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/candidate_blocks.py). One core, 64 lanes.

| variant | lines | what runs |
| --- | --- | --- |
| `main` | 56, 64, 79, 90, 127, 258, 328 | counts, two prefix scans, compact, restore |
| `map` | 149 | 4-row binary search packed as `(task, length)` with `vintlv` |
| `max` | 198 | bf16 block max, `block_size` 8. PTO is `vcmax(..., group=8)` |

The main kernel's 2.03x cycle ratio is seven ASC launches of the 64-lane scan against four PTO launches. Those launches are not restacked.
