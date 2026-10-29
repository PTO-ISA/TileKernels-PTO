# fused_sample

Source: [`fused_sample.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py) (20 `T.SimdVF` regions, lines 35, 214, 223, 261, 282, 292, 327, 347, 391, 446, 469, 484, 522, 531, 550, 566, 580, 592, 625, 667).

The simulated kernel is the part that is not already covered by a smaller case:

- init running max, merge two chunks of online softmax, write `logsumexp` (lines 214, 223, 261)
- argmax, ties take the smallest index (line 625)

The other regions are the same vector patterns, checked in the cases that own that pattern:

- thresholded exp sums and merging adds: `sum_smaller_probs` (lines 35, 292, 469, 484)
- bitmask-style select: `structure_output` (lines 550, 566)
- fixed-point fold and both inclusive scans: `random_sampling` (lines 531, 580, 592, 667)
- quartile bit shifts and predicate pick: the same `vshl` / `vsel` / `vcmp` sequence as `load_paged_kv_cache` and `schedule_mtp_varlen` (lines 327, 347, 391, 446, 522)

Sim shape: vocab 128, one core. A full 8192-wide search is not run under the simulator.
