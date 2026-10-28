# PTO Parallel operator acceptance

These tests call the new `tile_kernels/{quant,mhc,moe}/*_parallel.py` entry points directly. The ASC and CUDA implementations remain available for comparison; public dispatch is unchanged. There are counterparts for all 25 `*_asc.py` modules in those three directories, plus a scalar sqrt-softplus helper.

The implementations express elementwise math, indexing, conditions, local accumulators and collectives with `T.Parallel`, `T.alloc_var`, `T.serial` and `T.reduce_*` inside `T.SimdVF`. Kernel launch, persistent scheduling, DMA, buffer versions, expert padding and inter-core reductions follow the ASC implementation. The generic pre-split and routing SIMT paths retain their original `T.Parallel` logic.

## Toolchain dependencies

Use [TileLang #36](https://github.com/PTO-ISA/tilelang/pull/36) on the `pto-dev` integration branch, including the existing [M2 dependency](https://github.com/PTO-ISA/tilelang/pull/11), and PTOAS with both fixes:

- [PTOAS #1703](https://github.com/hw-native-sys/PTOAS/pull/1703): conservative barriers for unknown vecscope memory footprints.
- [PTOAS #1705](https://github.com/hw-native-sys/PTOAS/pull/1705): conversion of shared tail masks across 16/32-bit consumers; stacked on #1703.

The acceptance environment uses Ascend 950, CANN 9.3, Python 3.12, PyTorch/torch_npu 2.14 and the PTO VPTO backend. Point the `ptoas` executable at the patched build. Mixed FP4/FP8 layout search needs a larger existing search budget:

```bash
export PTO_FLAGS='--vmi-layout-max-frontier-entries=4096 --vmi-layout-max-transitions=1000000'
```

## Recorded validation

On 2026-10-28, all **110 device cases passed** on Ascend 950: 44 mHC/core, 26 quantization, 23 remaining operator/compiler features, and 17 MoE/batched-SF/atomic cases. All four task-submit jobs exited successfully. The tested compiler commits are TileLang `dec29c2f901e11fe2b82369533611e45274870a9` and PTOAS `81d9b3570f46a0b2616d4b00f38011c451409778` (including #1703).

## Run on a queued device

Install the repository and select the intended TileLang/PTOAS builds before submitting. All commands that access an NPU must run through `task-submit`. Do not use pytest-xdist here: one queue allocation corresponds to one test process/device.

```bash
task-submit --device auto --max-time 3600 --timeout 3700 --run '
  export PTO_FLAGS="--vmi-layout-max-frontier-entries=4096 --vmi-layout-max-transitions=1000000"
  python -m pytest -q -s tests/parallel --tb=short
'
```

Tests reject execution without `TASK_DEVICE`. Inputs and independent PyTorch references are constructed on CPU; outputs are copied back and compared. Quantization and index outputs use exact comparisons where applicable, while reduction and transcendental paths have explicit tolerances. The tests are correctness acceptance, not a performance benchmark or an exhaustive enumeration of all shape/configuration combinations.

`record()` verifies generated `pto.vmi.*` and writes PTODSL sources to `artifacts/parallel-generated/`, or to `PARALLEL_GENERATED_DIR` when set. The unchanged generic SIMT pre-split path is checked numerically without the VMI assertion.

| ASC module / Parallel counterpart | Acceptance test |
|---|---|
| quant/batched_transpose_weight_sf | `test_parallel_moe_batch_device.py::test_batched_sf` |
| quant/cast | `test_parallel_new_quant_device.py::test_cast` |
| quant/cast_back | `test_parallel_device.py::test_cast_back` |
| quant/norm_forward, norm_backward | `test_parallel_new_quant_device.py::test_norm`; quant/Mega SF in `test_parallel_quant_paths_device.py` |
| quant/per_block_cast | `test_parallel_quant_extra_device.py::test_per_block`; packed FP4/FP8 in `test_parallel_quant_layout_device.py` |
| quant/per_block_cast_lossless | `test_parallel_new_quant_device.py::test_lossless` |
| quant/per_channel_cast | `test_parallel_device.py::test_per_channel` |
| quant/per_token_cast_and_cast_back | `test_parallel_new_quant_device.py::test_roundtrip` |
| quant/per_token_cast | `test_parallel_unblocked_device.py::test_per_token`; packed layouts in `test_parallel_quant_layout_device.py` |
| quant/swiglu_forward, swiglu_backward | `test_parallel_new_quant_device.py::test_swiglu`; quantized backward in `test_parallel_quant_paths_device.py` |
| mhc/expand | `test_parallel_device.py::test_expand` (both directions) |
| mhc/head_compute_mix | `test_parallel_head_device.py` (both directions) |
| mhc/multilayer_recompute | `test_parallel_mhc_device.py::test_multilayer` |
| mhc/norm_fn | merge and forward partial reduction in `test_parallel_mhc_device.py`; backward RMS in `test_parallel_unblocked_device.py` |
| mhc/post, pre_apply_mix | `test_parallel_device.py::test_mix_fwd`; `test_parallel_unblocked_device.py::test_mhc_backward` |
| mhc/pre_big_fuse, pre_split_mixes | `test_parallel_mhc_last_device.py` (SIMD and generic SIMT, backward included) |
| mhc/sinkhorn | `test_parallel_sinkhorn_device.py` (forward/backward, 1/3/10 iterations) |
| moe/moe_topk_gate_forward, moe_topk_gate_backward | `test_parallel_moe_batch_device.py::test_moe` |
| moe/normalize_weight | `test_parallel_device.py::test_normalize_weight` |
| moe/topk_gate | `test_parallel_unblocked_device.py::test_topk` (stable ties and tails) |

Additional tests exercise exact division, gathers, transpose, branch-local temporaries, FP4 rounding boundaries, and signed/unsigned integer atomics.

## Necessary differences from ASC

- Additional reduction temporaries are included in the UB budget. Stochastic cast reuses one hash scratch tile across the original four-buffer schedule.
- Small mHC mix rows are flattened for continuous vector writes; single values use compiler-generated point writeback.
- Packed per-block column scales pad their UB row stride to 32 bytes and preserve the caller's output stride.
- FP4 uses pair-aligned even extents. FP32 narrowing uses round-to-odd BF16 before FP4 to preserve midpoint rounding.
- SwigLU uses exact division; the fast approximate-division certificate is unnecessary.
- MoE backward expresses selected-expert contributions as a dense expert loop with a serial top-k selection. The ASC global-memory copies and sequence partitioning are retained. Its one-dimensional Persistent uses `group_size=1`; nonnegative sequence divisions stay int32 before explicitly widening GM/UB address coordinates.
