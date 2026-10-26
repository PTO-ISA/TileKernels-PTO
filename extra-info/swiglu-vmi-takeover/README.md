# SwiGLU ASC → VMI takeover

This work supersedes PTO-ISA/TileKernels-PTO#20, with provenance retained from kuri780's original VMI implementation. The replacement preserves the current ASC scheduling shell and routes the Ascend public API to the VMI factories.

## Review disposition

| Original feedback | Implementation / evidence | State |
|---|---|---|
| Omit optional masks (discussion_r4344151453) | Full masks omitted for operations whose frontend supports optional masks; required reduction/comparison/scalar-operation masks retained. | Implemented; full validation in progress |
| Restore differently typed T.copy using TileLang PR14 (discussion_r4344353367) | ASC DMA expressions restored verbatim. Latest pto-dev compiled and ran packed SF paths without applying PR14; PR14 itself remains open. No claim that it merged. | Representative packed paths pass; extended paths pending |
| Use a 256-lane amax register (discussion_r4344372026) | FP8 forward without clamp counts computes 256 lanes and reduces with group=8: eight outputs, each covering 32 channels. group=4 would combine 64 channels and violate sf_block=(1,32). Clamp-count path retains64. | hidden2048/3584 bit-exact, three rounds |
| Remove operator-local software exact divide | All local exact-div/sigmoid correction routines removed; precision='exact' requests the toolchain implementation. | Implemented; adversarial validation pending |
| Match ASC outside SIMD | Scalar predicates, buffers, DMA copies, pipeline stages, persistent loops and Simt atomic count code match ASC via normalized AST audit. | Audit passes; helper contracts also require manual review |

## Frozen toolchain

- TileLang PTO-ISA/pto-dev: `4eb83fdeb2fb317a0485052f51d0b77f4238d8b5`.
- PTOAS GitCode cann/master: `04c785491e844ec6d642e444a4e3ee84d1a60496`, plus a local Python-only atomic signedness fix. This is not unmodified master. The patch normalizes integer atomic pointer/value operands to signless common IR and restores the authored return type.
- CANN9.2.0; isolated builds, source and binary hashes recorded outside the repository pending final packaging.

## Representative performance (not full acceptance)

Same process, NPU1, ASC/VMI alternating order, three msprof rounds. Ratio=ASC_us/VMI_us; higher is faster. Weighted forward FP8 with packed column-major scales, no clamp:

| hidden | ASC median us | VMI median us | median ratio | output/SF mismatch bytes |
|---|---|---|---|---|
| 2048 | 186.491 | 189.362 | 0.9848 | 0 |
| 3584 | 299.453 | 301.357 | 0.9937 | 0 |

These measurements do not characterize backward, other scale layouts, clamp counts, or partial-width cases. Final performance tables and verified dependencies will replace this provisional report before merge.

## Core matrix result, historical candidate v5

All24 forward and24 backward cases passed precision checks, each with3 alternating-order timing rounds. Median per-case ratios grouped by direction/dtype:

| direction | BF16 | FP32 | FP8 |
|---|---|---|---|
| forward | 1.0333 | 1.0242 | 0.9719 |
| backward | 0.9030 | 0.9917 | 0.8771 |

These group medians hide regressions: FP8 forward clamp at hidden2048 is0.8689. Experimental128-lane clamp compute/counters improves two gates to0.9471/0.9382 but has not completed full validation. The submitted source includes the divisor-aware logical-width fix for hidden not divisible by256; it matchesv5 on the core hidden2048/3584 configurations. The separate divisor-aware candidate has passed26 extended forward cases, including hidden128/384/4608 and all3 SF configurations with and without psum. All26 backward extended cases also pass.

Backward artifacts show the BF16 memory path already matches ASC. One arithmetic difference is the reference's specialized single-neighbor sigmoid correction versus the general double-neighbor correction selected by VMI precision='exact'. A diagnostic ASC-only general-div substitution changes ratios toward0.95; this is causal-probe evidence, not a replacement acceptance baseline. Operator-local software divide remains removed.

## Reproducing the audit

`check_shell.py` compares the normalized ASC/VMI outer scheduling AST. `bench_core.py` runs the repository level-0 cases through the public API with original ASC and VMI factories, checks bytes (weight gradient uses the repository1e-4 tolerance), and records three alternating-order msprof rounds. It requires `SWIGLU_AUDIT_ROOT` pointing to an isolated directory with `tk`, `tilelang`, `ptoas`, `pysite/sitecustomize.py` and `ptoas-build.json` matching the exact commits above. `TILELANG_CACHE_DIR` must be fresh per revision, with matching PTODSL/native libraries and CANN environment. The host layout and pinned versions are asserted; the script is an investigation harness rather than a general installed-package benchmark. The public-API correctness tests against torch and final revision artifact/canary audit remain outstanding.

The JSON case files describe the additional branch coverage; raw device logs and compiler dumps remain outside this PR to avoid committing large temporary artifacts. The per-case core table is historical candidatev5 and must be refreshed for the final source.
