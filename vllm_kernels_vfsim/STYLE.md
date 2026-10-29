# Where PTO is longer than ASC

VMI asks for `size=` and a mask on stores and on scalar-vector ops. A direct ASC body (one load, one add, one store) grows by those arguments and is not shorter. That shows up as source length, not usually as cycles, because the extra tokens are operands of the same instructions.

Cases that are genuinely more verbose:

- **Merging predicates.** VMI `pmode` only zeroes inactive lanes. `sum_smaller_probs` (and the same pattern in the fused sampler) turns ASC `vadd(..., "MODE_MERGING")` into an add plus `vsel` so the accumulator is kept. The extra select is real work.
- **Predicate spill.** There is no `vpld` / `vpst`. A keep-mask that ASC stores with `pld`/`pst` has to be materialized as 0/1 and reloaded. The simulated sampler does not spill a mask; it keeps the predicate in a register. The warning stands for a full `fused_sample` mask round-trip.
- **Integer divide.** `V.vdiv` is f16/f32 only. `get_kv_index` `dflash` uses a block size of 8, so PTO takes the low bits with `vand` instead of ASC's integer `vdiv`. That is shorter, not longer. A non-power-of-two modulus would need a float divide and a truncating convert, which is the longer form.
- **Broadcast loads.** `vld(..., dist="BRC_B32")` is `vload` of one lane plus `vbrc`. PTOAS may fuse the pair. Until it does, the source has two calls where ASC has one. `swa_index` and `structure_output` are the clear examples. Their cycle ratios stay near 1.0 when the fuse happens, and are the place to look when a ratio drops below 0.8.

`gather` goes the other way: the ASC VF is a scalar `T.Parallel` fill, and PTO is one `vbrc` plus one `vstore` per row.

`dflash` head mean is the case that shows the width choice. Spelled as two 64-lane iterations, PTO took 108 cycles against ASC's 66 (0.61x) because each half-register convert is a separate `vload` plus `vcvt`. One 128-lane convert of the whole row is 71 cycles, 0.93x ASC, inside the normal band.

`norm` quant and `store_kv_cache` replace four half-register `vcmax` stores with one `vcmax(..., group=4)`. That is the shorter form, and both are PTO wins in `PERF.md` (1.44x and 1.49x).


