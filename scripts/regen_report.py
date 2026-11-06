"""Rebuild deepinfer_kernels/REPORT.md from data/*/*.jsonl. Newest row per family/name wins."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "deepinfer_kernels" / "REPORT.md"


def load_rows() -> list[dict]:
    newest: dict[tuple[str, str], dict] = {}
    for path in sorted((DATA / "asc").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            newest[(row["family"], row["name"])] = row
    return sorted(newest.values(), key=lambda row: (row["family"], row["name"]))


# Why a row stays under 0.95, or why it does not match. The case notes only
# describe the shape.
CAUSES = {
    "candidate/block_max": "Block max is a grouped vcmax over bf16 tiles. ASC reduces the same tile with packed vmax.",
    "bitmask/b512_v129280_ftp": "PTO broadcasts the mask with vload+vbrc and walks the left bits serially; ASC uses BRC_B32.",
    "candidate/restore_indices": "Restore gathers block ids with vgather; ASC uses vgather2.",
    "compress/decode": "Softmax sum is a float vdiv. ASC divides integers. The tile is unchanged.",
    "cp_mirror/prepare_b4096_c1_2": "PTO has no dynamic shift, so the binary search is straight-line constant shifts. TILE is unchanged.",
    "dflash/input_store": "64 serial f32 iterations. T.unroll(..., unroll_factor=) is rejected on Ascend.",
    "flash_mla/b256_k1024_full": "Same SIMT body. The two medians are a few tenths of a microsecond apart on a 7us kernel.",
    "fused_transform/norm_fp8_h8_d128_n1024": "FP8 scale packing is scalar vcvt/vselr per head. The rope-only row is the same schedule without that cast.",
    "fused_transform/norm_rope_h8_d128_n4096": "Same multi-core schedule. The extra time is the vmi lowering of the norm and rope VFs.",
    "gather/large_d128": "PTO rejects T.Parallel inside SimdVF, so the zero fill is a 256-lane vbrc+vstore.",
    "image/batch_downsample_concat": "Concat is a per-element VF. ASC moves the same tile with copies.",
    "image/batch_downsample_pool": "Pool reduction stays in the VF. ASC is closer to the copy bandwidth of the same tile.",
    "kv_index/paged_n1024_b128": "Paged gather is vgather. ASC uses vgather2. The SWA path, which is copies, matches.",
    "kv_index_cat/q16384_w128_k1024": "Explicitly unrolled VF on a bandwidth-bound index cat. The tile is unchanged.",
    "load_kv/v41_fp4": "Dequant is a 64-lane vselr table plus a grouped scale broadcast. ASC uses packed SIMD.",
    "load_kv/v41_fp8": "Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load.",
    "norm/bf16": "One row per VF. Multi-row plus the compensated rstd exceeds the layout search. Hidden 7168 is unchanged.",
    "norm/bf16_residual": "One row per VF, same layout-search limit as the plain bf16 row. Hidden 7168 is unchanged.",
    "sampler/get_accept_result": "Same SIMT accept scan. The medians differ by under a microsecond.",
    "sampler/remap_draft_topk": "Remap gathers with vgather. ASC uses vgather2.",
    "schedule/n1024_nextn7": "The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg.",
    "store_kv/bf16": "Same copy schedule. The medians differ by about 7us on a 100us kernel.",
    "store_kv/v41_fp4": "Quant spills the inverse, widens the LUT to 64 lanes for vselr, then stores with a narrow mask.",
    "store_kv/v41_fp8": "Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input.",
    "swa/n8192_w4096": "ASC is at memory bandwidth. The PTO index math stays in the VF, so the same tile is slower.",
    "fused_sample/b256_v1024": "ptoas rejects pto.vmi.extui unless both types are unsigned. The 0/1 prefix is built as uint32 in the source; task submission entered maintenance before that spelling was recompiled.",
    "fused_sample/b256_v1024_reject": "Same extui rejection as the non-reject row. The unsigned prefix spelling was not recompiled.",
    "sample/b256_v1024": "bisheng exits 139 (SIGSEGV) compiling the kernel, with the scheduler on or off.",
    "sample/b256_v129280": "Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch.",
    "norm/fp8_npc32_packed": "Measured max abs 32 on y. The grouped amax is a 128-lane vcmax; an unmasked store writes past the 4 live lanes. A 4-of-128 mask is in the source and was not re-run.",
}


def _cause(row: dict) -> str:
    return CAUSES.get(f"{row['family']}/{row['name']}", "")


def main() -> None:
    rows = load_rows()
    lines = [
        "# ASC vs PTO on-device report",
        "",
        "Kernels are the DeepInfer [`tilelang_ops/asc`](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/asc) set",
        "(DeepInfer `a679bb8`), launched on all vector cores.",
        "Toolchain: CANN 9.2.0, torch-npu 2.9.0.post6, ptoas 0.72, tilelang 0.1.15 (`pto-dev` `479d5cd1`).",
        "`PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144`.",
        "Device: Ascend950DT. `ratio = ASC_us / PTO_us`. A row is a perf miss when ratio < 0.95.",
        "Timing is `do_bench(..., backend='msprof')` median over `BENCH_REP` repeats after 5 warmup launches.",
        "",
        "## Shapes",
        "",
        "Each row is one case from DeepInfer's `@pytest.mark.benchmark` lists",
        "(DeepInfer `a679bb8`), not a serving trace and not every pytest combination.",
        "The sizes that carry the comparison are DeepSeek-class prefill shapes:",
        "RMSNorm is 32768 tokens by hidden 7168, the logit kernels use vocab 129280,",
        "and gather uses 131072 rows.",
        "",
        "Source lists:",
        "",
        "- Norm, including `32768 × 7168`: [norm.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/norm.py#L472)",
        "- KV cache benchmark length, 65536 tokens: [cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L388)",
        "- Gather `(n, d, m)`: [gather.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/gather.py#L98)",
        "- Candidate blocks, `max_model_len = 1 << 20`: [candidate_blocks.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py#L184)",
        "- Fused transform, captured 550B Q cases: [fused_transform.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/fused_transform.py#L673)",
        "- Sampling batch 256 or 4096 and vocab 129280 or 1024: [random_sampling.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L101)",
        "- Sampler vocab 129280: [sampler.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L634)",
        "- Image embeds: [image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L240)",
        "- Remaining kernels use the matching module under [tilelang_ops/](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops).",
        "",
        "## Smaller than the DeepInfer list",
        "",
        "Five rows use a smaller size than the published benchmark. The original size",
        "still needs a run wherever the note is marked TODO.",
        "",
        "- **KV load and store.** DeepInfer times every layout at 65536 tokens",
        "  ([cache_common.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L388)).",
        "  This report uses 8192 (bf16), 4096 (fp8 and fp4), and 1024 (mxfp8 index).",
        "  A 65536-token cache fits under the 512 MiB tensor cap, so the cut is a harness choice.",
        "  TODO: rerun each layout at 65536 tokens.",
        "- **Candidate block-max.** The test allocates `max_model_len = 1 << 20`",
        "  ([candidate_blocks.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/candidate_blocks.py#L184)).",
        "  At 256 rows that buffer is 512 MiB of bf16, which is the harness cap.",
        "  The case sets the kernel width to the valid length, 65536, the smallest `valid_len` in that list.",
        "  Tasks per row are `ceil(max_model_len / 16384)`, so this width is 1 task per row and `1 << 20` is 64.",
        "  TODO: rerun `max_model_len = 1 << 20` at a row count that stays under 512 MiB (one row is 2 MiB).",
        "- **Fused sample.** The sampling tests use vocab 129280 and 1024",
        "  ([random_sampling.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L101)).",
        "  Random sampling in this report has both. Fused sample has only 1024.",
        "  The kernel accepts 129280: it requires a multiple of 64 and chunks a vocab of 8192 or more by 8192.",
        "  TODO: add the vocab-129280 fused-sample row, including the reject specialization.",
        "- **Fused transform.** The captured cases are 64 heads by 512 and 32 heads by 128, with up to 32768 tokens",
        "  ([fused_transform.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/fused_transform.py#L673)).",
        "  These rows are 8 heads by 128 at 4096 and 1024 tokens.",
        "  The KV-cache store does not run: the kernel allows `store_cache` only when `num_heads == 1`, and these rows use 8 heads.",
        "  TODO: run a captured head shape (64×512 or 32×128) and a `num_heads == 1` row that stores the KV cache.",
        "- **Image pad and downsample.** `n_vit = 54` is the benchmark grid side, about 54×54 patches per image.",
        "  The rgb benchmark builds 50 images at that grid",
        "  ([image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L386)).",
        "  Downsample is `(8, 24)`, `(32, 54)`, and `(8, 96)`",
        "  ([image_embed_ops.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/image_embed_ops.py#L617)).",
        "  This report uses 8 images at `n_vit = 54`.",
        "  TODO: rgb at 50 images, and downsample at `(32, 54)` and `(8, 96)`.",
        "",
        "Kernels that finish in a few microseconds are launch-bound.",
        "Their ratio is not a saturated prefill measurement.",
        "",
        "| family | case | correct | ASC us | PTO us | ratio | ASC GB/s | PTO GB/s | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    misses = []
    errors = []
    for row in rows:
        if not row.get("correct"):
            errors.append(row)
            cause = _cause(row)
            first = (row.get("error") or "").splitlines()[0][:240]
            detail = " ".join(part for part in (cause, first) if part).replace("|", "/")
            lines.append(f"| {row['family']} | {row['name']} | no |  |  |  |  |  | {detail} |")
            continue
        ratio = row.get("ratio")
        flag = ""
        if ratio is not None and ratio < 0.95:
            flag = "perf"
            misses.append(row)
        note = " ".join(part for part in (flag, _cause(row) if flag else "", row.get("notes") or "", row.get("compared") or "") if part)
        lines.append(
            "| {family} | {name} | yes | {asc:.2f} | {pto:.2f} | {ratio:.3f} | {ag:.2f} | {pg:.2f} | {note} |".format(
                family=row["family"],
                name=row["name"],
                asc=row["asc_us"],
                pto=row["pto_us"],
                ratio=ratio,
                ag=row.get("asc_gbs") or 0,
                pg=row.get("pto_gbs") or 0,
                note=note.replace("|", "/"),
            )
        )
    lines.append("")
    lines.append(f"Rows: {len(rows)}. Correct: {len(rows) - len(errors)}. Perf misses (ratio < 0.95): {len(misses)}.")
    if misses:
        lines.append("")
        lines.append("## Perf misses")
        lines.append("")
        for row in misses:
            lines.append(
                f"- `{row['family']}/{row['name']}` ratio {row['ratio']:.3f}. {_cause(row)} {row.get('notes') or ''}"
            )
    if errors:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for row in errors:
            first = (row.get("error") or "").splitlines()[0][:400]
            lines.append(f"- `{row['family']}/{row['name']}`: {_cause(row)} {first}")
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
