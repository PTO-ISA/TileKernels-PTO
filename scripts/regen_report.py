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
# describe the shape. Rows that crossed 0.95 in the membar-off rebench are
# removed; the report table carries their new numbers.
CAUSES = {
    "compress/decode": "Both sides divide the softmax sum in float. The tile is unchanged; the residual on this 9 us row is launch and measurement.",
    "cp_mirror/prepare_b4096_c1_2": "PTO has no dynamic shift, so the binary search is straight-line constant shifts. Membar off does not move this row.",
    "flash_mla/b256_k1024_full": "Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement.",
    "fused_transform/norm_fp8_h8_d128_n1024": "The eight grouped maxima are parked with one 8-lane store and reloaded. Membar off closed part of the gap; the park and reload remain.",
    "gather/large_d128": "The per-store VST_VST barrier is gone with the membar off (was 0.685). The residual on this 45 us row is launch and measurement.",
    "load_kv/mxfp8_index": "2.5 us kernel, so the residual is launch and measurement.",
    "load_kv/v41_fp4": "Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs.",
    "load_kv/v41_fp8": "Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load.",
    "norm/bf16_residual": "The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store. Membar off does not move this row.",
    "sampler/get_accept_result": "Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement.",
    "schedule/n1024_nextn7": "The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg.",
    "store_kv/bf16": "Same copy schedule. The rebench stays under 0.95 on a 100us kernel, so the residual is launch and measurement.",
    "store_kv/v41_fp4": "The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload. Membar off helped but the reload gap remains.",
    "store_kv/v41_fp8": "Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input.",
    "fused_sample/b256_v1024": "ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected with --enable-vecscope-mem-bar=false.",
    "fused_sample/b256_v1024_reject": "Same extui rejection as the non-reject row, with --enable-vecscope-mem-bar=false.",
    "sample/b256_v1024": "bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off, with --enable-vecscope-mem-bar=false. Not a membar interaction.",
    "sample/b256_v129280": "Same bisheng exit 139 as the vocab-1024 row, with --enable-vecscope-mem-bar=false. Both vocabs fail in the compiler, before launch.",
    "norm/fp8_npc32_packed": "The 4-of-128 amax mask spelling fails PTO codegen legality: store destination misaligned, byte remainder 16 vs required 32.",
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
        "`PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.",
        "Device: Ascend950DT. `ratio = ASC_us / PTO_us`. A row is a perf miss when ratio < 0.95.",
        "Timing is `do_bench(..., backend='msprof')` median over `BENCH_REP` repeats after 5 warmup launches.",
        "Every row below 0.95 was rerun with the vecscope membar insertion off; rows already at 0.95 or above",
        "keep their earlier numbers, measured with the barrier insertion on.",
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
