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


# Why a row stays under 0.95, or why it does not compile. The case notes only
# describe the shape.
CAUSES = {
    "compress/decode": "Both sides divide the softmax sum in float. The tile is unchanged; the residual on this 9 us row is launch and measurement.",
    "cp_mirror/prepare_b4096_c1_2": "PTO has no dynamic shift, so the binary search is straight-line constant shifts.",
    "flash_mla/b256_k1024_full": "Same SIMT body. The rebench stays under 0.95 on a 7us kernel, so the residual is launch and measurement.",
    "fused_transform/norm_fp8_h8_d128_n1024": "The eight grouped maxima are parked with one 8-lane store and reloaded.",
    "fused_transform/store_mxfp8_h1_d128_n32768": "The store_cache arm writes each token through the paged slot_mapping; the quantized store into the MXFP8 index cache is 11 us behind on a 33 us row.",
    "gather/large_d128": "The residual on this 45 us row is launch and measurement.",
    "gather_engram/n1024_nextn5": "Same gather body; the residual on this 9 us row is launch and measurement.",
    "load_kv/v41_fp4": "Dequant reloads each group scale with a grouped vload plus vbrc. Batching the four groups into one barrier returned NaNs.",
    "load_kv/v41_fp8": "Dequant reloads the scale through a grouped vload+vbrc. ASC uses a broadcast load.",
    "norm/bf16_residual": "The tile's two rows share one VF: one vcadd per row, then one rstd. The square-sum is a one-lane store.",
    "sampler/get_accept_result": "Same SIMT accept scan. The rebench stays under 0.95, so the residual is launch and measurement.",
    "sampler/map_lookup": "3.5 us kernel, so the residual is launch and measurement.",
    "schedule/n1024_nextn7": "The rank is a full-vector broadcast of vcadd. A 1-lane vadd is not a legal vreg.",
    "state_cache/load_to_cache": "4.5 us kernel, so the residual is launch and measurement.",
    "store_kv/bf16": "Same copy schedule. The gap holds at the full 65536-token benchmark size.",
    "store_kv/v41_fp4": "The FP4 reciprocal is eight native-lane vsel, and the row's inverses share one grouped reload.",
    "store_kv/v41_fp8": "Quant spills the inverse and reloads it with a grouped vload+vbrc. A converted vector is not a legal vbrc group input.",
    "fused_sample/b256_v1024": "ptoas rejects pto.vmi.extui unless both types are unsigned. The uint32 prefix spelling is still rejected.",
    "fused_sample/b256_v1024_reject": "Same extui rejection as the non-reject row.",
    "fused_sample/b256_v129280": "Same extui rejection as the vocab-1024 row.",
    "fused_sample/b256_v129280_reject": "Same extui rejection as the vocab-1024 row.",
    "sample/b256_v1024": "bisheng exits 139 (SIGSEGV) compiling the kernel, scheduler on or off. Not a membar interaction.",
    "sample/b256_v129280": "Same bisheng exit 139 as the vocab-1024 row. Both vocabs fail in the compiler, before launch.",
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
        "Toolchain: CANN 9.2.0, torch-npu 2.9.0.post6, ptoas 0.76, tilelang 0.1.15+cpu.git59f3924e",
        "([tile-ai/tilelang](https://github.com/tile-ai/tilelang/tree/59f3924e85ea2f6d3ccec2f00c87044951a783f7) `59f3924`).",
        "`PTO_FLAGS=--pto-backend=vpto --pto-level=level3 --vmi-layout-max-frontier-entries=65536 --vmi-layout-max-transitions=262144 --enable-vecscope-mem-bar=false`.",
        "Device: Ascend950DT. `ratio = ASC_us / PTO_us`. A row is a perf miss when ratio < 0.95.",
        "Timing is `do_bench(..., backend='msprof')` median over `BENCH_REP` repeats after 5 warmup launches.",
        "",
        "## Shapes",
        "",
        "Each row is one case from DeepInfer's `@pytest.mark.benchmark` lists",
        "(DeepInfer `a679bb8`), not a serving trace and not every pytest combination.",
        "The sizes that carry the comparison are DeepSeek-class prefill shapes:",
        "RMSNorm is 32768 tokens by hidden 7168, the logit kernels use vocab 129280,",
        "gather uses 131072 rows, KV load and store run every layout at 65536 tokens,",
        "and candidate block-max scans the full `1 << 20` model length at 256 rows.",
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
