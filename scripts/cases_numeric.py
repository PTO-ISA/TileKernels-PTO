"""ASC vs PTO numeric cases. One production shape per kernel specialization.

prepare() builds device tensors, compiles the ASC and PTO factories, clones
writable buffers, runs each kernel once, and returns relaunch closures.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BenchCase

import torch

from vllm_kernels.asc import candidate_blocks as asc_candidate
from vllm_kernels.asc import compress as asc_compress
from vllm_kernels.asc import dflash as asc_dflash
from vllm_kernels.asc import image_embed_ops as asc_image
from vllm_kernels.asc import load_paged_kv_cache as asc_load
from vllm_kernels.asc import norm as asc_norm
from vllm_kernels.asc import store_kv_cache as asc_store
from vllm_kernels.common import BF16_KV_FORMAT, MXFP8_INDEX_KV_FORMAT, V41_FORMAT, V41_FP4_FORMAT
from vllm_kernels.pto import candidate_blocks as pto_candidate
from vllm_kernels.pto import compress as pto_compress
from vllm_kernels.pto import dflash as pto_dflash
from vllm_kernels.pto import image_embed_ops as pto_image
from vllm_kernels.pto import load_paged_kv_cache as pto_load
from vllm_kernels.pto import norm as pto_norm
from vllm_kernels.pto import store_kv_cache as pto_store

# Test list in DeepInfer tilelang_ops/norm.py (_HIDDEN_SIZES and the benchmark
# grid). 7168 is the largest hidden, and it is a multiple of 128.
_NORM_HIDDEN = 7168
_NORM_TOKENS = 32768
_CAP_BYTES = 512 << 20
_TENSOR_LIMIT = 256 << 20

# Kernels requested but not emitted. Empty: every specialization below is built.
SKIPPED: list[str] = []


def _device() -> torch.device:
    import torch_npu  # noqa: F401

    dev = int(os.environ.get("BENCH_DEVICES", "1"))
    torch.npu.set_device(dev)
    return torch.device("npu", dev)


def _cap(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.nbytes <= _CAP_BYTES, (tuple(tensor.shape), tensor.dtype, tensor.nbytes)
    return tensor


def _fork(tensor: torch.Tensor) -> torch.Tensor:
    """Copy that keeps strides. clone() would densify an as_strided scale view."""
    out = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()), dtype=tensor.dtype, device=tensor.device)
    out.copy_(tensor)
    return out


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.nbytes for tensor in tensors)


def _launch(kernel, args):
    def fn() -> None:
        kernel(*args)

    return fn


def _run(asc_kernel, pto_kernel, asc_args, pto_args):
    asc_fn = _launch(asc_kernel, asc_args)
    pto_fn = _launch(pto_kernel, pto_args)
    asc_fn()
    pto_fn()
    torch.npu.synchronize()
    return asc_fn, pto_fn


def _compare_float8(tensor: torch.Tensor) -> torch.Tensor:
    # CPU allclose does not implement float8. Snapshot to fp32 after the launch.
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.float()
    return tensor


def _cuseqlen(seqlens: torch.Tensor) -> torch.Tensor:
    head = torch.zeros(1, dtype=torch.int32)
    return torch.cumsum(torch.cat((head, seqlens.to(torch.int32))), dim=0).to(torch.int32)


def _compressed_meta(seqlen_q: torch.Tensor, seqlen_k: torch.Tensor, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    new_blocks = torch.clamp(seqlen_k // ratio - (seqlen_k - seqlen_q) // ratio, min=1)
    cu = _cuseqlen(new_blocks)
    total_c = int(cu[-1])
    mapping = torch.zeros(total_c, dtype=torch.int32)
    for seq in range(1, seqlen_q.shape[0] + 1):
        end = int(cu[seq])
        if end < total_c:
            mapping[end] += 1
    return cu, torch.cumsum(mapping, dim=0).to(torch.int32)


def _norm_case(name: str, *, has_residual: bool, fp8: bool) -> BenchCase:
    hidden = _NORM_HIDDEN
    num_tokens = _NORM_TOKENS

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        x = _cap(torch.randn(1, num_tokens, hidden, dtype=torch.bfloat16, device=dev))
        weight = torch.empty(hidden, dtype=torch.bfloat16, device=dev)
        weight.normal_(mean=1.0, std=0.1)
        residual = _cap(torch.randn_like(x)) if has_residual else None
        y_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        y = torch.empty(1, num_tokens, hidden, dtype=y_dtype, device=dev)
        residual_out = torch.empty_like(x) if has_residual else None
        stride_s = int(x.stride(1))
        if fp8:
            # Ascend packed ue8m0 is int16 (2 scales per element). Col-major
            # packed layout matches norm_tl's transpose_sf strides. Shared sf
            # is not supported (shared_sf_block_m must be 0). npc is 32.
            num_groups = hidden // 32
            packs = num_groups // 2
            storage = torch.empty(1, packs, num_tokens, dtype=torch.int16, device=dev)
            sf = storage.view(torch.uint8).as_strided(
                (1, num_tokens, packs, 2),
                (num_groups * num_tokens, 2, num_tokens * 2, 1),
            )
            sf.zero_()
            assert sf.shape == (1, num_tokens, num_groups // 2, 2)
            assert sf.stride() == (num_groups * num_tokens, 2, num_tokens * 2, 1)
        else:
            sf = None
        asc_y, pto_y = _fork(y), _fork(y)
        asc_residual_out = _fork(residual_out) if residual_out is not None else None
        pto_residual_out = _fork(residual_out) if residual_out is not None else None
        asc_sf = _fork(sf) if sf is not None else None
        pto_sf = _fork(sf) if sf is not None else None

        def factory(mod):
            return mod.norm_tl(
                hidden,
                "bfloat16",
                "float8_e4m3fn" if fp8 else "bfloat16",
                "float",
                has_residual,
                1e-6,
                1.5,
                fp8,
                32 if fp8 else 0,
                fp8,  # use_tma_aligned_col_major_sf
                fp8,  # round_sf
                fp8,  # use_packed_ue8m0; Ascend keeps this path, skips shared sf
                False,  # return_origin_out
                2,  # xdim: wrapper uses 2 when the logical batch is 1
                0,  # shared_sf_block_m
                stride_s,
                stride_s,
                stride_s,
                stride_s,
                stride_s,
                stride_s,
            )

        def args(out, residual_buf, scale):
            return (
                x,
                out,
                weight,
                residual if residual is not None else x,
                residual_buf if residual_buf is not None else x,
                scale if scale is not None else x,
                x,
                x,
            )

        asc_fn, pto_fn = _run(
            factory(asc_norm),
            factory(pto_norm),
            args(asc_y, asc_residual_out, asc_sf),
            args(pto_y, pto_residual_out, pto_sf),
        )
        compared = [("y", _compare_float8(asc_y), _compare_float8(pto_y))]
        traffic = [x, weight, asc_y]
        if has_residual:
            compared.append(("residual_out", asc_residual_out, pto_residual_out))
            traffic.extend((residual, asc_residual_out))
        if fp8:
            compared.append(("sf", asc_sf, pto_sf))
            traffic.append(asc_sf)
        return asc_fn, pto_fn, compared, _bytes(*traffic)

    return BenchCase(
        "norm",
        name,
        prepare,
        exact=False,
        notes=f"tokens={num_tokens} hidden={hidden} residual={has_residual} fp8={fp8}",
    )


def _candidate_block_max_case() -> BenchCase:
    num_rows, valid_len, block_size = 256, 65536, 16
    # Kernel width is max_model_len. The bench fixes it at 1<<20 even for a
    # short valid_len; that row is 512MiB. Use valid_len as the width.
    max_model_len = valid_len

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        logits = _cap(torch.randn(num_rows, max_model_len, dtype=torch.bfloat16, device=dev))
        valid_lens = torch.full((num_rows,), valid_len, dtype=torch.int32, device=dev)
        num_blocks = (max_model_len + block_size - 1) // block_size
        align = 1024 // logits.element_size()
        score_stride = (num_blocks + align - 1) // align * align
        scores = torch.zeros(num_rows, score_stride, dtype=torch.bfloat16, device=dev)[:, :num_blocks]
        valid_blocks_size = (num_rows + 3) // 4 * 4
        valid_blocks = torch.zeros(valid_blocks_size, dtype=torch.int32, device=dev)
        asc_scores, pto_scores = _fork(scores), _fork(scores)
        asc_valid, pto_valid = _fork(valid_blocks), _fork(valid_blocks)
        # out_idx=[4, 5, 6] allocates task_records, num_tasks, core_offsets.
        asc_fn, pto_fn = _run(
            asc_candidate.candidate_block_max_tl(max_model_len, block_size),
            pto_candidate.candidate_block_max_tl(max_model_len, block_size),
            (logits, valid_lens, asc_scores, asc_valid),
            (logits, valid_lens, pto_scores, pto_valid),
        )
        compared = [
            ("block_scores", asc_scores, pto_scores),
            ("valid_blocks", asc_valid[:num_rows], pto_valid[:num_rows]),
        ]
        return asc_fn, pto_fn, compared, _bytes(logits, valid_lens, asc_scores, asc_valid[:num_rows])

    return BenchCase(
        "candidate",
        "block_max",
        prepare,
        exact=False,
        notes=f"rows={num_rows} valid_len={valid_len} block={block_size}",
    )


def _candidate_compact_case() -> BenchCase:
    num_rows, topk_blocks, block_size = 128, 1024, 16

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        assert 0 < topk_blocks <= 4096 and block_size in (8, 16)
        blocks = torch.arange(topk_blocks, dtype=torch.int32, device=dev).repeat(num_rows, 1)
        blocks[:, -1] = -1
        assert blocks.stride(1) == 1
        valid_lens = torch.full((num_rows,), topk_blocks * block_size, dtype=torch.int32, device=dev)
        compact = torch.empty(num_rows, dtype=torch.int32, device=dev)
        asc_out, pto_out = _fork(compact), _fork(compact)
        asc_fn, pto_fn = _run(
            asc_candidate.get_candidate_compact_lens_tl(topk_blocks, block_size),
            pto_candidate.get_candidate_compact_lens_tl(topk_blocks, block_size),
            (blocks, valid_lens, asc_out),
            (blocks, valid_lens, pto_out),
        )
        return asc_fn, pto_fn, [("compact_lens", asc_out, pto_out)], _bytes(blocks, valid_lens, asc_out)

    return BenchCase(
        "candidate",
        "compact_lens",
        prepare,
        notes=f"rows={num_rows} topk_blocks={topk_blocks} block={block_size}",
    )


def _candidate_restore_case() -> BenchCase:
    num_rows, topk_tokens, topk_blocks, block_size = 128, 1024, 128, 16

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        assert 0 < topk_blocks <= 4096 and block_size in (8, 16)
        indices = torch.randint(0, topk_blocks * block_size, (num_rows, topk_tokens), dtype=torch.int32, device=dev)
        indices[:, ::17] = -1
        blocks = torch.arange(topk_blocks, dtype=torch.int32, device=dev).repeat(num_rows, 1)
        blocks[:, ::11] = -1
        offsets = torch.randint(0, 1 << 20, (num_rows,), dtype=torch.int32, device=dev)
        assert indices.stride(0) % 8 == 0 and blocks.stride(0) % 8 == 0
        assert indices.stride(1) == 1 and blocks.stride(1) == 1
        asc_idx, pto_idx = _fork(indices), _fork(indices)
        asc_fn, pto_fn = _run(
            asc_candidate.restore_candidate_indices_inplace_tl(topk_tokens, topk_blocks, block_size, True),
            pto_candidate.restore_candidate_indices_inplace_tl(topk_tokens, topk_blocks, block_size, True),
            (asc_idx, blocks, offsets),
            (pto_idx, blocks, offsets),
        )
        return asc_fn, pto_fn, [("compact_indices", asc_idx, pto_idx)], _bytes(indices, blocks, offsets, asc_idx)

    return BenchCase(
        "candidate",
        "restore_indices",
        prepare,
        notes=f"rows={num_rows} topk_tokens={topk_tokens} topk_blocks={topk_blocks} block={block_size} offset=1",
    )


def _compress_prefill_case() -> BenchCase:
    # Bench draws qlen in [1, 8192). Full 8192 keeps each tensor under 256MB
    # (score/latent are 128MB). qstart is not ratio-aligned so one window row
    # is read from state_cache.
    ratio, dim, num_seq, qlen, extra = 2, 512, 8, 8192, 1025

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        seqlen_q = torch.full((num_seq,), qlen, dtype=torch.int32)
        seqlen_k = seqlen_q + extra
        cu_q = _cuseqlen(seqlen_q)
        cu_k = _cuseqlen(seqlen_k)
        cu_c, mapping = _compressed_meta(seqlen_q, seqlen_k, ratio)
        total_q = int(cu_q[-1])
        total_c = int(mapping.shape[0])
        score = _cap(torch.randn(total_q, dim, dtype=torch.float32, device=dev))
        latent = _cap(torch.randn(total_q, dim, dtype=torch.float32, device=dev))
        assert score.nbytes <= _TENSOR_LIMIT and latent.nbytes <= _TENSOR_LIMIT
        num_slots, state_len = num_seq * 2, 138
        state = torch.randn(num_slots, state_len, 2, dim, dtype=torch.float32, device=dev)
        assert state.stride(1) == 2 * dim and state.stride(0) % 8 == 0
        assert score.stride(0) % 8 == 0 and latent.stride(0) % 8 == 0
        slots = torch.arange(num_seq, dtype=torch.int32, device=dev)
        cu_q, cu_k, cu_c = (t.to(dev) for t in (cu_q, cu_k, cu_c))
        mapping = mapping.to(dev)
        mirror = torch.arange(total_q, dtype=torch.int32, device=dev)
        kv = torch.zeros(total_c, dim, dtype=torch.bfloat16, device=dev)
        asc_state, pto_state = _fork(state), _fork(state)
        asc_kv, pto_kv = _fork(kv), _fork(kv)

        def args(state_buf, kv_buf):
            return (score, latent, state_buf, slots, mapping, cu_q, cu_k, cu_c, mirror, kv_buf)

        asc_fn, pto_fn = _run(
            asc_compress.compress_and_update_state_prefill_tl(dim, ratio, False),
            pto_compress.compress_and_update_state_prefill_tl(dim, ratio, False),
            args(asc_state, asc_kv),
            args(pto_state, pto_kv),
        )
        compared = [("kv_compressed", asc_kv, pto_kv), ("state_cache", asc_state, pto_state)]
        return asc_fn, pto_fn, compared, _bytes(score, latent, state, kv)

    return BenchCase(
        "compress",
        "prefill",
        prepare,
        exact=False,
        notes=f"ratio={ratio} dim={dim} has_index=0 seqs={num_seq} qlen={qlen}",
    )


def _compress_decode_case() -> BenchCase:
    ratio, dim, num_seq, nextn = 2, 512, 64, 6

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        total_q = num_seq * nextn
        score = torch.randn(total_q, dim, dtype=torch.float32, device=dev)
        latent = torch.randn(total_q, dim, dtype=torch.float32, device=dev)
        num_slots, state_len = num_seq * 2, 138
        state = torch.randn(num_slots, state_len, 2, dim, dtype=torch.float32, device=dev)
        assert state.stride(1) == 2 * dim and state.stride(0) % 8 == 0
        assert score.stride(0) % 8 == 0
        # First token of each sequence ends a compress block (bench all_seq_do_compress).
        qstart = torch.full((num_seq,), 4096, dtype=torch.int32, device=dev)
        qstart = qstart // ratio * ratio + ratio - 1
        offsets = torch.arange(nextn, dtype=torch.int32, device=dev).repeat(num_seq)
        context = qstart.repeat_interleave(nextn) + offsets + 1
        slots = torch.arange(num_seq, dtype=torch.int32, device=dev).repeat_interleave(nextn)
        kv = torch.zeros(total_q, dim, dtype=torch.bfloat16, device=dev)
        asc_state, pto_state = _fork(state), _fork(state)
        asc_kv, pto_kv = _fork(kv), _fork(kv)

        def args(state_buf, kv_buf):
            return (score, latent, state_buf, slots, context, kv_buf)

        asc_fn, pto_fn = _run(
            asc_compress.compress_and_update_state_decode_tl(dim, ratio),
            pto_compress.compress_and_update_state_decode_tl(dim, ratio),
            args(asc_state, asc_kv),
            args(pto_state, pto_kv),
        )
        compared = [("kv_compressed", asc_kv, pto_kv), ("state_cache", asc_state, pto_state)]
        return asc_fn, pto_fn, compared, _bytes(score, latent, state, kv)

    return BenchCase(
        "compress",
        "decode",
        prepare,
        exact=False,
        notes=f"ratio={ratio} dim={dim} seqs={num_seq} nextn={nextn}",
    )


def _confidence_case() -> BenchCase:
    n, n_embed = 4097, 4096

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        assert (n_embed + 0) <= 11000 and n_embed % 64 == 0
        hidden = torch.randn(n, n_embed, dtype=torch.bfloat16, device=dev) / 10
        assert hidden.stride(0) % 8 == 0
        # has_markov_emb is false; x1 is still in the signature with width 1.
        markov = torch.zeros(n, 1, dtype=torch.bfloat16, device=dev)
        weight = torch.randn(n_embed, dtype=torch.bfloat16, device=dev) / 10
        out = torch.empty(n, dtype=torch.float32, device=dev)
        asc_out, pto_out = _fork(out), _fork(out)
        temp = 0.85
        asc_fn, pto_fn = _run(
            asc_dflash.confidence_head_fused_tl(n_embed, 0, False),
            pto_dflash.confidence_head_fused_tl(n_embed, 0, False),
            (hidden, markov, weight, asc_out, temp),
            (hidden, markov, weight, pto_out, temp),
        )
        return asc_fn, pto_fn, [("out", asc_out, pto_out)], _bytes(hidden, weight, asc_out)

    return BenchCase("dflash", "confidence_head", prepare, exact=False, notes=f"N={n} n_embed={n_embed} markov=0")


def _dflash_store_case() -> BenchCase:
    n, n_embed, hc_mult = 128, 4096, 4

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        assert n_embed <= 8192 and n_embed % 128 == 0
        hidden = torch.randn(n, hc_mult, n_embed, dtype=torch.bfloat16, device=dev)
        out = torch.empty(n, n_embed, dtype=torch.bfloat16, device=dev)
        assert out.stride(0) % 4 == 0
        asc_out, pto_out = _fork(out), _fork(out)
        asc_fn, pto_fn = _run(
            asc_dflash.dflash_input_store_tl(n_embed, hc_mult),
            pto_dflash.dflash_input_store_tl(n_embed, hc_mult),
            (hidden, asc_out),
            (hidden, pto_out),
        )
        return asc_fn, pto_fn, [("out", asc_out, pto_out)], _bytes(hidden, asc_out)

    return BenchCase(
        "dflash",
        "input_store",
        prepare,
        exact=False,
        notes=f"n={n} n_embed={n_embed} hc_mult={hc_mult}",
    )


def _replace_embeds_case() -> BenchCase:
    num_tokens, hidden = 4096, 4096

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        assert hidden % 128 == 0
        embeds = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
        image_id = 100
        ids = torch.zeros(num_tokens, dtype=torch.int32, device=dev)
        ids[0] = 101
        ids[1:513] = image_id
        ids[513] = 102
        ids[514] = 103
        ids[600:856] = image_id
        img_tokens = int((ids == image_id).sum())
        images = torch.randn(img_tokens, hidden, dtype=torch.bfloat16, device=dev)
        pad_ids = torch.tensor([101, 102, 103], dtype=torch.int32, device=dev)
        pad_embeds = torch.randn(pad_ids.shape[0], hidden, dtype=torch.bfloat16, device=dev)
        offset = torch.cumsum((ids == image_id).to(torch.int32), dim=0).to(torch.int32)
        asc_embeds, pto_embeds = _fork(embeds), _fork(embeds)

        def args(buf):
            return (buf, ids, images, image_id, offset, pad_ids, pad_embeds)

        asc_fn, pto_fn = _run(
            asc_image.replace_input_embeds_tl(hidden),
            pto_image.replace_input_embeds_tl(hidden),
            args(asc_embeds),
            args(pto_embeds),
        )
        return asc_fn, pto_fn, [("input_embeds", asc_embeds, pto_embeds)], _bytes(embeds, images, pad_embeds)

    return BenchCase("image", "replace_embeds", prepare, exact=False, notes=f"tokens={num_tokens} hidden={hidden}")


def _rgb_case() -> BenchCase:
    patch_size, num_imgs, n_vit = 14, 8, 54

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        raw_chunks = []
        cu_raw = [0]
        reals = []
        crops = []
        for i in range(num_imgs):
            width = max(2, n_vit - (i % 5))
            height = max(2, n_vit - 2 - ((i * 3) % 7))
            if width == height:
                height = max(2, height - 1)
            crops.append((width, height))
            grid_w, grid_h = width * patch_size, height * patch_size
            real_w = max(1, grid_w - (3 + i % 5))
            real_h = max(1, grid_h - (2 + (i * 3) % 7))
            pixels = torch.randint(0, 256, (real_h * real_w * 3,), dtype=torch.uint8)
            raw_chunks.append(pixels)
            cu_raw.append(cu_raw[-1] + pixels.numel())
            reals.append((real_w, real_h))
        raw = torch.cat(raw_chunks).to(dev)
        cu_raw_t = torch.tensor(cu_raw, dtype=torch.int64, device=dev)
        real_t = torch.tensor(reals, dtype=torch.int32, device=dev)
        crop_t = torch.tensor(crops, dtype=torch.int32, device=dev)
        seqlens = crop_t[:, 0] * crop_t[:, 1]
        cu_vit = torch.zeros(num_imgs + 1, dtype=torch.int32, device=dev)
        cu_vit[1:] = torch.cumsum(seqlens, dim=0).to(torch.int32)
        total = int(cu_vit[-1])
        out = torch.zeros(total, 3, patch_size, patch_size, dtype=torch.bfloat16, device=dev)
        asc_out, pto_out = _fork(out), _fork(out)
        mean, std, pad_color = 0.5, 0.5, float(int(0.5 * 255))

        def args(buf):
            return (raw, cu_raw_t, real_t, crop_t, cu_vit, mean, std, pad_color, buf)

        asc_fn, pto_fn = _run(
            asc_image.rgb_pad_to_patches_tl(patch_size),
            pto_image.rgb_pad_to_patches_tl(patch_size),
            args(asc_out),
            args(pto_out),
        )
        return asc_fn, pto_fn, [("output", asc_out, pto_out)], _bytes(raw, asc_out)

    return BenchCase(
        "image",
        "rgb_pad",
        prepare,
        exact=False,
        notes=f"patch={patch_size} imgs={num_imgs} n_vit={n_vit} padded=1",
    )


def _downsample_case(do_downsample: bool) -> BenchCase:
    ratio, num_imgs, n_vit, dim = 3, 8, 54, 1024

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        crops = torch.full((num_imgs, 2), n_vit, dtype=torch.int64, device=dev)
        seqlens = crops[:, 0] * crops[:, 1]
        cu = torch.zeros(num_imgs + 1, dtype=torch.int64, device=dev)
        cu[1:] = torch.cumsum(seqlens, dim=0)
        down_w = (crops[:, 0] + ratio - 1) // ratio
        down_h = (crops[:, 1] + ratio - 1) // ratio
        down_lens = down_w * down_h
        cu_down = torch.zeros_like(cu)
        cu_down[1:] = torch.cumsum(down_lens, dim=0)
        num_down = int(down_lens.sum())
        image_id = torch.searchsorted(cu_down, torch.arange(num_down, device=dev), right=True) - 1
        feats = torch.randn(int(seqlens.sum()), dim, dtype=torch.bfloat16, device=dev)
        embeds = torch.zeros(num_down, dim, ratio, ratio, dtype=torch.bfloat16, device=dev)
        pooled = torch.zeros(num_down, dim, dtype=torch.bfloat16, device=dev)
        asc_embeds, pto_embeds = _fork(embeds), _fork(embeds)
        asc_pool, pto_pool = _fork(pooled), _fork(pooled)

        def args(embed_buf, pool_buf):
            return (feats, image_id, crops, cu_down, cu, embed_buf, pool_buf)

        asc_fn, pto_fn = _run(
            asc_image.batch_downsample_tl(ratio, dim, do_downsample, torch.bfloat16),
            pto_image.batch_downsample_tl(ratio, dim, do_downsample, torch.bfloat16),
            args(asc_embeds, asc_pool),
            args(pto_embeds, pto_pool),
        )
        if do_downsample:
            compared = [("images_embeds_sum", asc_pool, pto_pool)]
            traffic = _bytes(feats, asc_pool)
        else:
            compared = [("images_embeds", asc_embeds, pto_embeds)]
            traffic = _bytes(feats, asc_embeds)
        return asc_fn, pto_fn, compared, traffic

    tag = "pool" if do_downsample else "concat"
    return BenchCase(
        "image",
        f"batch_downsample_{tag}",
        prepare,
        exact=False,
        notes=f"imgs={num_imgs} n_vit={n_vit} dim={dim} ratio={ratio} downsample={int(do_downsample)}",
    )


def _paged_storage(layout, num_blocks: int, block_size: int, device: torch.device):
    alignment = math.lcm(32, layout.bytes_per_token) if layout.token_interleaved else 32
    block_bytes = block_size * layout.bytes_per_token
    block_stride = (block_bytes + alignment - 1) // alignment * alignment
    storage = torch.empty((num_blocks, block_stride), dtype=torch.uint8, device=device)
    return storage, block_stride


def _interleaved_records(storage: torch.Tensor, block_stride: int, block_size: int, bytes_per_token: int) -> torch.Tensor:
    return torch.as_strided(
        storage,
        (storage.shape[0], block_size, bytes_per_token),
        (block_stride, bytes_per_token, 1),
    )


def _store_case(name: str, layout, num_tokens: int) -> BenchCase:
    block_size = 64

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        num_blocks = (num_tokens + block_size - 1) // block_size
        kv = _cap(torch.randn(num_tokens, layout.num_elements, dtype=torch.bfloat16, device=dev))
        slots = torch.arange(num_tokens, dtype=torch.int32, device=dev)
        if layout.quantized_dtype is None:
            cache = torch.zeros(num_blocks, block_size, layout.num_elements, dtype=torch.bfloat16, device=dev)
            asc_cache, pto_cache = _fork(cache), _fork(cache)
            asc_kernel = asc_store.store_kv_cache_tl(layout, block_size)
            pto_kernel = pto_store.store_kv_cache_tl(layout, block_size)
        else:
            assert layout.sf_format != "fp32" and layout.bf16_elements == 0
            storage, block_stride = _paged_storage(layout, num_blocks, block_size, dev)
            storage.zero_()
            cache = _interleaved_records(storage, block_stride, block_size, layout.bytes_per_token)
            assert cache.shape[-1] == layout.quantized_bytes + layout.sf_bytes
            asc_cache, pto_cache = _fork(cache), _fork(cache)
            asc_kernel = asc_store.store_kv_cache_tl(layout, block_size)
            pto_kernel = pto_store.store_kv_cache_tl(layout, block_size)
        asc_fn, pto_fn = _run(
            asc_kernel,
            pto_kernel,
            (kv, asc_cache, slots),
            (kv, pto_cache, slots),
        )
        compared = [("kv_cache", asc_cache, pto_cache)]
        return asc_fn, pto_fn, compared, _bytes(kv, asc_cache, slots)

    # exact is per case; quant caches are uint8, BF16 caches are float.
    exact = layout.quantized_dtype is not None
    return BenchCase(
        "store_kv",
        name,
        prepare,
        exact=exact,
        notes=f"block={block_size} tokens={num_tokens} layout={layout.note}",
    )


def _aligned_segments(num_tokens: int, block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = (num_tokens + block_size - 1) // block_size
    src = torch.arange(num_blocks, dtype=torch.int32, device=device)
    length = (num_tokens - src * block_size).clamp(max=block_size)
    return torch.stack((src, torch.zeros_like(src), length, src * block_size), dim=1)


def _load_views(layout, storage: torch.Tensor, block_stride: int, block_size: int):
    num_blocks = storage.shape[0]
    data_bytes = layout.quantized_bytes
    if layout.token_interleaved:
        records = _interleaved_records(storage, block_stride, block_size, layout.bytes_per_token)
        data = records[..., :data_bytes]
        sf = records[..., data_bytes:]
    else:
        data = torch.as_strided(storage, (num_blocks, block_size, data_bytes), (block_stride, data_bytes, 1))
        sf_bytes = layout.sf_bytes
        # Index scales are one int32 per token (4 ue8m0 bytes).
        assert sf_bytes == 4 and block_stride % 4 == 0 and (block_size * data_bytes) % 4 == 0
        sf = torch.as_strided(
            storage.view(torch.int32),
            (num_blocks, block_size, 1),
            (block_stride // 4, sf_bytes // 4, 1),
            storage_offset=(block_size * data_bytes) // 4,
        )
    if layout.do_cast or layout.token_interleaved:
        if layout.quantized_dtype == "e2m1":
            data = data.view(torch.int8)
        else:
            data = data.view(torch.float8_e4m3fn)
        if layout.sf_format == "e4m3":
            sf = sf.view(torch.float8_e4m3fn)
    else:
        data = data.view(torch.float8_e4m3fn)
    return data, sf


def _load_case(name: str, layout, num_tokens: int) -> BenchCase:
    block_size = 64

    def prepare():
        dev = _device()
        torch.manual_seed(0)
        num_blocks = (num_tokens + block_size - 1) // block_size
        segments = _aligned_segments(num_tokens, block_size, dev)
        if layout.quantized_dtype is None:
            cache = torch.randn(num_blocks, block_size, layout.num_elements, dtype=torch.bfloat16, device=dev)
            assert cache.stride(0) % 16 == 0 and cache.stride(1) % 16 == 0
            out = torch.empty(num_tokens, layout.num_elements, dtype=torch.bfloat16, device=dev)
            asc_out, pto_out = _fork(out), _fork(out)
            asc_kernel = asc_load.load_paged_kv_cache_tl(layout, block_size)
            pto_kernel = pto_load.load_paged_kv_cache_tl(layout, block_size)
            asc_fn, pto_fn = _run(
                asc_kernel,
                pto_kernel,
                (segments, asc_out, asc_out, cache, cache),
                (segments, pto_out, pto_out, cache, cache),
            )
            return asc_fn, pto_fn, [("kv", asc_out, pto_out)], _bytes(cache, asc_out, segments)

        storage, block_stride = _paged_storage(layout, num_blocks, block_size, dev)
        storage.copy_(torch.randint(0, 256, storage.shape, dtype=torch.uint8, device=dev))
        data, sf = _load_views(layout, storage, block_stride, block_size)
        kwargs = {}
        if layout.do_cast and layout.quantized_dtype == "e2m1":
            kwargs = {
                "fp4_value_block_stride": data.stride(0) * 2,
                "fp4_value_stride": data.stride(1) * 2,
                "fp4_scale_stride": sf.stride(1),
            }
            assert kwargs["fp4_value_block_stride"] % 2 == 0 and kwargs["fp4_value_stride"] % 2 == 0
            assert data.stride(0) % 32 == 0 and sf.stride(0) % 32 == 0
            assert data.stride(1) % layout.stride_alignment == 0
        elif layout.do_cast:
            assert data.stride(0) % 32 == 0 and sf.stride(0) % 32 == 0
            assert data.stride(1) % layout.stride_alignment == 0
        else:
            assert data.stride(0) % 32 == 0 and data.stride(1) % 32 == 0
            assert sf.stride(0) % 8 == 0 and sf.shape[-1] == 1
        if layout.do_cast or layout.quantized_dtype is None:
            out = torch.empty(num_tokens, layout.num_elements, dtype=torch.bfloat16, device=dev)
            asc_out, pto_out = _fork(out), _fork(out)
            asc_sf = pto_sf = None
        else:
            out = torch.empty(num_tokens, layout.quantized_bytes, dtype=torch.float8_e4m3fn, device=dev)
            scale = torch.empty(num_tokens, dtype=torch.int32, device=dev)
            asc_out, pto_out = _fork(out), _fork(out)
            asc_sf, pto_sf = _fork(scale), _fork(scale)
        asc_kernel = asc_load.load_paged_kv_cache_tl(layout, block_size, **kwargs)
        pto_kernel = pto_load.load_paged_kv_cache_tl(layout, block_size, **kwargs)

        def args(kv_buf, sf_buf):
            return (segments, kv_buf, sf_buf if sf_buf is not None else kv_buf, data, sf)

        asc_fn, pto_fn = _run(
            asc_kernel,
            pto_kernel,
            args(asc_out, asc_sf),
            args(pto_out, pto_sf),
        )
        if asc_sf is None:
            compared = [("kv", asc_out, pto_out)]
            traffic = _bytes(data, sf, asc_out, segments)
        else:
            compared = [
                ("kv", asc_out.view(torch.uint8), pto_out.view(torch.uint8)),
                ("kv_sf", asc_sf, pto_sf),
            ]
            traffic = _bytes(data, sf, asc_out, asc_sf, segments)
        return asc_fn, pto_fn, compared, traffic

    exact = layout.quantized_dtype is not None and not layout.do_cast
    return BenchCase(
        "load_kv",
        name,
        prepare,
        exact=exact,
        notes=f"block={block_size} tokens={num_tokens} pattern=aligned layout={layout.note}",
    )


CASES: list[BenchCase] = [
    _norm_case("bf16", has_residual=False, fp8=False),
    _norm_case("bf16_residual", has_residual=True, fp8=False),
    _norm_case("fp8_npc32_packed", has_residual=False, fp8=True),
    _candidate_block_max_case(),
    _candidate_compact_case(),
    _candidate_restore_case(),
    _compress_prefill_case(),
    _compress_decode_case(),
    _confidence_case(),
    _dflash_store_case(),
    _replace_embeds_case(),
    _rgb_case(),
    _downsample_case(False),
    _downsample_case(True),
    _store_case("bf16", BF16_KV_FORMAT, 8192),
    _store_case("v41_fp8", V41_FORMAT, 4096),
    _store_case("v41_fp4", V41_FP4_FORMAT, 4096),
    _load_case("bf16", BF16_KV_FORMAT, 8192),
    _load_case("v41_fp8", V41_FORMAT, 4096),
    _load_case("v41_fp4", V41_FP4_FORMAT, 4096),
    _load_case("mxfp8_index", MXFP8_INDEX_KV_FORMAT, 1024),
]


if __name__ == "__main__":
    print(f"cases={len(CASES)} skipped={SKIPPED}")
    for case in CASES:
        print(f"{case.family}/{case.name} exact={case.exact} {case.notes}")
