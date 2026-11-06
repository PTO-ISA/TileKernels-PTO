"""Production-sized ASC vs PTO launches. Import builds no tensors and prints nothing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BenchCase

import torch

_DEV = "npu"


def _bind(kernel, args):
    def _fn() -> None:
        kernel(*args)

    return _fn


def _finish(asc_k, pto_k, asc_args, pto_args, compared, nbytes):
    asc_k(*asc_args)
    pto_k(*pto_args)
    return _bind(asc_k, asc_args), _bind(pto_k, pto_args), compared, nbytes


def _fork(items: list):
    """One buffer set keeps the originals; the other is a clone of every tensor."""
    asc = []
    pto = []
    for item in items:
        if isinstance(item, torch.Tensor):
            asc.append(item)
            pto.append(item.clone())
        else:
            asc.append(item)
            pto.append(item)
    return asc, pto


def _nbytes(*tensors: torch.Tensor) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def _i32(*shape: int, low: int = 0, high: int = 1000) -> torch.Tensor:
    return torch.randint(low, high, shape, dtype=torch.int32, device=_DEV)


def _remap_half(n_seq: int, n_prev: int) -> torch.Tensor:
    remap = torch.full((n_seq,), -1, dtype=torch.int32, device=_DEV)
    n_old = n_seq // 2
    remap[torch.randperm(n_seq, device=_DEV)[:n_old]] = _i32(n_old, high=n_prev)
    return remap


def _state_views(
    num_layers: int,
    feature_dim: int,
    cache_block: int,
    tensor_block: int,
    align_to: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded layer stride (multiple of align_to) matching the state_cache bench."""

    def _view(block: int, raw: torch.Tensor) -> torch.Tensor:
        pad = raw.shape[1]
        return raw.as_strided((num_layers, block, feature_dim), (pad, feature_dim, 1))

    def _raw(block: int) -> torch.Tensor:
        logical = block * feature_dim
        pad = (logical + align_to - 1) // align_to * align_to
        return torch.randint(0, 256, (num_layers, pad), dtype=torch.uint8, device=_DEV)

    raw_t = _raw(tensor_block)
    raw_c = _raw(cache_block)
    asc_t, asc_c = _view(tensor_block, raw_t), _view(cache_block, raw_c)
    pto_t, pto_c = _view(tensor_block, raw_t.clone()), _view(cache_block, raw_c.clone())
    for view in (asc_t, asc_c, pto_t, pto_c):
        if view.stride(0) % 32 != 0 or view.stride(-1) != 1:
            raise RuntimeError(f"state cache stride {view.stride()} is not 32B-aligned")
    return asc_t, asc_c, pto_t, pto_c


def _yoco_contrib_pad_len(seq_lens: list[int], keep_lens: list[int], cp_size: int) -> int:
    kept = [0] * cp_size
    for sq, keep in zip(seq_lens, keep_lens, strict=True):
        ps = (sq + 2 * cp_size - 1) // (2 * cp_size)
        clip = sq - keep
        for rank in range(cp_size):
            for block in (rank, 2 * cp_size - 1 - rank):
                kept[rank] += max(0, min(ps * (block + 1), sq) - max(ps * block, clip))
    return max(kept)


def prepare_seqlens():
    from deepinfer_kernels.asc.compute_seqlens_k_zos import compute_seqlens_k_zos_tl as asc_factory
    from deepinfer_kernels.pto.compute_seqlens_k_zos import compute_seqlens_k_zos_tl as pto_factory

    torch.manual_seed(1)
    nextn, n_seq = 3, 1024
    n_prev = n_seq + 16
    seqlens_k = _i32(n_seq, low=256, high=356)
    seqlens_q = _i32(n_seq, low=1, high=nextn + 1)
    remap = _remap_half(n_seq, n_prev)
    prev = _i32(n_prev, low=1, high=nextn + 1)
    asc_args, pto_args = _fork([seqlens_k, seqlens_q, remap, prev])
    nbytes = _nbytes(seqlens_k, seqlens_k, seqlens_q, remap, prev)
    return _finish(
        asc_factory(nextn),
        pto_factory(nextn),
        asc_args,
        pto_args,
        [("seqlens_k", asc_args[0], pto_args[0])],
        nbytes,
    )


def prepare_swa():
    from deepinfer_kernels.asc.swa_index import expand_swa_index_tl as asc_factory
    from deepinfer_kernels.pto.swa_index import expand_swa_index_tl as pto_factory

    torch.manual_seed(2)
    num_tokens, width = 8192, 4096
    pivots = _i32(num_tokens, 4, low=-1024, high=100000)
    pivots[:, 1] = _i32(num_tokens, high=width + 1)
    pivots[:, 3] = _i32(num_tokens, high=width + 1)
    pivots[:4] = torch.tensor(
        [[0, 0, 0, 0], [7, width, 99, 0], [0, 0, 99, width], [123, 3, 456, width]],
        dtype=torch.int32,
        device=_DEV,
    )
    out = torch.empty((num_tokens, width), dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([pivots, out])
    nbytes = _nbytes(pivots, out)
    return _finish(
        asc_factory(width),
        pto_factory(width),
        asc_args,
        pto_args,
        [("output", asc_args[1], pto_args[1])],
        nbytes,
    )


def prepare_flash_mla():
    from deepinfer_kernels.asc.flash_mla_utils import convert_kv_index_tl as asc_factory
    from deepinfer_kernels.pto.flash_mla_utils import convert_kv_index_tl as pto_factory

    torch.manual_seed(3)
    b, s, k, n, block_size = 256, 1, 1024, 4096, 16
    # Full context: every lane is valid, and logical indices stay inside block_tables.
    topk = _i32(b, s, k, high=n * block_size)
    context = torch.full((b, s), n * block_size, dtype=torch.int32, device=_DEV)
    tables = _i32(b, n, high=n)
    asc_args, pto_args = _fork([topk, context, tables])
    nbytes = _nbytes(topk, topk, context, tables)
    return _finish(
        asc_factory(s, k, n, block_size),
        pto_factory(s, k, n, block_size),
        asc_args,
        pto_args,
        [("topk_idxs", asc_args[0], pto_args[0])],
        nbytes,
    )


def _prepare_gather(d: int, seed: int):
    from deepinfer_kernels.asc.gather import gather_tl as asc_factory
    from deepinfer_kernels.pto.gather import gather_tl as pto_factory

    torch.manual_seed(seed)
    n = m = 131072
    # Wrapper views the row as int8, so D is the byte width (float32 d * 4).
    x = torch.randn(n, d, dtype=torch.float32, device=_DEV).contiguous()
    index = _i32(m, high=n)
    index[::4] = -1
    index[1::4] = n
    y = torch.empty(m, d, dtype=torch.float32, device=_DEV)
    asc_x, asc_index, asc_y = x, index, y
    pto_x, pto_index, pto_y = x.clone(), index.clone(), y.clone()

    def _args(src, idx, dst):
        src8 = src.view(torch.int8)
        dst8 = dst.view(torch.int8)
        return (src8, idx, dst8), src8.shape[1], src8.stride(0)

    asc_args, asc_d, asc_stride = _args(asc_x, asc_index, asc_y)
    pto_args, pto_d, pto_stride = _args(pto_x, pto_index, pto_y)
    if asc_d != pto_d or asc_stride != pto_stride:
        raise RuntimeError("gather int8 views diverged")
    nbytes = _nbytes(x, y, index)
    return _finish(
        asc_factory(asc_d, torch.int8, torch.int32, asc_stride),
        pto_factory(pto_d, torch.int8, torch.int32, pto_stride),
        asc_args,
        pto_args,
        [("Y", asc_y, pto_y)],
        nbytes,
    )


def prepare_gather_large():
    return _prepare_gather(128, 4)


def prepare_gather_small():
    return _prepare_gather(4, 5)


def prepare_gather_engram():
    from deepinfer_kernels.asc.gather_engram_tokens_zos import gather_engram_tokens_zos_tl as asc_factory
    from deepinfer_kernels.pto.gather_engram_tokens_zos import gather_engram_tokens_zos_tl as pto_factory

    torch.manual_seed(6)
    n_seq, nextn, max_ngram_size, vocab = 1024, 5, 4, 100000
    max_seqlen_q = 1 + nextn
    window = nextn + max_ngram_size - 1
    old_ratio = 0.9
    prev_n = max(1, int(n_seq * old_ratio))
    n_old = int(n_seq * old_ratio)
    seqlens_q = torch.full((n_seq,), max_seqlen_q, dtype=torch.int32, device=_DEV)
    cu = torch.arange(1, n_seq + 1, dtype=torch.int32, device=_DEV) * max_seqlen_q
    total_q = n_seq * max_seqlen_q
    remap = torch.full((n_seq,), -1, dtype=torch.int32, device=_DEV)
    remap[torch.randperm(n_seq, device=_DEV)[:n_old]] = _i32(n_old, high=prev_n)
    win = _i32(n_seq, window, high=vocab)
    fresh = _i32(n_seq, max_seqlen_q, high=vocab)
    sampled = _i32(prev_n, high=vocab)
    predict = _i32(prev_n, nextn, high=vocab)
    accepted = _i32(prev_n, high=nextn + 1)
    lookup = torch.randperm(vocab, dtype=torch.int32, device=_DEV)
    out = torch.empty((total_q, max_ngram_size), dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([win, fresh, sampled, predict, accepted, remap, lookup, seqlens_q, cu, out])
    nbytes = _nbytes(win, fresh, sampled, predict, accepted, remap, lookup, seqlens_q, cu, out)
    return _finish(
        asc_factory(max_seqlen_q, max_ngram_size, 2, -1),
        pto_factory(max_seqlen_q, max_ngram_size, 2, -1),
        asc_args,
        pto_args,
        [("out_ngram_token_ids", asc_args[-1], pto_args[-1])],
        nbytes,
    )


def prepare_gather_ids():
    from deepinfer_kernels.asc.gather_input_ids_zos import gather_input_ids_zos_tl as asc_factory
    from deepinfer_kernels.pto.gather_input_ids_zos import gather_input_ids_zos_tl as pto_factory

    torch.manual_seed(7)
    n_seq, nextn = 1024, 7
    max_seqlen_q = 1 + nextn
    old_ratio = 0.9
    prev_n = max(1, int(n_seq * old_ratio))
    n_old = int(n_seq * old_ratio)
    seqlens_q = torch.full((n_seq,), max_seqlen_q, dtype=torch.int32, device=_DEV)
    cu = torch.arange(1, n_seq + 1, dtype=torch.int32, device=_DEV) * max_seqlen_q
    total_q = n_seq * max_seqlen_q
    fresh = _i32(n_seq, max_seqlen_q, low=1, high=10000)
    sampled = _i32(prev_n, low=1, high=10000)
    predict = _i32(prev_n, nextn, low=1, high=10000)
    remap = torch.full((n_seq,), -1, dtype=torch.int32, device=_DEV)
    remap[torch.randperm(n_seq, device=_DEV)[:n_old]] = _i32(n_old, high=prev_n)
    out = torch.empty((total_q,), dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([fresh, sampled, predict, remap, seqlens_q, cu, out])
    nbytes = _nbytes(fresh, sampled, predict, remap, seqlens_q, cu, out)
    return _finish(
        asc_factory(max_seqlen_q),
        pto_factory(max_seqlen_q),
        asc_args,
        pto_args,
        [("out_input_ids", asc_args[-1], pto_args[-1])],
        nbytes,
    )


def prepare_kv_index_cat():
    from deepinfer_kernels.asc.kv_index_cat import kv_index_cat_tl as asc_factory
    from deepinfer_kernels.pto.kv_index_cat import kv_index_cat_tl as pto_factory

    torch.manual_seed(8)
    total_q, window, topk = 16384, 128, 1024
    total_k_swa, total_k = 666, 23333
    kv_index = _i32(total_q, window, high=10000)
    kv_ca = _i32(total_q, topk, low=-1, high=10000)
    out_shape = window + topk
    out_stride = (out_shape + 3) // 4 * 4

    def _args(index, ca):
        storage = torch.zeros((total_q, out_stride), dtype=torch.int32, device=_DEV)
        out = storage[:, :out_shape]
        return (index, ca, out[:, :window], out[:, window:], total_k_swa, total_k), out

    asc_args, asc_out = _args(kv_index, kv_ca)
    pto_args, pto_out = _args(kv_index.clone(), kv_ca.clone())
    nbytes = _nbytes(kv_index, kv_ca, asc_out)
    return _finish(
        asc_factory("int32", window, topk, out_stride),
        pto_factory("int32", window, topk, out_stride),
        asc_args,
        pto_args,
        [("out", asc_out, pto_out)],
        nbytes,
    )


def _prepare_kv_index(has_block_tables: bool, index_length: int, seed: int):
    from deepinfer_kernels.asc.get_kv_index import load_kv_index_tl as asc_factory
    from deepinfer_kernels.pto.get_kv_index import load_kv_index_tl as pto_factory

    torch.manual_seed(seed)
    n, block_size = 1024, 128
    offset_len = torch.empty((n, 2), dtype=torch.int32, device=_DEV)
    # Aligned offsets keep paged gathers inside a table of n * index_length / block_size.
    offset_len[:, 0] = torch.arange(n, dtype=torch.int32, device=_DEV) * index_length
    offset_len[:, 1] = index_length
    kv_index = torch.empty((n, index_length), dtype=torch.int32, device=_DEV)
    if has_block_tables:
        tables = torch.arange(n * index_length // block_size, dtype=torch.int32, device=_DEV)
    else:
        tables = None
    asc_off, asc_out = offset_len, kv_index
    pto_off, pto_out = offset_len.clone(), kv_index.clone()
    asc_tables = tables
    pto_tables = None if tables is None else tables.clone()
    asc_args = (asc_off, asc_out, asc_tables)
    pto_args = (pto_off, pto_out, pto_tables)
    nbytes = _nbytes(offset_len, kv_index) + (0 if tables is None else _nbytes(tables))
    return _finish(
        asc_factory(index_length, block_size, has_block_tables),
        pto_factory(index_length, block_size, has_block_tables),
        asc_args,
        pto_args,
        [("kv_index", asc_out, pto_out)],
        nbytes,
    )


def prepare_kv_swa():
    return _prepare_kv_index(False, 128, 9)


def prepare_kv_paged():
    return _prepare_kv_index(True, 8192, 10)


def prepare_dflash_input():
    from deepinfer_kernels.asc.get_kv_index import get_fused_dflash_get_input_tl as asc_factory
    from deepinfer_kernels.pto.get_kv_index import get_fused_dflash_get_input_tl as pto_factory

    torch.manual_seed(11)
    real_cnt, dflash_block_size, swa_block_size = 1024, 5, 512
    window_size = swa_block_size
    kv_len = window_size + dflash_block_size
    padded_kv_len = (kv_len + 63) // 64 * 64
    seq_lens = torch.randint(1, 64, (real_cnt,), dtype=torch.int32)
    ftp = torch.zeros(real_cnt, dtype=torch.int32)
    ftp[1:] = torch.cumsum(seq_lens[:-1], dim=0, dtype=torch.int32)
    total_q = int(ftp[-1].item() + seq_lens[-1].item())
    ftp = ftp.to(_DEV)
    seq_lens = seq_lens.to(_DEV)
    positions = _i32(total_q, high=100000)
    positions[0] = 0
    slot_mapping = _i32(total_q, high=100000)
    last_valid = _i32(real_cnt, high=6)
    last_valid[0] = 0
    newx = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=_DEV)
    pos_out = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=_DEV)
    ids = torch.empty((real_cnt, dflash_block_size), dtype=torch.int32, device=_DEV)
    vtm = torch.empty((real_cnt, dflash_block_size), dtype=torch.bool, device=_DEV)
    kvi = torch.empty((real_cnt * dflash_block_size, padded_kv_len), dtype=torch.int32, device=_DEV)
    # has_gate_force_random is False, so the wrapper passes None for that output.
    base = [slot_mapping, positions, ftp, last_valid, newx, pos_out, ids, vtm, None, kvi, kv_len, window_size]
    asc_args, pto_args = _fork(base)
    nbytes = _nbytes(slot_mapping, positions, ftp, last_valid, newx, pos_out, ids, vtm, kvi)
    del seq_lens
    return _finish(
        asc_factory(swa_block_size, padded_kv_len, True, False),
        pto_factory(swa_block_size, padded_kv_len, True, False),
        asc_args,
        pto_args,
        [
            ("newx_slot_mapping", asc_args[4], pto_args[4]),
            ("positions", asc_args[5], pto_args[5]),
            ("input_ids", asc_args[6], pto_args[6]),
            ("valid_token_mask", asc_args[7], pto_args[7]),
            ("kv_index", asc_args[9], pto_args[9]),
        ],
        nbytes,
    )


def prepare_schedule():
    from deepinfer_kernels.asc.schedule_mtp_varlen import compute_seqlens_q_by_confidence_rates_tl as asc_factory
    from deepinfer_kernels.pto.schedule_mtp_varlen import compute_seqlens_q_by_confidence_rates_tl as pto_factory

    torch.manual_seed(12)
    num_seqs, nextn = 1024, 7
    total_q = int((1 + nextn) * num_seqs * 0.95)
    n_prev = num_seqs + 16
    remap = _remap_half(num_seqs, n_prev)
    prev = torch.rand(n_prev, nextn, dtype=torch.float32, device=_DEV) * 0.8 + 0.1
    cum = torch.empty((num_seqs, nextn), dtype=torch.float32, device=_DEV)
    seqlens_q = torch.empty((num_seqs,), dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([remap, prev, cum, seqlens_q, total_q])
    nbytes = _nbytes(remap, prev, cum, seqlens_q)
    return _finish(
        asc_factory(nextn=nextn),
        pto_factory(nextn=nextn),
        asc_args,
        pto_args,
        [("cum_rates", asc_args[2], pto_args[2]), ("seqlens_q", asc_args[3], pto_args[3])],
        nbytes,
    )


def _prepare_mtp(vocab: int, has_index: bool, has_confidence: bool, seed: int):
    from deepinfer_kernels.asc.store_mtp_info import store_mtp_info_tl as asc_factory
    from deepinfer_kernels.pto.store_mtp_info import store_mtp_info_tl as pto_factory

    torch.manual_seed(seed)
    req_cnt, nextn = 128, 4
    # Slots 1..req_cnt are unique; every 8th request is slot 0 and must be skipped.
    write_slots = torch.arange(1, req_cnt + 1, dtype=torch.int32, device=_DEV)
    write_slots[::8] = 0
    token_ids = _i32(req_cnt, nextn, high=100000)
    logprobs = torch.randn((req_cnt, nextn, vocab), dtype=torch.float32, device=_DEV)
    confidence = torch.rand((req_cnt, nextn), dtype=torch.float32, device=_DEV) if has_confidence else None
    index = _i32(req_cnt, nextn, vocab, high=129280) if has_index else None
    pool_rows = (req_cnt + 1) * nextn
    token_pool = _i32(pool_rows, high=100000)
    logprobs_pool = torch.randn((pool_rows, vocab), dtype=torch.float32, device=_DEV)
    index_pool = _i32(pool_rows, vocab, high=129280) if has_index else None
    confidence_pool = torch.rand((pool_rows,), dtype=torch.float32, device=_DEV) if has_confidence else None
    args = [write_slots, token_ids, logprobs, index, confidence, token_pool, logprobs_pool, index_pool, confidence_pool]
    asc_args, pto_args = _fork(args)
    compared = [
        ("token_ids_pool", asc_args[5], pto_args[5]),
        ("logprobs_pool", asc_args[6], pto_args[6]),
    ]
    if has_index:
        compared.append(("logprobs_index_pool", asc_args[7], pto_args[7]))
    if has_confidence:
        compared.append(("confidence_pool", asc_args[8], pto_args[8]))
    read = [t for t in args if isinstance(t, torch.Tensor)]
    nbytes = _nbytes(*read)
    return _finish(
        asc_factory(nextn, vocab, has_index, has_confidence),
        pto_factory(nextn, vocab, has_index, has_confidence),
        asc_args,
        pto_args,
        compared,
        nbytes,
    )


def prepare_mtp_no_index():
    return _prepare_mtp(129280, False, True, 13)


def prepare_mtp_index():
    return _prepare_mtp(1024, True, True, 14)


def prepare_dflash_copy():
    from deepinfer_kernels.asc.dflash_copy import copy_dflash_decode_input_tl as asc_factory
    from deepinfer_kernels.pto.dflash_copy import copy_dflash_decode_input_tl as pto_factory

    torch.manual_seed(15)
    src_n, dst_n = 257, 512

    def _rows(n: int, offset: int) -> list[torch.Tensor]:
        base = torch.arange(n, dtype=torch.int32, device=_DEV) + offset
        return [
            base,
            base + 1000,
            base + 2000,
            base + 3000,
            (base + 3000).to(torch.float32) + 0.25,
            (base + 3000).to(torch.float32) + 0.75,
            base + 3000,
            (base + 3000).to(torch.int64).to(torch.uint64),
            base + 3000,
        ]

    src = _rows(src_n, 10)
    padding = _rows(dst_n, 100)
    dst = _rows(dst_n, 1000)
    labels = (
        "first_token_positions",
        "last_valid_offset",
        "golden_tokens",
        "sample_positions",
        "temperatures",
        "top_ps",
        "top_ks",
        "seq_seeds",
        "mtp_info_write_slots",
    )
    asc_args, pto_args = _fork(src + padding + dst)
    n = len(labels)
    compared = [(labels[i], asc_args[2 * n + i], pto_args[2 * n + i]) for i in range(n)]
    nbytes = _nbytes(*src, *padding, *dst, *dst)
    return _finish(
        asc_factory(True),
        pto_factory(True),
        asc_args,
        pto_args,
        compared,
        nbytes,
    )


def _prepare_state(cache_to_tensor: bool, seed: int):
    if cache_to_tensor:
        from deepinfer_kernels.asc.state_cache import select_state_slice_tl as asc_factory
        from deepinfer_kernels.pto.state_cache import select_state_slice_tl as pto_factory
    else:
        from deepinfer_kernels.asc.state_cache import load_state_to_cache_tl as asc_factory
        from deepinfer_kernels.pto.state_cache import load_state_to_cache_tl as pto_factory

    torch.manual_seed(seed)
    feature_dim = 448 + 64 * 2 + 8
    cache_block, tensor_block, align_to = 130 + 128, 130, 32
    num_layers = 8
    last_hit = 500
    asc_t, asc_c, pto_t, pto_c = _state_views(num_layers, feature_dim, cache_block, tensor_block, align_to)
    asc_args = (asc_t, asc_c, last_hit)
    pto_args = (pto_t, pto_c, last_hit)
    moved = num_layers * tensor_block * feature_dim
    nbytes = moved * 2
    if cache_to_tensor:
        compared = [("state_tensor", asc_t, pto_t)]
    else:
        compared = [("state_cache", asc_c, pto_c)]
    return _finish(
        asc_factory(cache_block, tensor_block, num_layers, feature_dim),
        pto_factory(cache_block, tensor_block, num_layers, feature_dim),
        asc_args,
        pto_args,
        compared,
        nbytes,
    )


def prepare_select_state():
    return _prepare_state(True, 16)


def prepare_load_state():
    return _prepare_state(False, 17)


def prepare_bitmask():
    from deepinfer_kernels.asc.structure_output import apply_token_bitmask_by_step_inplace_tl as asc_factory
    from deepinfer_kernels.pto.structure_output import apply_token_bitmask_by_step_inplace_tl as pto_factory

    torch.manual_seed(18)
    num_requests, vocab, step = 512, 129280, 0
    # One packed row per request keeps the logits tensor under the 512MB cap.
    # vocab 129280 is the benchmark vocab and a multiple of 64.
    logits = torch.randn((num_requests, vocab), dtype=torch.float32, device=_DEV)
    bitmask_size = vocab // 32
    bitmask = torch.randint(-(2**31), 2**31 - 1, (num_requests, bitmask_size), dtype=torch.int32, device=_DEV)
    indices = torch.arange(num_requests, dtype=torch.int32, device=_DEV)
    ftp = torch.arange(num_requests, dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([logits, bitmask, indices, ftp, step])
    nbytes = _nbytes(logits, logits, bitmask, indices, ftp)
    return _finish(
        asc_factory(vocab, True),
        pto_factory(vocab, True),
        asc_args,
        pto_args,
        [("logits", asc_args[0], pto_args[0])],
        nbytes,
    )


def prepare_sum_probs():
    from deepinfer_kernels.asc.sum_smaller_probs import sum_smaller_probs_tl as asc_factory
    from deepinfer_kernels.pto.sum_smaller_probs import sum_smaller_probs_tl as pto_factory

    torch.manual_seed(19)
    batch, vocab = 512, 129280
    probs = torch.rand((batch, vocab), dtype=torch.float32, device=_DEV)
    logprobs = (probs / probs.sum(dim=-1, keepdim=True)).log().contiguous()
    sample_results = _i32(batch, high=vocab)
    output = torch.empty((batch,), dtype=torch.float32, device=_DEV)
    asc_args, pto_args = _fork([logprobs, sample_results, output])
    nbytes = _nbytes(logprobs, sample_results, output)
    return _finish(
        asc_factory(vocab),
        pto_factory(vocab),
        asc_args,
        pto_args,
        [("output", asc_args[2], pto_args[2])],
        nbytes,
    )


def prepare_cp_mirror():
    from deepinfer_kernels.asc.prepare_cp_mirror import prepare_cp_mirror_tl as asc_factory
    from deepinfer_kernels.pto.prepare_cp_mirror import prepare_cp_mirror_tl as pto_factory

    torch.manual_seed(20)
    # Largest prepare bench: batch 4096, ratios (1, 2), cp_size 8, last rank.
    batch, cp_size, cp_rank = 4096, 8, 7
    ratios = (1, 2)
    generator = torch.Generator(device="cpu").manual_seed(20)
    sq = torch.randint(1, 129, (batch,), dtype=torch.int32, generator=generator)
    context = torch.randint(0, 257, (batch,), dtype=torch.int32, generator=generator)
    cu_q = torch.zeros(batch + 1, dtype=torch.int32)
    cu_k = torch.zeros(batch + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(sq, dim=0, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(sq + context, dim=0, dtype=torch.int32)
    total_q = int(cu_q[-1].item())
    output_len = int((2 * ((sq + 2 * cp_size - 1) // (2 * cp_size))).sum().item())
    cu_q = cu_q.to(_DEV)
    cu_k = cu_k.to(_DEV)
    num_rows = len(ratios) + 1
    cu_lens = torch.empty((num_rows, batch + 1), dtype=torch.int32, device=_DEV)
    reverse = torch.empty((total_q,), dtype=torch.int32, device=_DEV)
    mirror = torch.empty((output_len,), dtype=torch.int32, device=_DEV)
    ks = torch.empty((len(ratios), output_len), dtype=torch.int32, device=_DEV)
    ke = torch.empty((len(ratios), output_len), dtype=torch.int32, device=_DEV)
    base = [cu_q, cu_k, cu_lens, reverse, mirror, ks, ke, cp_rank]
    asc_args, pto_args = _fork(base)
    nbytes = _nbytes(cu_q, cu_k, cu_lens, reverse, mirror, ks, ke)
    return _finish(
        asc_factory(cp_size=cp_size, compress_ratios=ratios),
        pto_factory(cp_size=cp_size, compress_ratios=ratios),
        asc_args,
        pto_args,
        [
            ("cu_lens", asc_args[2], pto_args[2]),
            ("mirror_cp_reverse_index", asc_args[3], pto_args[3]),
            ("mirror_cp_index", asc_args[4], pto_args[4]),
            ("ks", asc_args[5], pto_args[5]),
            ("ke", asc_args[6], pto_args[6]),
        ],
        nbytes,
    )


def prepare_yoco():
    from deepinfer_kernels.asc.prepare_cp_mirror import yoco_cp_boundary_fused_tl as asc_factory
    from deepinfer_kernels.pto.prepare_cp_mirror import yoco_cp_boundary_fused_tl as pto_factory

    torch.manual_seed(21)
    # Bench shape (256, 500) from _CP_MIRROR_BENCH_SHAPES. cp_size 4, keep = seqlen.
    n_seq, seqlen, cp_size, cp_rank = 256, 500, 4, 0
    keep = min(seqlen, 4096)
    seq_lens = [seqlen] * n_seq
    keep_lens = [keep] * n_seq
    cu_q = (torch.arange(n_seq + 1, dtype=torch.int32) * seqlen).to(_DEV)
    cu_c = (torch.arange(n_seq + 1, dtype=torch.int32) * keep).to(_DEV)
    total_q_c = n_seq * keep
    contrib_pad = _yoco_contrib_pad_len(seq_lens, keep_lens, cp_size)
    cu_lens = torch.empty((cp_size + 1, n_seq + 1), dtype=torch.int32, device=_DEV)
    contrib = torch.empty((contrib_pad,), dtype=torch.int32, device=_DEV)
    reverse = torch.empty((total_q_c,), dtype=torch.int32, device=_DEV)
    base = [cu_q, cu_c, cu_lens, contrib, reverse, cp_rank]
    asc_args, pto_args = _fork(base)
    nbytes = _nbytes(cu_q, cu_c, cu_lens, contrib, reverse)
    return _finish(
        asc_factory(cp_size=cp_size),
        pto_factory(cp_size=cp_size),
        asc_args,
        pto_args,
        [
            ("cu_lens", asc_args[2], pto_args[2]),
            ("yoco_contrib_index", asc_args[3], pto_args[3]),
            ("yoco_gathered_reverse_index", asc_args[4], pto_args[4]),
        ],
        nbytes,
    )


def prepare_block_tables():
    from deepinfer_kernels.asc.prepare_v4_decode_metadata import copy_block_tables_tl as asc_factory
    from deepinfer_kernels.pto.prepare_v4_decode_metadata import copy_block_tables_tl as pto_factory

    torch.manual_seed(22)
    n_seq, max_seqlen_q, max_num_blocks = 256, 8, 4096
    num_blocks = max_num_blocks
    seqlens_q = torch.full((n_seq,), max_seqlen_q, dtype=torch.int32, device=_DEV)
    cu = torch.arange(1, n_seq + 1, dtype=torch.int32, device=_DEV) * max_seqlen_q
    total_q = n_seq * max_seqlen_q
    offsets = torch.arange(n_seq + 1, dtype=torch.int32, device=_DEV) * num_blocks
    tables_in = _i32(n_seq * num_blocks, high=1000)
    tables_out = torch.zeros((total_q, max_num_blocks), dtype=torch.int32, device=_DEV)
    asc_args, pto_args = _fork([seqlens_q, cu, tables_in, offsets, tables_out])
    nbytes = _nbytes(seqlens_q, cu, tables_in, offsets, tables_out)
    return _finish(
        asc_factory(max_seqlen_q, max_num_blocks),
        pto_factory(max_seqlen_q, max_num_blocks),
        asc_args,
        pto_args,
        [("block_tables_out", asc_args[4], pto_args[4])],
        nbytes,
    )


def prepare_v4_meta():
    from deepinfer_kernels.asc.prepare_v4_decode_metadata import prepare_v4_decode_metadata_tl as asc_factory
    from deepinfer_kernels.pto.prepare_v4_decode_metadata import prepare_v4_decode_metadata_tl as pto_factory

    torch.manual_seed(23)
    # Largest bench n_seq. Ratios match the bench: ca=sparse=(1, 2), dense empty.
    n_seq, max_seqlen_q = 1024, 8
    max_num_blocks, window_size, swa_block_size, block_size = 4096, 4096, 8192, 128
    ca_ratios, dense_ratios, sparse_ratios = (1, 2), (), (1, 2)
    force_load_balance = False
    seqlens_q = torch.full((n_seq,), max_seqlen_q, dtype=torch.int32, device=_DEV)
    seqlens_k = seqlens_q + 256
    cu = torch.arange(1, n_seq + 1, dtype=torch.int32, device=_DEV) * max_seqlen_q
    total_q = n_seq * max_seqlen_q
    min_ratio = min(ca_ratios)
    token_per_block = block_size // min_ratio
    sk = int(max_seqlen_q + 256)
    num_blocks = max(1, (sk // min_ratio + token_per_block - 1) // token_per_block)
    offsets = torch.arange(n_seq + 1, dtype=torch.int32, device=_DEV) * num_blocks
    tables_in = _i32(n_seq * num_blocks, high=1000)
    state_in = _i32(n_seq, high=100)
    placeholder = torch.zeros((n_seq,), dtype=torch.bool, device=_DEV)
    positions = torch.zeros((total_q,), dtype=torch.int32, device=_DEV)
    valid = torch.zeros((total_q,), dtype=torch.bool, device=_DEV)
    gate = torch.zeros((total_q,), dtype=torch.bool, device=_DEV)
    fix = torch.zeros((total_q,), dtype=torch.bool, device=_DEV)
    context = torch.zeros((total_q,), dtype=torch.int32, device=_DEV)
    klen = torch.zeros((len(sparse_ratios), total_q), dtype=torch.int32, device=_DEV)
    kv_ca = torch.zeros((max(len(dense_ratios), 1), total_q, 2), dtype=torch.int32, device=_DEV)
    kv_swa = torch.zeros((total_q, 2), dtype=torch.int32, device=_DEV)
    slot_swa = torch.zeros((total_q,), dtype=torch.int32, device=_DEV)
    query = torch.zeros((total_q,), dtype=torch.int32, device=_DEV)
    state_out = torch.zeros((total_q,), dtype=torch.int32, device=_DEV)
    pos_c = torch.zeros((len(ca_ratios), total_q), dtype=torch.int32, device=_DEV)
    slot_ca = torch.zeros((len(ca_ratios), total_q), dtype=torch.int32, device=_DEV)
    base = [
        seqlens_q,
        seqlens_k,
        cu,
        tables_in,
        offsets,
        state_in,
        placeholder,
        positions,
        valid,
        gate,
        fix,
        context,
        klen,
        kv_ca,
        kv_swa,
        slot_swa,
        query,
        state_out,
        pos_c,
        slot_ca,
    ]
    asc_args, pto_args = _fork(base)
    labels = (
        "positions",
        "valid_token_mask",
        "gate_force_random",
        "fix_routing_mask",
        "context_lens",
        "klen",
        "kv_ca_index_offset_len",
        "kv_swa_index_offset_len",
        "slot_mapping_store_swa",
        "query_indices",
        "state_block_idx_out",
        "positions_c",
        "slot_mapping_store_ca",
    )
    compared = [(labels[i], asc_args[7 + i], pto_args[7 + i]) for i in range(len(labels))]
    nbytes = _nbytes(*base)
    return _finish(
        asc_factory(
            max_seqlen_q,
            max_num_blocks,
            window_size,
            swa_block_size,
            block_size,
            force_load_balance,
            ca_ratios,
            dense_ratios,
            sparse_ratios,
        ),
        pto_factory(
            max_seqlen_q,
            max_num_blocks,
            window_size,
            swa_block_size,
            block_size,
            force_load_balance,
            ca_ratios,
            dense_ratios,
            sparse_ratios,
        ),
        asc_args,
        pto_args,
        compared,
        nbytes,
    )


CASES: list[BenchCase] = [
    BenchCase("seqlens", "n1024_nextn3", prepare_seqlens, notes="half the rows remapped; in-place seqlens_k"),
    BenchCase("swa", "n8192_w4096", prepare_swa, notes="lens clipped to width"),
    BenchCase("flash_mla", "b256_k1024_full", prepare_flash_mla, notes="in-place topk; full context; block_size 16"),
    BenchCase("gather", "large_d128", prepare_gather_large, notes="float32 viewed as int8, D=512; some OOB indexes"),
    BenchCase("gather", "small_d4", prepare_gather_small, notes="float32 viewed as int8, D=16; some OOB indexes"),
    BenchCase("gather_engram", "n1024_nextn5", prepare_gather_engram, notes="max_ngram_size 4, old_ratio 0.9, vocab 100000"),
    BenchCase("gather_ids", "n1024_nextn7", prepare_gather_ids, notes="old_ratio 0.9"),
    BenchCase("kv_index_cat", "q16384_w128_k1024", prepare_kv_index_cat, notes="largest _KV_INDEX_CAT_CASES row"),
    BenchCase("kv_index", "swa_n1024", prepare_kv_swa, notes="index_length 128, block_size 128, no block tables"),
    BenchCase("kv_index", "paged_n1024_b128", prepare_kv_paged, notes="index_length 8192, block_size 128, in-bounds tables"),
    BenchCase(
        "dflash_input",
        "n1024_blk5_swa512",
        prepare_dflash_input,
        notes="valid_token_mask on, gate_force_random off (None)",
    ),
    BenchCase(
        "schedule",
        "n1024_nextn7",
        prepare_schedule,
        exact=False,
        notes="bench nextn=7, total_q=(1+nextn)*num_seqs*0.95; total_q is a scalar input",
    ),
    BenchCase("mtp", "v129280_no_index", prepare_mtp_no_index, notes="req 128, nextn 4, confidence on, unique slots, slot 0 skipped"),
    BenchCase("mtp", "v1024_index_conf", prepare_mtp_index, notes="req 128, nextn 4, index and confidence, slot 0 skipped"),
    BenchCase("dflash_copy", "src257_dst512", prepare_dflash_copy, notes="has_driver_inputs"),
    BenchCase("state_cache", "select_slice", prepare_select_state, notes="feature 584, cache 258, tensor 130, layers 8"),
    BenchCase("state_cache", "load_to_cache", prepare_load_state, notes="same shape as select_slice, opposite direction"),
    BenchCase("bitmask", "b512_v129280_ftp", prepare_bitmask, notes="has_ftp, step 0, one row per request"),
    BenchCase("sum_probs", "b512_v129280", prepare_sum_probs, exact=False, notes="normalized logprobs, vocab 129280"),
    BenchCase("cp_mirror", "prepare_b4096_c1_2", prepare_cp_mirror, notes="cp_size 8, cp_rank 7, ratios (1, 2)"),
    BenchCase("yoco", "n256_len500", prepare_yoco, notes="cp_size 4, cp_rank 0, keep=seqlen"),
    BenchCase("block_tables", "n256_q8_b4096", prepare_block_tables, notes="full max_num_blocks rows"),
    BenchCase("v4_meta", "n1024_q8", prepare_v4_meta, notes="ca=sparse (1, 2), dense empty, block 128"),
]
