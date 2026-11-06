"""ASC vs PTO launches for sampler, random sampling, fused sample, and fused transform.

prepare() puts contiguous tensors on the BENCH_DEVICES npu, compiles the ASC and
PTO factories, clones every buffer, runs each kernel once, and returns relaunch
closures. Import builds no tensors.

No public kernel is omitted. The fused-transform KV-cache store arm is not
launched: store_cache requires num_heads == 1, and these rows use num_heads=8.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BenchCase

import torch

# remap_draft_topk_tl scatters token ids into a 43136-wide pool, one range at a
# time, and unrolls the ranges when vocab_size <= 4 * pool.
_REMAP_POOL = 43136
_VOCAB = 129280
_TOPK = 1024


def _dev() -> torch.device:
    import torch_npu  # noqa: F401

    index = int(os.environ.get("BENCH_DEVICES", "1"))
    torch.npu.set_device(index)
    return torch.device("npu", index)


def _bind(kernel, args):
    def fn() -> None:
        kernel(*args)

    return fn


def _nbytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.nbytes for tensor in tensors)


def _place(dev: torch.device, args: list):
    placed = []
    for item in args:
        if isinstance(item, torch.Tensor):
            placed.append(item.to(dev).contiguous())
        else:
            placed.append(item)
    return placed


def _split(args: list):
    """ASC keeps the placed buffers. PTO gets a clone of every tensor."""
    asc, pto = [], []
    for item in args:
        if isinstance(item, torch.Tensor):
            asc.append(item)
            pto.append(item.clone())
        else:
            asc.append(item)
            pto.append(item)
    return asc, pto


def _launch(asc_kernel, pto_kernel, host_args: list, outputs: list[tuple[str, int]], traffic: list[torch.Tensor]):
    args = _place(_dev(), host_args)
    asc_args, pto_args = _split(args)
    asc_fn, pto_fn = _bind(asc_kernel, asc_args), _bind(pto_kernel, pto_args)
    asc_fn()
    pto_fn()
    torch.npu.synchronize()
    compared = [(name, asc_args[index], pto_args[index]) for name, index in outputs]
    return asc_fn, pto_fn, compared, _nbytes(*traffic)


def _requests(req_cnt: int, max_steps: int, seed: int):
    """Packed rows with seqlen in [1, max_steps + 1], matching the benchmarks."""
    gen = torch.Generator().manual_seed(seed)
    seqlen_q = torch.randint(1, max_steps + 2, (req_cnt,), generator=gen, dtype=torch.int32)
    assert int(seqlen_q.min()) >= 1 and int(seqlen_q.max()) <= max_steps + 1
    ftp = torch.zeros(req_cnt, dtype=torch.int32)
    ftp[1:] = torch.cumsum(seqlen_q.to(torch.int64), 0)[:-1].to(torch.int32)
    total_q = int(seqlen_q.sum())
    assert int(ftp[-1] + seqlen_q[-1]) == total_q
    row_to_req = torch.repeat_interleave(torch.arange(req_cnt, dtype=torch.int32), seqlen_q)
    rows = torch.arange(total_q, dtype=torch.int32)
    local_step = rows - ftp[row_to_req.long()]
    assert int(local_step.min()) >= 0
    return gen, ftp, seqlen_q, row_to_req, local_step


def _subset(rows: int, cols: int, vocab: int, gen: torch.Generator) -> torch.Tensor:
    """Unique token ids per row, each in [0, vocab)."""
    perm = torch.randperm(vocab, generator=gen)
    start = torch.randint(0, vocab, (rows, 1), generator=gen)
    return perm[(start + torch.arange(cols)) % vocab].to(torch.int32).contiguous()


def _sampler():
    _dev()
    from deepinfer_kernels.asc import sampler as asc
    from deepinfer_kernels.pto import sampler as pto

    return asc, pto


def prepare_verification():
    asc, pto = _sampler()
    req_cnt, max_steps = 256, 7
    _gen, ftp, seqlen_q, _row_to_req, _local_step = _requests(req_cnt, max_steps, 7)
    total_q = int(seqlen_q.sum())
    # UB holds BR=8 requests by steps_aligned=((7+8)//8)*8 == 16. seqlen <= 8 fits.
    row_to_req = torch.empty(total_q, dtype=torch.int32)
    local_step = torch.empty(total_q, dtype=torch.int32)
    is_verify = torch.empty(total_q, dtype=torch.bool)
    kernel_asc = asc.prepare_verification_metadata_tl(max_steps)
    kernel_pto = pto.prepare_verification_metadata_tl(max_steps)
    args = [ftp, seqlen_q, row_to_req, local_step, is_verify]
    return _launch(
        kernel_asc,
        kernel_pto,
        args,
        [("row_to_req", 2), ("local_step", 3), ("is_verify", 4)],
        args,
    )


def prepare_expand():
    asc, pto = _sampler()
    req_cnt, max_steps = 256, 7
    gen, _ftp, seqlen_q, row_to_req, local_step = _requests(req_cnt, max_steps, 11)
    total_q = row_to_req.shape[0]
    temperatures = torch.rand(req_cnt, generator=gen)
    top_ps = torch.rand(req_cnt, generator=gen)
    top_ks = torch.randint(-1, 1024, (req_cnt,), generator=gen, dtype=torch.int32)
    sample_positions = torch.randint(0, 10**8, (req_cnt,), generator=gen, dtype=torch.int32)
    seq_seeds = torch.randint(0, 10**18, (req_cnt,), generator=gen, dtype=torch.int64).to(torch.uint64)
    temperatures_q = torch.empty(total_q)
    top_ps_q = torch.empty(total_q)
    top_ks_q = torch.empty(total_q, dtype=torch.int32)
    sample_positions_q = torch.empty(total_q, dtype=torch.int32)
    seq_seeds_q = torch.empty(total_q, dtype=torch.uint64)
    kernel_asc = asc.expand_sampling_params_tl()
    kernel_pto = pto.expand_sampling_params_tl()
    args = [
        temperatures,
        top_ps,
        top_ks,
        sample_positions,
        seq_seeds,
        row_to_req,
        local_step,
        temperatures_q,
        top_ps_q,
        top_ks_q,
        sample_positions_q,
        seq_seeds_q,
    ]
    return _launch(
        kernel_asc,
        kernel_pto,
        args,
        [
            ("temperatures_q", 7),
            ("top_ps_q", 8),
            ("top_ks_q", 9),
            ("sample_positions_q", 10),
            ("seq_seeds_q", 11),
        ],
        args,
    )


def prepare_map_lookup():
    asc, pto = _sampler()
    rows, force_topk, vocab = 256, _TOPK, _VOCAB
    assert force_topk % 64 == 0
    gen = torch.Generator().manual_seed(13)
    # Correctness test's by_key float32 path, at the benchmark width 1024.
    key = _subset(rows, force_topk, vocab, gen)
    value = torch.randn(rows, force_topk, generator=gen)
    positions = torch.randint(-force_topk, 2 * force_topk, (rows, 1), generator=gen, dtype=torch.int32)
    random_keys = torch.randint(-vocab, 2 * vocab, (rows, 1), generator=gen, dtype=torch.int32)
    in_range = (positions >= 0) & (positions < force_topk)
    query = torch.where(in_range, key.gather(1, positions.clamp(0, force_topk - 1)), random_keys).to(torch.int32)
    out = torch.empty(rows, 1)
    kernel_asc = asc.map_lookup_tl(force_topk, mode="by_key", value_dtype=torch.float32)
    kernel_pto = pto.map_lookup_tl(force_topk, mode="by_key", value_dtype=torch.float32)
    args = [key, value, query, out]
    return _launch(kernel_asc, kernel_pto, args, [("out", 3)], args)


def prepare_remap():
    asc, pto = _sampler()
    total_q, force_topk, vocab = 256, _TOPK, _VOCAB
    assert force_topk % 64 == 0 and force_topk <= 4096
    assert vocab <= 4 * _REMAP_POOL
    gen = torch.Generator().manual_seed(17)
    pool_rows = total_q
    draft_index = _subset(pool_rows, force_topk, vocab, gen)
    draft_logprobs = torch.randn(pool_rows, force_topk, generator=gen)
    picked = torch.randint(0, force_topk, (pool_rows, 1), generator=gen)
    draft_ids = draft_index.gather(1, picked).squeeze(1).contiguous()
    draft_ids.masked_fill_(torch.rand(pool_rows, generator=gen) < 0.1, -1)
    read_rows = torch.randint(0, pool_rows, (total_q,), generator=gen, dtype=torch.int32)
    target_index = _subset(total_q, force_topk, vocab, gen)
    aligned = torch.empty(total_q, force_topk)
    local_ids = torch.empty(total_q, dtype=torch.int32)
    identity = torch.empty(total_q, dtype=torch.int32)
    kernel_asc = asc.remap_draft_topk_tl(force_topk, vocab)
    kernel_pto = pto.remap_draft_topk_tl(force_topk, vocab)
    args = [draft_index, draft_logprobs, draft_ids, target_index, read_rows, aligned, local_ids, identity]
    return _launch(
        kernel_asc,
        kernel_pto,
        args,
        [("aligned_logprobs", 5), ("draft_local_ids", 6), ("identity_rows", 7)],
        args,
    )


def prepare_accept():
    asc, pto = _sampler()
    # Benchmark packing at total_q=256. The test covers has_golden True and False;
    # the benchmark row passes golden.
    total_q, request_length, vocab = 256, 16, _VOCAB
    req_cnt = total_q // request_length
    gen = torch.Generator().manual_seed(19)
    raw_accept = torch.rand(total_q, generator=gen) < 0.8
    golden = torch.randint(-1, vocab, (total_q,), generator=gen, dtype=torch.int32)
    draft = torch.randint(0, vocab, (total_q,), generator=gen, dtype=torch.int32)
    read_rows = torch.arange(total_q - 1, -1, -1, dtype=torch.int32)
    ftp = torch.arange(0, total_q, request_length, dtype=torch.int32)
    row_to_req = torch.arange(req_cnt, dtype=torch.int32).repeat_interleave(request_length)
    is_verify = torch.arange(total_q) % request_length != request_length - 1
    reject = torch.randint(0, vocab, (total_q,), generator=gen, dtype=torch.int32)
    fresh = torch.randint(0, vocab, (total_q,), generator=gen, dtype=torch.int32)
    candidate = torch.where(is_verify, torch.where(raw_accept, draft[read_rows.long()], reject), fresh).to(torch.int32)
    token_ids = torch.empty(total_q, dtype=torch.int32)
    accept_out = torch.empty(total_q, dtype=torch.bool)
    is_final = torch.empty(total_q, dtype=torch.bool)
    kernel_asc = asc.get_accept_result_tl(True)
    kernel_pto = pto.get_accept_result_tl(True)
    args = [
        raw_accept,
        golden,
        draft,
        read_rows,
        candidate,
        is_verify,
        ftp,
        row_to_req,
        token_ids,
        accept_out,
        is_final,
    ]
    return _launch(
        kernel_asc,
        kernel_pto,
        args,
        [("token_ids", 8), ("effective_accept", 9), ("is_final", 10)],
        args,
    )


def prepare_finalize():
    asc, pto = _sampler()
    req_cnt, max_steps, topk_n, num_stop = 256, 3, 64, 2
    # finalize_verification_tl assumes each block of BR=4 requests has
    # 1..BR*(num_steps+1) rows and each request has at most num_steps+1 rows.
    gen, ftp, seqlen_q, row_to_req, _local_step = _requests(req_cnt, max_steps, 23)
    total_q = int(seqlen_q.sum())
    block = 4 * (max_steps + 1)
    for base in range(0, req_cnt, 4):
        count = int(seqlen_q[base : base + 4].sum())
        assert 0 < count <= block
    rows = torch.arange(total_q)
    final_rows = ftp + seqlen_q - 1
    is_verify = rows != final_rows[row_to_req.long()]
    accept = torch.rand(total_q, generator=gen) < 0.7
    reject_at = torch.where(is_verify & ~accept, rows, total_q).to(torch.int32)
    final_rows = final_rows.scatter_reduce(0, row_to_req.long(), reject_at, reduce="amin", include_self=True)
    effective_accept = is_verify & (rows < final_rows[row_to_req.long()])
    is_final = rows == final_rows[row_to_req.long()]
    assert int(is_final.sum()) == req_cnt
    token_ids = torch.randint(0, _VOCAB, (total_q,), generator=gen, dtype=torch.int32)
    token_logprobs = torch.randn(total_q, generator=gen)
    sum_smaller = torch.rand(total_q, generator=gen)
    top_logprobs = torch.randn(total_q, topk_n, generator=gen)
    top_ids = torch.randint(0, _VOCAB, (total_q, topk_n), generator=gen, dtype=torch.int32)
    accept_probs = torch.rand(total_q, generator=gen)
    confidence = torch.rand(total_q, generator=gen)
    confidence_rows = torch.arange(total_q, dtype=torch.int32)
    stop_ids = torch.randint(0, _VOCAB, (req_cnt, num_stop), generator=gen, dtype=torch.int32)
    last_valid = torch.empty(req_cnt, dtype=torch.int32)
    num_generated = torch.empty(max_steps, dtype=torch.int32)
    num_verified = torch.empty(max_steps, dtype=torch.int32)
    confidence_error = torch.empty(1)
    emitted = torch.empty(req_cnt, dtype=torch.bool)
    sampled_ids = torch.empty(req_cnt, dtype=torch.int32)
    sampled_lp = torch.empty(req_cnt)
    sampled_ssp = torch.empty(req_cnt)
    sampled_top_lp = torch.empty(req_cnt, topk_n)
    sampled_top_ids = torch.empty(req_cnt, topk_n, dtype=torch.int32)
    verified_ids = torch.empty(total_q, dtype=torch.int32)
    verified_lp = torch.empty(total_q)
    verified_ssp = torch.empty(total_q)
    verified_top_lp = torch.empty(total_q, topk_n)
    verified_top_ids = torch.empty(total_q, topk_n, dtype=torch.int32)
    kernel_asc = asc.finalize_verification_tl(True, True, max_steps, num_stop)
    kernel_pto = pto.finalize_verification_tl(True, True, max_steps, num_stop)
    args = [
        effective_accept,
        is_final,
        token_ids,
        token_logprobs,
        sum_smaller,
        top_logprobs,
        top_ids,
        ftp,
        seqlen_q,
        row_to_req,
        is_verify,
        accept_probs,
        confidence,
        confidence_rows,
        stop_ids,
        last_valid,
        num_generated,
        num_verified,
        confidence_error,
        emitted,
        sampled_ids,
        sampled_lp,
        sampled_ssp,
        sampled_top_lp,
        sampled_top_ids,
        verified_ids,
        verified_lp,
        verified_ssp,
        verified_top_lp,
        verified_top_ids,
    ]
    outputs = [
        ("last_valid_offset", 15),
        ("num_generated_tokens", 16),
        ("num_verified_tokens", 17),
        ("batch_confidence_error", 18),
        ("emitted_stop_token", 19),
        ("sampled_token_ids", 20),
        ("sampled_token_logprobs", 21),
        ("sampled_sum_smaller_probs", 22),
        ("sampled_top_logprobs", 23),
        ("sampled_top_token_ids", 24),
        ("verified_token_ids", 25),
        ("verified_token_logprobs", 26),
        ("verified_sum_smaller_probs", 27),
        ("verified_top_logprobs", 28),
        ("verified_top_token_ids", 29),
    ]
    return _launch(kernel_asc, kernel_pto, args, outputs, args)


def _sample_mods():
    _dev()
    from deepinfer_kernels.asc import random_sampling as asc
    from deepinfer_kernels.pto import random_sampling as pto

    return asc, pto


def _prepare_random(vocab: int):
    def prepare():
        asc, pto = _sample_mods()
        batch = 256
        gen = torch.Generator().manual_seed(vocab)
        probs = torch.randn(batch, vocab, generator=gen).softmax(-1).contiguous()
        assert probs.shape[1] == vocab and probs.stride(1) == 1
        # Identical values reach both kernels; _split clones the tensor.
        seq_seeds = (torch.arange(batch, dtype=torch.int64) + 42).to(torch.uint64)
        sample_positions = torch.zeros(batch, dtype=torch.int32)
        out = torch.empty(batch, dtype=torch.int32)
        kernel_asc = asc.random_sampling_tl(vocab, False)
        kernel_pto = pto.random_sampling_tl(vocab, False)
        args = [probs, seq_seeds, sample_positions, out]
        return _launch(kernel_asc, kernel_pto, args, [("token_ids", 3)], args)

    return prepare


def _fused_mods():
    _dev()
    from deepinfer_kernels.asc import fused_sample as asc
    from deepinfer_kernels.pto import fused_sample as pto

    return asc, pto


def _prepare_fused(do_reject: bool):
    def prepare():
        asc, pto = _fused_mods()
        batch, vocab = 256, 1024
        assert vocab % 64 == 0
        gen = torch.Generator().manual_seed(31 + int(do_reject))
        logits = torch.randn(batch, vocab, generator=gen).contiguous()
        assert logits.stride(0) % 8 == 0
        temperatures = torch.ones(batch)
        top_p = torch.full((batch,), 0.95)
        top_k = torch.full((batch,), 128, dtype=torch.int32)
        sample_positions = torch.randint(0, 10**8, (batch,), generator=gen, dtype=torch.int32)
        seq_seeds = (torch.arange(batch, dtype=torch.int64) + 42).to(torch.uint64)
        is_verify = torch.ones(batch, dtype=torch.bool)
        logprobs = torch.empty(batch, vocab)
        candidate = torch.empty(batch, dtype=torch.int32)
        if do_reject:
            draft_lp = torch.randn(batch, vocab, generator=gen).log_softmax(-1).contiguous()
            draft_ids = torch.randint(0, vocab, (batch,), generator=gen, dtype=torch.int32)
            draft_ids.masked_fill_(torch.rand(batch, generator=gen) < 0.1, -1)
        else:
            # Same ranks and dtypes as the reject specialization. The flag drops
            # the reads; the prim_func still lists every buffer.
            draft_lp = torch.zeros(batch, vocab)
            draft_ids = torch.zeros(batch, dtype=torch.int32)
        assert draft_lp.stride(0) % 8 == 0
        read_rows = torch.arange(batch, dtype=torch.int32)
        accept = torch.zeros(batch, dtype=torch.bool)
        draft_selected = torch.zeros(batch)
        accept_probs = torch.zeros(batch)
        lse = torch.empty(batch, 2)
        kernel_asc = asc.fused_sample_tl(vocab, True, do_reject)
        kernel_pto = pto.fused_sample_tl(vocab, True, do_reject)
        args = [
            logits,
            temperatures,
            top_p,
            top_k,
            sample_positions,
            seq_seeds,
            is_verify,
            logprobs,
            candidate,
            draft_lp,
            read_rows,
            draft_ids,
            accept,
            draft_selected,
            accept_probs,
            0,
            lse,
        ]
        outputs = [
            ("logprobs", 7),
            ("candidate_token_ids", 8),
            ("accept_result_verify", 12),
            ("draft_selected_logprobs", 13),
            ("accept_probs", 14),
        ]
        traffic = [logits, temperatures, top_p, top_k, sample_positions, seq_seeds, logprobs, candidate]
        if do_reject:
            traffic.extend((is_verify, draft_lp, read_rows, draft_ids, accept, draft_selected, accept_probs))
        return _launch(kernel_asc, kernel_pto, args, outputs, traffic)

    return prepare


def _transform_mods():
    _dev()
    from deepinfer_kernels.asc import fused_transform as asc
    from deepinfer_kernels.pto import fused_transform as pto

    return asc, pto


def _transform_kwargs(**overrides):
    flags = dict(
        do_norm=True,
        norm_eps=1e-6,
        do_rope=False,
        rotary_dim=0,
        rope_offset=0,
        positions_dtype="int32",
        is_neox=False,
        conjugate=False,
        do_hadamard=False,
        hadamard_scale=1.0,
        cast_fmt="",
        num_per_channels=0,
        round_sf=False,
        sf_format="",
        preserve_last_group=False,
        return_bf16=False,
        bf16_stage="transform",
        return_quant=False,
        cache_layout=None,
        block_size=1,
    )
    flags.update(overrides)
    return flags


def prepare_norm_rope():
    asc, pto = _transform_mods()
    num_heads, head_dim, n, rotary_dim = 8, 128, 4096, 64
    gen = torch.Generator().manual_seed(37)
    x = torch.randn(n, num_heads, head_dim, generator=gen, dtype=torch.bfloat16)
    weight = torch.randn(head_dim, generator=gen, dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (n,), generator=gen, dtype=torch.int32)
    cossin = torch.randn(4096, rotary_dim, generator=gen)
    assert int(positions.max()) < cossin.shape[0]
    rows = n * num_heads
    out_bf16 = torch.empty(n, num_heads, head_dim, dtype=torch.bfloat16)
    # Signature slots the no-cast kernel still declares and does not write.
    out_quant = torch.empty(rows, head_dim, dtype=torch.float8_e4m3fn)
    out_sf = torch.empty(rows, 1)
    kv_cache = torch.empty(1, 1, head_dim, dtype=torch.uint8)
    slot_mapping = torch.zeros(n, dtype=torch.int32)
    flags = _transform_kwargs(
        do_rope=True,
        rotary_dim=rotary_dim,
        rope_offset=head_dim - rotary_dim,
        return_bf16=True,
    )
    kernel_asc = asc.fused_norm_rope_hadamard_cast_store_kvcache_tl(num_heads, head_dim, **flags)
    kernel_pto = pto.fused_norm_rope_hadamard_cast_store_kvcache_tl(num_heads, head_dim, **flags)
    args = [x, weight, positions, cossin, None, out_bf16, out_quant, out_sf, kv_cache, slot_mapping]
    return _launch(kernel_asc, kernel_pto, args, [("bf16", 5)], [x, weight, positions, cossin, out_bf16])


def prepare_norm_fp8():
    asc, pto = _transform_mods()
    num_heads, head_dim, n = 8, 128, 1024
    group = 32  # e4m3 ue8m0 groups the kernel allows: 32, 64, 128
    num_scales = head_dim // group
    gen = torch.Generator().manual_seed(41)
    x = torch.randn(n, num_heads, head_dim, generator=gen, dtype=torch.bfloat16)
    weight = torch.randn(head_dim, generator=gen, dtype=torch.bfloat16)
    rows = n * num_heads
    positions = torch.zeros(n, dtype=torch.int32)
    cossin = torch.zeros(1, 1)
    out_bf16 = torch.empty(rows, head_dim, dtype=torch.bfloat16)
    out_quant = torch.empty(n, num_heads, head_dim, dtype=torch.float8_e4m3fn)
    out_sf = torch.empty(n, num_heads, num_scales, dtype=torch.uint8)
    kv_cache = torch.empty(1, 1, head_dim + num_scales, dtype=torch.uint8)
    slot_mapping = torch.zeros(n, dtype=torch.int32)
    flags = _transform_kwargs(
        cast_fmt="e4m3",
        num_per_channels=group,
        round_sf=True,
        sf_format="ue8m0",
        return_quant=True,
    )
    kernel_asc = asc.fused_norm_rope_hadamard_cast_store_kvcache_tl(num_heads, head_dim, **flags)
    kernel_pto = pto.fused_norm_rope_hadamard_cast_store_kvcache_tl(num_heads, head_dim, **flags)
    args = [x, weight, positions, cossin, None, out_bf16, out_quant, out_sf, kv_cache, slot_mapping]
    asc_fn, pto_fn, compared, nbytes = _launch(
        kernel_asc,
        kernel_pto,
        args,
        [("quant", 6), ("sf", 7)],
        [x, weight, out_quant, out_sf],
    )
    # CPU allclose has no float8 kernel. ue8m0 scales are uint8 exponent codes.
    shown = []
    for label, asc_out, pto_out in compared:
        shown.append((label, asc_out.float(), pto_out.float()))
    return asc_fn, pto_fn, shown, nbytes


_CACHE_NOTE = "KV-cache store skipped: store_cache requires num_heads==1 and a paged slot_mapping; num_heads=8"

CASES: list[BenchCase] = [
    BenchCase("sampler", "prepare_verification_metadata", prepare_verification, notes="req 256, max_steps 7"),
    BenchCase("sampler", "expand_sampling_params", prepare_expand, notes="req 256, max_steps 7"),
    BenchCase(
        "sampler",
        "map_lookup",
        prepare_map_lookup,
        notes="by_key float32, rows 256, force_topk 1024",
    ),
    BenchCase(
        "sampler",
        "remap_draft_topk",
        prepare_remap,
        notes="total_q 256, force_topk 1024, vocab 129280, pool 43136",
    ),
    BenchCase(
        "sampler",
        "get_accept_result",
        prepare_accept,
        notes="total_q 256, has_golden, request length 16",
    ),
    BenchCase(
        "sampler",
        "finalize_verification",
        prepare_finalize,
        exact=False,
        rtol=1e-3,
        atol=1e-3,
        notes="req 256, max_steps 3, topk 64, confidence and ssp",
    ),
    BenchCase(
        "sample",
        "b256_v1024",
        _prepare_random(1024),
        notes="is_nextn_predict False, shared seeds, token ids",
    ),
    BenchCase(
        "sample",
        "b256_v129280",
        _prepare_random(129280),
        notes="is_nextn_predict False, shared seeds, token ids",
    ),
    BenchCase(
        "fused_sample",
        "b256_v1024",
        _prepare_fused(False),
        exact=False,
        rtol=2e-2,
        atol=2e-2,
        notes="do_sample, top_p 0.95, top_k 128, shared seeds",
    ),
    BenchCase(
        "fused_sample",
        "b256_v1024_reject",
        _prepare_fused(True),
        exact=False,
        rtol=2e-2,
        atol=2e-2,
        notes="do_reject_sampling, same tensor list, shared seeds",
    ),
    BenchCase(
        "fused_transform",
        "norm_rope_h8_d128_n4096",
        prepare_norm_rope,
        exact=False,
        rtol=2e-2,
        atol=2e-2,
        notes=f"bf16 norm+rope, rotary 64; {_CACHE_NOTE}",
    ),
    BenchCase(
        "fused_transform",
        "norm_fp8_h8_d128_n1024",
        prepare_norm_fp8,
        exact=False,
        rtol=2e-1,
        atol=1e-1,
        notes=f"norm + e4m3 ue8m0 group 32; {_CACHE_NOTE}",
    ),
]
