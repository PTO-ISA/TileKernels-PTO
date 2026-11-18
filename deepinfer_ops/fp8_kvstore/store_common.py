"""Shared non-VF sequential V4.1 FP8 store implementation."""

from typing import Any

import tile_kernels
from tilelang.ascend import language as T

BLOCK_SIZE = 64
TOKENS_PER_TILE = 16
NUM_STAGES = 3
NUM_ELEMENTS = 512
DATA_BYTES = 512
SF_BYTES = 16
PAYLOAD_BYTES = DATA_BYTES + SF_BYTES
OUT_ROW_BYTES = (PAYLOAD_BYTES + 31) // 32 * 32
SF_COL = DATA_BYTES


def make_store_tl(quantize_tile: Any) -> Any:
    n = T.dynamic("n")
    num_blocks = T.dynamic("num_blocks")
    cache_block_stride = T.dynamic("cache_block_stride", dtype="int64")
    cache_token_stride = T.dynamic("cache_token_stride", dtype="int64")
    cache_strides = (cache_block_stride, cache_token_stride, 1)
    num_cores = tile_kernels.get_num_vec_cores()

    @T.macro
    def classify_slots(slots: T.Buffer, active: Any):
        flags = T.alloc_var(T.int32, 2)
        first = slots[0]
        if first == -1:
            flags = 0
        for h in T.serial(active):
            slot = slots[h]
            if slot != -1:
                flags = flags | 1
            if slot != first + h:
                flags = flags & 1
        return flags

    @T.macro
    def write_sections(out: T.Buffer, scales: T.Buffer, cache: T.Buffer, h: Any, count: Any, blk: Any, off: Any):
        del scales
        out_bytes = T.view(out, (TOKENS_PER_TILE, OUT_ROW_BYTES), T.uint8)
        T.copy(out_bytes[h : h + count, :PAYLOAD_BYTES], cache[blk, off : off + count, :PAYLOAD_BYTES])

    @T.macro
    def write_run(out: T.Buffer, scales: T.Buffer, cache: T.Buffer, active: Any, blk: Any, off: Any):
        first_count = T.min(active, BLOCK_SIZE - off)
        write_sections(out, scales, cache, 0, first_count, blk, off)
        if first_count < active:
            write_sections(out, scales, cache, first_count, active - first_count, blk + 1, 0)

    @T.prim_func
    def _store_kv_cache(
        kv: T.Tensor((n, NUM_ELEMENTS), T.bfloat16),
        kv_cache: T.StridedTensor((num_blocks, BLOCK_SIZE, PAYLOAD_BYTES), cache_strides, T.uint8),
        slot_mapping: T.Tensor((n,), T.int32),
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((TOKENS_PER_TILE, NUM_ELEMENTS), T.bfloat16)
            out_ub = T.alloc_shared((TOKENS_PER_TILE, OUT_ROW_BYTES), T.float8_e4m3fn)
            slots_ub = T.alloc_shared((TOKENS_PER_TILE,), T.int32)
            sf_ub = T.view(out_ub, (TOKENS_PER_TILE, OUT_ROW_BYTES), T.uint8)
            T.annotate_buffer_versions({x_ub: NUM_STAGES, out_ub: NUM_STAGES, slots_ub: NUM_STAGES})
            for tile in T.Persistent(
                [T.ceildiv(n, TOKENS_PER_TILE)], num_cores, core_id, group_size=1, num_stages=NUM_STAGES
            ):
                start = tile * TOKENS_PER_TILE
                active = T.min(TOKENS_PER_TILE, n - start)
                T.copy(slot_mapping[start : start + active], slots_ub[0:active])
                metadata_flags = classify_slots(slots_ub, active)
                if metadata_flags & 1 != 0:
                    T.copy(kv[start : start + active, :], x_ub[0:active, :], l2_cache_ctrl="NOTALLOC_KEEP")
                    quantize_tile(x_ub, out_ub, sf_ub, active)
                    if metadata_flags & 2 != 0:
                        first = slots_ub[0]
                        write_run(out_ub, sf_ub, kv_cache, active, first // BLOCK_SIZE, first % BLOCK_SIZE)
                    else:
                        for h in T.serial(active):
                            slot = slots_ub[h]
                            if slot != -1:
                                write_sections(out_ub, sf_ub, kv_cache, h, 1, slot // BLOCK_SIZE, slot % BLOCK_SIZE)

    return _store_kv_cache
