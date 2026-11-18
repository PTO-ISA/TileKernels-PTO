import tilelang.contrib.ptodsl as tl
from ptodsl import pto

@pto.jit(name="_store_kv_cache_kernel", kernel_kind="vector", target="a5", mode="explicit")
def _store_kv_cache_kernel(kv: pto.ptr(pto.bf16, "gm"), kv_cache: pto.ptr(pto.ui8, "gm"), slot_mapping: pto.ptr(pto.si32, "gm"), cache_block_stride: pto.si64, cache_token_stride: pto.si64, n: pto.si32, num_blocks: pto.si32):
  pto.init_core()
  bx = pto.cast(pto.get_block_idx(), pto.si32)
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  flags = pto.const(0, dtype=pto.si32)
  slots_ub_version_counter_1 = pto.const(0, dtype=pto.si32)
  x_ub_version_counter_2 = pto.const(0, dtype=pto.si32)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  pto.set_flag("S", "MTE2", event_id=0)
  pto.set_flag("S", "MTE2", event_id=1)
  pto.set_flag("S", "MTE2", event_id=2)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  slots_ub_version_counter_1 = pto.const(0, dtype=pto.si32)
  x_ub_version_counter_2 = pto.const(0, dtype=pto.si32)
  for _tl_range_w in range(0, pto.div((n + 1151), 1152), 1):
    w = pto.cast(_tl_range_w, pto.si32)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      pto.wait_flag("S", "MTE2", event_id=(3 & ((slots_ub_version_counter_1 % 3) >> 31)) + (slots_ub_version_counter_1 % 3))
      pto.mte_gm_ub(pto.castptr(pto.addptr(slot_mapping, (pto.cast(w, pto.si64) * 1152) + (pto.cast(bx, pto.si64) * 16)), pto.ptr(pto.ui8, "gm")), pto.castptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub")), (((3 & ((slots_ub_version_counter_1 % 3) >> 31)) * 16) + ((slots_ub_version_counter_1 % 3) * 16)) + 18816), pto.ptr(pto.ui8, "ub")), 0, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 4, nburst=(1, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 4, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 4))
      pto.set_flag("MTE2", "S", event_id=(3 & ((slots_ub_version_counter_1 % 3) >> 31)) + (slots_ub_version_counter_1 % 3))
      pto.wait_flag("MTE2", "S", event_id=(3 & ((slots_ub_version_counter_1 % 3) >> 31)) + (slots_ub_version_counter_1 % 3))
    first = pto.const(0, dtype=pto.si32)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      first = pto.cast(pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub")), (((3 & ((slots_ub_version_counter_1 % 3) >> 31)) * 16) + ((slots_ub_version_counter_1 % 3) * 16)) + 18816), pto.si32)
    tl_cond_0 = pto.const(0, dtype=pto.si8)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      tl_cond_0 = pto.cast(pto.select(tl.as_logical_bool(first == -1), pto.const(1, dtype=pto.si8), pto.const(0, dtype=pto.si8)), pto.si8)
      flags = pto.const(2, dtype=pto.si32)
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_0)):
        flags = pto.const(0, dtype=pto.si32)
      for _tl_range_h in range(0, pto.min(16, ((n - (bx * 16)) - (w * 1152))), 1):
        h = pto.cast(_tl_range_h, pto.si32)
        slot = pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub")), ((((3 & ((slots_ub_version_counter_1 % 3) >> 31)) * 16) + ((slots_ub_version_counter_1 % 3) * 16)) + h) + 18816)
        tl_cond_2 = slot != (first + h)
        if tl.as_logical_bool(slot != -1):
          flags = pto.cast(flags | 1, pto.si32)
        if tl.as_logical_bool(tl_cond_2):
          flags = pto.cast(flags & 1, pto.si32)
    metadata_flags = pto.const(0, dtype=pto.si32)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      metadata_flags = pto.cast(flags, pto.si32)
    tl_cond_6 = pto.const(0, dtype=pto.si8)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      tl_cond_6 = pto.cast(pto.select(tl.as_logical_bool(0 < (metadata_flags & 1)), pto.const(1, dtype=pto.si8), pto.const(0, dtype=pto.si8)), pto.si8)
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.wait_flag("V", "MTE2", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        pto.mte_gm_ub(pto.castptr(pto.addptr(kv, (pto.cast(w, pto.si64) * 589824) + (pto.cast(bx, pto.si64) * 8192)), pto.ptr(pto.ui8, "gm")), pto.castptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)), pto.ptr(pto.ui8, "ub")), 4, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 1024, nburst=(1, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 1024, pto.min(16, ((n - (bx * 16)) - (w * 1152))) * 1024))
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.set_flag("MTE2", "V", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    tl_cond_5 = pto.const(0, dtype=pto.si8)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        tl_cond_5 = pto.cast(pto.select(tl.as_logical_bool(0 < (metadata_flags & 2)), pto.const(1, dtype=pto.si8), pto.const(0, dtype=pto.si8)), pto.si8)
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.wait_flag("MTE3", "V", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.wait_flag("MTE2", "V", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        with pto.vecscope():
          mask = pto.vmi.create_mask(64, size=64)
          one = pto.vmi.vbrc(pto.ui32(1), size=64)
          bias = pto.vmi.vbrc(pto.ui32(254), size=64)
          for _tl_range_row in range(0, pto.min(16, ((n - (bx * 16)) - (w * 1152))), 1):
            row = pto.cast(_tl_range_row, pto.si32)
            values = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)), 0, size=64), to_dtype=pto.f32)
            mask_1 = pto.vmi.create_mask(64, size=64)
            amax = pto.vmi.vcmax(pto.vmi.vabs(values, mask_1), mask_1, group=2)
            amax_1 = pto.vmi.vbrc(amax, group=2, size=64)
            clamped = pto.vmi.vmaxs(amax_1, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits, one, mask), 23, mask), 1, mask)
            inverse = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense, mask), 23, mask), to_dtype=pto.f32)
            scale = pto.vmi.vcmax(encoded_dense, mask_1, group=2)
            packed = pto.vmi.vcvt(pto.vmi.vmul(values, inverse, mask_1), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49152), 0)
            mask_2 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49664), 0, mask_2)
            values_1 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 64), 0, size=64), to_dtype=pto.f32)
            mask_3 = pto.vmi.create_mask(64, size=64)
            amax_2 = pto.vmi.vcmax(pto.vmi.vabs(values_1, mask_3), mask_3, group=2)
            amax_3 = pto.vmi.vbrc(amax_2, group=2, size=64)
            clamped_1 = pto.vmi.vmaxs(amax_3, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_1 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_1, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_1 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_1, one, mask), 23, mask), 1, mask)
            inverse_1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_1, mask), 23, mask), to_dtype=pto.f32)
            scale_1 = pto.vmi.vcmax(encoded_dense_1, mask_3, group=2)
            packed_1 = pto.vmi.vcvt(pto.vmi.vmul(values_1, inverse_1, mask_3), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_1, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49216), 0)
            mask_4 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_1, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49666), 0, mask_4)
            values_2 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 128), 0, size=64), to_dtype=pto.f32)
            mask_5 = pto.vmi.create_mask(64, size=64)
            amax_4 = pto.vmi.vcmax(pto.vmi.vabs(values_2, mask_5), mask_5, group=2)
            amax_5 = pto.vmi.vbrc(amax_4, group=2, size=64)
            clamped_2 = pto.vmi.vmaxs(amax_5, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_2 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_2, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_2 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_2, one, mask), 23, mask), 1, mask)
            inverse_2 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_2, mask), 23, mask), to_dtype=pto.f32)
            scale_2 = pto.vmi.vcmax(encoded_dense_2, mask_5, group=2)
            packed_2 = pto.vmi.vcvt(pto.vmi.vmul(values_2, inverse_2, mask_5), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_2, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49280), 0)
            mask_6 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_2, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49668), 0, mask_6)
            values_3 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 192), 0, size=64), to_dtype=pto.f32)
            mask_7 = pto.vmi.create_mask(64, size=64)
            amax_6 = pto.vmi.vcmax(pto.vmi.vabs(values_3, mask_7), mask_7, group=2)
            amax_7 = pto.vmi.vbrc(amax_6, group=2, size=64)
            clamped_3 = pto.vmi.vmaxs(amax_7, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_3 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_3, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_3 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_3, one, mask), 23, mask), 1, mask)
            inverse_3 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_3, mask), 23, mask), to_dtype=pto.f32)
            scale_3 = pto.vmi.vcmax(encoded_dense_3, mask_7, group=2)
            packed_3 = pto.vmi.vcvt(pto.vmi.vmul(values_3, inverse_3, mask_7), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_3, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49344), 0)
            mask_8 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_3, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49670), 0, mask_8)
            values_4 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 256), 0, size=64), to_dtype=pto.f32)
            mask_9 = pto.vmi.create_mask(64, size=64)
            amax_8 = pto.vmi.vcmax(pto.vmi.vabs(values_4, mask_9), mask_9, group=2)
            amax_9 = pto.vmi.vbrc(amax_8, group=2, size=64)
            clamped_4 = pto.vmi.vmaxs(amax_9, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_4 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_4, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_4 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_4, one, mask), 23, mask), 1, mask)
            inverse_4 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_4, mask), 23, mask), to_dtype=pto.f32)
            scale_4 = pto.vmi.vcmax(encoded_dense_4, mask_9, group=2)
            packed_4 = pto.vmi.vcvt(pto.vmi.vmul(values_4, inverse_4, mask_9), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_4, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49408), 0)
            mask_10 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_4, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49672), 0, mask_10)
            values_5 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 320), 0, size=64), to_dtype=pto.f32)
            mask_11 = pto.vmi.create_mask(64, size=64)
            amax_10 = pto.vmi.vcmax(pto.vmi.vabs(values_5, mask_11), mask_11, group=2)
            amax_11 = pto.vmi.vbrc(amax_10, group=2, size=64)
            clamped_5 = pto.vmi.vmaxs(amax_11, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_5 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_5, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_5 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_5, one, mask), 23, mask), 1, mask)
            inverse_5 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_5, mask), 23, mask), to_dtype=pto.f32)
            scale_5 = pto.vmi.vcmax(encoded_dense_5, mask_11, group=2)
            packed_5 = pto.vmi.vcvt(pto.vmi.vmul(values_5, inverse_5, mask_11), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_5, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49472), 0)
            mask_12 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_5, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49674), 0, mask_12)
            values_6 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 384), 0, size=64), to_dtype=pto.f32)
            mask_13 = pto.vmi.create_mask(64, size=64)
            amax_12 = pto.vmi.vcmax(pto.vmi.vabs(values_6, mask_13), mask_13, group=2)
            amax_13 = pto.vmi.vbrc(amax_12, group=2, size=64)
            clamped_6 = pto.vmi.vmaxs(amax_13, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_6 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_6, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_6 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_6, one, mask), 23, mask), 1, mask)
            inverse_6 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_6, mask), 23, mask), to_dtype=pto.f32)
            scale_6 = pto.vmi.vcmax(encoded_dense_6, mask_13, group=2)
            packed_6 = pto.vmi.vcvt(pto.vmi.vmul(values_6, inverse_6, mask_13), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_6, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49536), 0)
            mask_14 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_6, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49676), 0, mask_14)
            values_7 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8192) + ((x_ub_version_counter_2 % 3) * 8192)) + (row * 512)) + 448), 0, size=64), to_dtype=pto.f32)
            mask_15 = pto.vmi.create_mask(64, size=64)
            amax_14 = pto.vmi.vcmax(pto.vmi.vabs(values_7, mask_15), mask_15, group=2)
            amax_15 = pto.vmi.vbrc(amax_14, group=2, size=64)
            clamped_7 = pto.vmi.vmaxs(amax_15, float.fromhex('0x1.a36e2eb1c432dp-14'), mask)
            bits_7 = pto.vmi.vinterpret_cast(pto.vmi.vmuls(clamped_7, float.fromhex('0x1.2492492492492p-9'), mask), to_dtype=pto.ui32)
            encoded_dense_7 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits_7, one, mask), 23, mask), 1, mask)
            inverse_7 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(bias, encoded_dense_7, mask), 23, mask), to_dtype=pto.f32)
            scale_7 = pto.vmi.vcmax(encoded_dense_7, mask_15, group=2)
            packed_7 = pto.vmi.vcvt(pto.vmi.vmul(values_7, inverse_7, mask_15), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            pto.vmi.vstore(packed_7, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49600), 0)
            mask_16 = pto.vmi.create_mask(2, size=2)
            pto.vmi.vstore(pto.vmi.vcvt(scale_7, to_dtype=pto.ui8), pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (row * 544)) + 49678), 0, mask_16)
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.set_flag("V", "MTE3", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.set_flag("V", "MTE2", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.pipe_barrier("MTE3")
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.wait_flag("V", "MTE3", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        if tl.as_logical_bool(tl.logical_not(tl.as_logical_bool(tl_cond_5))):
          for _tl_range_h_1 in range(0, pto.min(16, ((n - (bx * 16)) - (w * 1152))), 1):
            h_1 = pto.cast(_tl_range_h_1, pto.si32)
            slot_1 = pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub")), ((((3 & ((slots_ub_version_counter_1 % 3) >> 31)) * 16) + ((slots_ub_version_counter_1 % 3) * 16)) + h_1) + 18816)
            if tl.as_logical_bool(slot_1 != -1):
              pto.pipe_barrier("MTE3")
              if tl.as_logical_bool(0 < (num_blocks - (slot_1 >> 6))):
                pto.mte_ub_gm(pto.addptr(buf_dyn_shmem, ((((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + (h_1 * 544)) + 49152), pto.addptr(kv_cache, ((pto.cast(slot_1, pto.si64) >> 6) * cache_block_stride) + ((pto.cast(slot_1, pto.si64) & 63) * cache_token_stride)), 528, nburst=(1, 528, 528), l2_cache="naci")
    first_1 = pto.const(0, dtype=pto.si32)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        if tl.as_logical_bool(tl.as_logical_bool(tl_cond_5)):
          first_1 = pto.cast(pto.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si32, "ub")), (((3 & ((slots_ub_version_counter_1 % 3) >> 31)) * 16) + ((slots_ub_version_counter_1 % 3) * 16)) + 18816), pto.si32)
      pto.set_flag("S", "MTE2", event_id=(3 & ((slots_ub_version_counter_1 % 3) >> 31)) + (slots_ub_version_counter_1 % 3))
      slots_ub_version_counter_1 = pto.cast(slots_ub_version_counter_1 + 1, pto.si32)
    tl_cond_3 = pto.const(0, dtype=pto.si8)
    if tl.as_logical_bool(((w * 72) + bx) < ((n + 15) >> 4)):
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        if tl.as_logical_bool(tl.as_logical_bool(tl_cond_5)):
          tl_cond_3 = pto.cast(pto.select(tl.as_logical_bool((48 < (first_1 & 63)) and (64 < (((n + (first_1 & 63)) - (bx * 16)) - (w * 1152)))), pto.const(1, dtype=pto.si8), pto.const(0, dtype=pto.si8)), pto.si8)
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        if tl.as_logical_bool(tl.as_logical_bool(tl_cond_5)):
          if tl.as_logical_bool(0 < (num_blocks - (first_1 >> 6))):
            pto.mte_ub_gm(pto.addptr(buf_dyn_shmem, (((3 & ((x_ub_version_counter_2 % 3) >> 31)) * 8704) + ((x_ub_version_counter_2 % 3) * 8704)) + 49152), pto.addptr(kv_cache, ((pto.cast(first_1, pto.si64) >> 6) * cache_block_stride) + ((pto.cast(first_1, pto.si64) & 63) * cache_token_stride)), 528, nburst=(pto.min(pto.min(16, ((n - (bx * 16)) - (w * 1152))), (64 - (first_1 & 63))), 544, cache_token_stride), l2_cache="naci")
      if tl.as_logical_bool(tl.as_logical_bool(tl_cond_6)):
        if tl.as_logical_bool(tl.as_logical_bool(tl_cond_5)):
          if tl.as_logical_bool(tl.as_logical_bool(tl_cond_3)):
            if tl.as_logical_bool(((64 < (pto.min(16, ((n - (bx * 16)) - (w * 1152))) + (first_1 & 63))) and (((((n >> 4) - bx) - (w * 72)) < 1) or (48 < (first_1 & 63)))) and (1 < (num_blocks - (first_1 >> 6)))):
              pto.mte_ub_gm(pto.addptr(buf_dyn_shmem, ((((3 & ((pto.cast(x_ub_version_counter_2, pto.si64) % 3) >> 63)) * 8704) + ((pto.cast(x_ub_version_counter_2, pto.si64) % 3) * 8704)) + (pto.min(pto.min(16, ((pto.cast(n, pto.si64) - (pto.cast(bx, pto.si64) * 16)) - (pto.cast(w, pto.si64) * 1152))), (64 - (pto.cast(first_1, pto.si64) & 63))) * 544)) + 49152), pto.addptr(kv_cache, ((pto.cast(first_1, pto.si64) >> 6) + 1) * cache_block_stride), 528, nburst=(pto.min(pto.max(0, ((pto.min(16, ((n - (bx * 16)) - (w * 1152))) + (first_1 & 63)) - 64)), (16 - pto.min(pto.min(16, ((n - (bx * 16)) - (w * 1152))), (64 - (first_1 & 63))))), 544, cache_token_stride), l2_cache="naci")
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      pto.set_flag("MTE3", "V", event_id=(3 & ((x_ub_version_counter_2 % 3) >> 31)) + (x_ub_version_counter_2 % 3))
    if tl.as_logical_bool((((w * 72) + bx) < ((n + 15) >> 4)) and tl.as_logical_bool(tl_cond_6)):
      x_ub_version_counter_2 = pto.cast(x_ub_version_counter_2 + 1, pto.si32)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)
  pto.wait_flag("S", "MTE2", event_id=0)
  pto.wait_flag("S", "MTE2", event_id=1)
  pto.wait_flag("S", "MTE2", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)
