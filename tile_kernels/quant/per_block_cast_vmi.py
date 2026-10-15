import math
import tilelang

from tile_kernels.utils import align
from tile_kernels.quant.common import *

# Re-bind below the star import: tile_kernels.quant.common exports its own `T`.
from tilelang.ascend import language as T
from tilelang.ascend.language import vmi as V


@tilelang.jit(target='pto', pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def get_per_block_cast_kernel_vmi(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    num_vec_cores: int,
):
    num_per_tokens, num_per_channels = out_config.sf_block
    assert (num_per_tokens, num_per_channels) == (32, 32)
    assert out_config.dtype in (T.float4_e2m1fn, T.float8_e4m3fn)
    quant_max = 6.0 if out_config.dtype == T.float4_e2m1fn else 448.0

    block_k_limit = 512 if in_config.dtype == T.float32 else 1024
    block_k = align(math.gcd(block_k_limit, hidden), 128)
    block_m = num_per_tokens
    num_stages = 2

    num_sf_cols_per_block = block_k // num_per_channels
    pack_factor = get_packed_ue8m0_pack_factor()
    assert num_sf_cols_per_block % pack_factor == 0
    packed_sf_cols_per_block = num_sf_cols_per_block // pack_factor
    num_stored_sf_cols_per_block = packed_sf_cols_per_block if out_config.use_packed_ue8m0 else num_sf_cols_per_block
    sf_ub_dtype = T.uint8 if out_config.use_packed_ue8m0 else T.float32

    token_group = 4 if out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0 else 1

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    @T.macro
    def f32_to_bf16_round_odd(x, mask):  # round_odd=False means round_to_zero here
        # in this kernel, we ensure that round_odd=False is used only when the low 16bit of fp32 inputs are all zero.
        bits = V.vinterpret_cast(x, 'uint32')
        low, high = V.vunzip(bits, to_dtype='uint16')
        return V.vinterpret_cast(V.vor(high, V.vmins(low, T.uint16(1), mask)), 'bfloat16')

    @T.macro
    def get_scale_and_inv(amax, fp8_max, inv_fp8_max, mask, lanes):
        if not out_config.round_sf:
            return V.vdiv(amax, fp8_max, mask, precision='exact'), V.vdiv(fp8_max, amax, mask, precision='exact')
        raw = V.vmul(amax, inv_fp8_max)
        bits = V.vinterpret_cast(raw, 'uint32')
        exp = V.vadds(V.vshrs(V.vsub(bits, V.vbrc(T.uint32(1), size=lanes), mask), 23, mask), T.uint32(1), mask)
        scale = V.vinterpret_cast(V.vshls(exp, 23, mask), 'float32')
        inv = V.vinterpret_cast(V.vshls(V.vsub(V.vbrc(T.uint32(254), size=lanes), exp, mask), 23, mask), 'float32')
        return scale, inv

    @T.macro
    def store_scale_four(sf_ub, sf_row, sf_j, scale4, mask4):
        if out_config.use_packed_ue8m0:
            _, high16 = V.vunzip(V.vinterpret_cast(scale4, 'uint32'), to_dtype='uint16')
            exponent16 = V.vshrs(high16, 7, mask4)
            exponent8 = V.vcvt(exponent16, 'uint8', saturate='NOSAT')
            if out_config.use_tma_aligned_col_major_sf:
                V.vstore(exponent8, sf_ub[sf_j // pack_factor, 0], V.create_mask(4, size=4))
            else:
                V.vstore(exponent8, sf_ub[0, sf_j], V.create_mask(4, size=4))
        elif out_config.use_tma_aligned_col_major_sf:
            V.vstore(scale4, sf_ub[sf_j, sf_row], group=4, stride=token_group)
        else:
            V.vstore(scale4, sf_ub[sf_row, sf_j], group=4, stride=1)

    @T.macro
    def load_scale_four(sf_ub, sf_j):
        slots = V.vload(sf_ub[0, sf_j], size=4, group=4, stride=1)
        return V.vbrc(slots, size=128, group=4)

    @T.macro
    def quant_block(x_ub, sf_ub, out_ub, sf_row):
        with T.SimdVF():
            mask4 = V.create_mask(1, size=4, group=4)
            mask128 = V.create_mask(128, size=128)
            one128 = V.vbrc(T.float32(1.0), size=128)
            eps128 = V.vbrc(T.float32(out_config.clamp_min_value), size=128)
            fp8_max4 = V.vbrc(T.float32(quant_max), size=4)
            inv_fp8_max4 = V.vbrc(T.float32(1.0 / quant_max), size=4)
            u254 = V.vbrc(T.uint32(254), size=128)

            for chunk in T.serial(block_k // 128):
                col = chunk * 128
                sf_col = chunk * 4  # 4 = 128 // num_per_channels
                if cast_only:
                    scale128 = load_scale_four(sf_ub, sf_col)
                    if out_config.round_sf:
                        exponent = V.vshrs(V.vinterpret_cast(scale128, 'uint32'), 23, mask128)
                        inverse_bits = V.vshls(V.vsub(u254, exponent, mask128), 23, mask128)
                        inverse_vec = V.vinterpret_cast(inverse_bits, 'float32')
                    else:
                        inverse_vec = V.vdiv(one128, scale128, mask128, precision='exact')
                else:
                    amax128 = V.alloc_var(T.float32, size=128)
                    if in_config.dtype == T.bfloat16:
                        amax_bf16 = V.alloc_var(T.bfloat16, size=128)
                        amax_bf16 = V.vbrc(T.bfloat16(0.0), size=128)
                        for row in T.serial(num_per_tokens):
                            amax_bf16 = V.vmax(amax_bf16, V.vabs(V.vload(x_ub[row, col], size=128)))
                        amax128 = V.vcvt(amax_bf16, 'float32')
                    else:
                        amax128 = V.vbrc(T.float32(0.0), size=128)
                        for row in T.serial(num_per_tokens):
                            amax128 = V.vmax(amax128, V.vabs(V.vload(x_ub[row, col], size=128)))
                    amax128 = V.vmax(amax128, eps128)
                    amax4 = V.vcmax(amax128, mask128, group=4)
                    scale4, inverse4 = get_scale_and_inv(amax4, fp8_max4, inv_fp8_max4, mask4, 4)
                    inverse_vec = V.vbrc(inverse4, size=128, group=4)
                    store_scale_four(sf_ub, sf_row, sf_col, scale4, mask4)

                if in_config.dtype == T.bfloat16 and out_config.round_sf:
                    sf_inv = V.vcvt(inverse_vec, 'bfloat16', rounding='Z')

                if not sf_only:
                    for row in T.serial(num_per_tokens):
                        if in_config.dtype == T.bfloat16 and out_config.round_sf:
                            quantized_bf16 = V.vmul(V.vload(x_ub[row, col], size=128), sf_inv)
                            if out_config.dtype == T.float8_e4m3fn:
                                quantized = V.vcvt(quantized_bf16, 'float32')
                        else:
                            values = V.vload(x_ub[row, col], size=128)
                            if in_config.dtype == T.bfloat16:
                                values_f32 = V.vcvt(values, 'float32')
                            else:
                                values_f32 = values
                            quantized = V.vmul(values_f32, inverse_vec)
                            if out_config.dtype == T.float4_e2m1fn:
                                quantized_bf16 = f32_to_bf16_round_odd(quantized, mask128)

                        if out_config.dtype == T.float4_e2m1fn:
                            V.vstore(V.vcvt(quantized_bf16, 'float4_e2m1fn', rounding='R'), out_ub[row, col])
                        else:
                            V.vstore(V.vcvt(quantized, 'float8_e4m3fn', rounding='R', saturate='SAT'), out_ub[row, col])

    @T.prim_func
    def per_block_cast_kernel_vmi(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_cols = max(num_stored_sf_cols_per_block, 16 // token_group if out_config.use_packed_ue8m0 else 8)
            if out_config.use_packed_ue8m0 and out_config.use_tma_aligned_col_major_sf:
                sf_ub = T.alloc_shared((sf_cols, pack_factor), sf_ub_dtype)
            elif out_config.use_packed_ue8m0:
                sf_ub = T.alloc_shared((1, sf_cols * pack_factor), sf_ub_dtype)
            elif out_config.use_tma_aligned_col_major_sf:
                sf_ub = T.alloc_shared((sf_cols, token_group), sf_ub_dtype)
            else:
                sf_ub = T.alloc_shared((token_group, sf_cols), sf_ub_dtype)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)

            versions = {x_ub: num_stages, sf_ub: num_stages}
            if not sf_only:
                versions[out_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_x, pid_y in T.Persistent(
                [T.ceildiv(num_tokens, block_m * token_group), T.ceildiv(hidden, block_k)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                sf_m_base = T.alloc_var(T.int32, init=pid_x * token_group)
                sf_k_base = T.alloc_var(T.int32, init=pid_y * num_sf_cols_per_block)
                stored_sf_k_base = T.alloc_var(T.int32, init=pid_y * num_stored_sf_cols_per_block)
                for token_offset in T.serial(token_group):
                    token_block = sf_m_base + token_offset
                    T.copy(x[token_block * block_m, pid_y * block_k], x_ub[0:block_m, 0:block_k])
                    if cast_only:
                        T.copy(out_sf[token_block, sf_k_base], sf_ub[0:1, 0:num_sf_cols_per_block])
                    quant_block(x_ub, sf_ub, out_ub, token_offset)
                    if not sf_only:
                        T.copy(out_ub[0:block_m, 0:block_k], out[token_block * block_m, pid_y * block_k])
                    if not cast_only and out_config.use_packed_ue8m0:
                        if out_config.use_tma_aligned_col_major_sf:
                            T.copy(
                                sf_ub[0:num_stored_sf_cols_per_block, 0:pack_factor],
                                out_sf[stored_sf_k_base, token_block * pack_factor],
                            )
                        else:
                            T.copy(
                                sf_ub[0:1, 0:num_stored_sf_cols_per_block * pack_factor],
                                out_sf[token_block, stored_sf_k_base * pack_factor],
                            )

                if not cast_only and not out_config.use_packed_ue8m0:
                    if out_config.use_tma_aligned_col_major_sf:
                        T.copy(sf_ub[0:num_stored_sf_cols_per_block, 0:token_group], out_sf[stored_sf_k_base, sf_m_base])
                    else:
                        T.copy(sf_ub[0:token_group, 0:num_stored_sf_cols_per_block], out_sf[sf_m_base, stored_sf_k_base])

    return per_block_cast_kernel_vmi
