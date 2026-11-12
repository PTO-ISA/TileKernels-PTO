import tilelang
from tilelang.ascend import language as T

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig
from tile_kernels.utils import align

# BF16 bit-field constants for power-of-two round_sf.
_ABS_MASK_BF16 = 0x7FFF
_BF16_EXP_BIAS = 0x7F00
_EXP_MASK_F32 = 0x7F800000
_F32_EXP_BIAS = 0x7F000000


def _get_block_shape(hidden: int, dtype: T.dtype, num_stages: int, num_per_channels: int) -> tuple[int, int]:
    """Choose the widest valid block_k and largest UB-fitting block_m."""
    stage_budget = get_max_ub_per_vector_core(use_simt=False) // num_stages

    alignment = 256 if hidden % 256 == 0 else 128
    for block_k in range(hidden, alignment - 1, -alignment):
        if hidden % block_k != 0:
            continue
        for block_m in range(32, 0, -1):
            num_sf = align(block_m * align(block_k // num_per_channels, 8), 64)
            ub_usage = block_m * block_k * dtype.bits // 4 + num_sf * 12
            if ub_usage <= stage_budget:
                return block_m, block_k


@tilelang.jit(target='pto')
def get_per_token_cast_and_cast_back_kernel_parallel(
    hidden: int,
    token_stride: int,
    dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    num_per_channels = out_config.sf_block[1]
    is_bf16_in = dtype == T.bfloat16
    assert dtype in (T.bfloat16, T.float32), 'Ascend per-token cast and cast-back only supports bf16 or fp32 input'
    assert (num_per_channels, out_config.use_e4m3_sf) in ((16, True), (32, False)), 'Ascend supports per-16 with E4M3 SF or per-32 without E4M3 SF'
    assert hidden % 128 == 0, 'Ascend per-token cast and cast-back hidden must be a multiple of 128'

    is_fp4_out = out_config.dtype == T.float4_e2m1fn
    round_sf = out_config.round_sf
    quant_max = 6.0 if is_fp4_out else 448.0
    round_carry = 0x003FFFFF if is_fp4_out else 0x001FFFFF
    quant_max_exp = 0x01000000 if is_fp4_out else 0x04000000

    num_stages = 3

    block_m, block_k = _get_block_shape(hidden, dtype, num_stages, num_per_channels)
    # Per-32 rows with a 128-element tail need four padding scales.
    num_groups = align(block_k // num_per_channels, 8)

    num_sf = align(block_m * num_groups, 64)
    use_bf16_sf = is_bf16_in and round_sf
    sf_inv_dtype = T.bfloat16 if use_bf16_sf else T.float32
    use_bf16_dequant = is_bf16_in and out_config.use_e4m3_sf
    sf_dtype = T.bfloat16 if use_bf16_dequant else T.float32

    num_tokens = T.dynamic('num_tokens')

    @T.macro
    def collect_amax(x_ub, amax_ub):
        with T.SimdVF(lanes=128):
            for row in T.serial(block_m):
                for j in T.Parallel(num_groups):
                    maximum = T.alloc_var(T.float32, init=0.0)
                    for k in T.serial(num_per_channels):
                        maximum = T.max(
                            maximum,
                            T.if_then_else(
                                j < block_k // num_per_channels,
                                T.abs(T.float32(x_ub[row, T.min(j, block_k // num_per_channels - 1) * num_per_channels + k])),
                                0.0,
                            ),
                        )
                    amax_ub[row * num_groups + j] = maximum

    @T.macro
    def compute_sf(amax_ub, sf_ub, sf_inv_ub):
        with T.SimdVF(lanes=128):
            for j in T.Parallel(block_m * num_groups):
                maximum = T.max(amax_ub[j], out_config.clamp_min_value)
                if round_sf:
                    rounded_exp = (T.reinterpret(maximum, T.uint32) + T.uint32(round_carry)) & T.uint32(_EXP_MASK_F32)
                    inverse = T.reinterpret(T.uint32(_F32_EXP_BIAS + quant_max_exp) - rounded_exp, T.float32)
                else:
                    scale = maximum / quant_max
                    if out_config.use_e4m3_sf:
                        scale = T.float32(T.cast(T.min(scale, 448.0), T.float8_e4m3fn))
                        inverse = 1.0 / scale
                    else:
                        inverse = quant_max / maximum
                    sf_ub[j] = T.cast(scale, sf_dtype)
                sf_inv_ub[j] = T.cast(inverse, sf_inv_dtype)

    @T.macro
    def cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub):
        cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub)

    @T.macro
    def cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub):
        with T.SimdVF(lanes=128):
            for row, j in T.Parallel(block_m, block_k):
                group = row * num_groups + j // num_per_channels
                inverse = T.float32(sf_inv_ub[group])
                scaled = T.float32(x_ub[row, j]) * inverse
                if use_bf16_sf:
                    scaled = T.float32(T.cast(scaled, T.bfloat16))
                if out_config.use_e4m3_sf and not is_fp4_out:
                    scaled = T.min(T.max(scaled, -448.0), 448.0)
                quantized = T.cast(scaled, out_config.dtype)
                if round_sf:
                    scale = 1.0 / inverse
                else:
                    scale = T.float32(sf_ub[group])
                out_ub[row, j] = T.cast(T.float32(quantized) * scale, dtype)

    @T.prim_func
    def per_token_cast_and_cast_back_kernel(x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype]):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), dtype)
            amax_ub = T.alloc_shared((num_sf,), T.float32)
            sf_ub = T.alloc_shared((num_sf,), sf_dtype)
            sf_inv_ub = T.alloc_shared((num_sf,), sf_inv_dtype)
            out_ub = T.alloc_shared((block_m, block_k), dtype)
            versions = {x_ub: num_stages, amax_ub: num_stages, sf_inv_ub: num_stages, out_ub: num_stages}
            if not round_sf:
                versions[sf_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), hidden // block_k],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                T.copy(x[pid_token * block_m, pid_hidden * block_k], x_ub)
                collect_amax(x_ub, amax_ub)
                compute_sf(amax_ub, sf_ub, sf_inv_ub)
                if num_per_channels == 16:
                    cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub)
                else:
                    cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub)
                T.copy(out_ub, x[pid_token * block_m, pid_hidden * block_k])

    return per_token_cast_and_cast_back_kernel
