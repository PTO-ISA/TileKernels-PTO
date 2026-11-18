"""Constants for the isolated V4.1 FP8 KV-store reproducer."""

from dataclasses import dataclass

FP8_AMAX_MARGIN = 1e-4
FLOAT8E4M3_MAX = 448.0


@dataclass(frozen=True)
class KVCacheFormat:
    quantized_dtype: str = "e4m3"
    quantized_elements: int = 512
    elements_per_sf: int = 32
    bf16_elements: int = 0
    do_cast: bool = True
    sf_format: str = "ue8m0"
    note: str = "V4.1 FP8 main KV"
    token_interleaved: bool = True

    @property
    def num_elements(self) -> int:
        return self.quantized_elements

    @property
    def quantized_bytes(self) -> int:
        return self.quantized_elements

    @property
    def num_scales(self) -> int:
        return self.quantized_elements // self.elements_per_sf

    @property
    def sf_bytes(self) -> int:
        return self.num_scales

    @property
    def sf_storage_elements(self) -> int:
        return self.sf_bytes

    @property
    def bytes_per_token(self) -> int:
        return self.quantized_bytes + self.sf_bytes


V41_FORMAT = KVCacheFormat()
