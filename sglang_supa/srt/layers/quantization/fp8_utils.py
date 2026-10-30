from enum import Enum
from typing import List, Optional

import torch
from sgl_kernel import (
    mxfp8_scaled_mm,
    sgl_per_token_group_quant_8bit_v2_interleaved,
    supa_fp8_blockwise_scaled_mm,
)
from sglang_supa.srt.layers.quantization.mxfp4 import (
    install_flashinfer_tllm_enums_fake,
)
from sglang_supa.srt.utils.common import is_supa_mxfp8_supported

install_flashinfer_tllm_enums_fake()

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    fp8_min,
    sglang_per_token_group_quant_fp8_row_padded,
)
from sglang.srt.layers.quantization.fp8_utils import (
    triton_w8a8_block_fp8_linear,
)
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils import ceil_align
from sglang.srt.utils.patch_torch import register_fake_if_exists


@register_fake_if_exists("sgl_kernel::fp8_blockwise_scaled_mm")
def _fp8_blockwise_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):
    # mat_a: [M, K], mat_b: [K, N] or [N, K] depending on callsite layout; output is [M, N].
    M = mat_a.shape[-2]
    N = mat_b.shape[-1]
    return mat_a.new_empty((M, N), dtype=out_dtype)


# Distinct op from fp8_blockwise_scaled_mm above; this is the one
# supa_w8a8_block_fp8_linear_with_fallback calls.
@register_fake_if_exists("sgl_kernel::supa_fp8_blockwise_scaled_mm")
def _supa_fp8_blockwise_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):
    # mat_a: [M, K], mat_b: [K, N] K-major; output is [M, N].
    M = mat_a.shape[-2]
    N = mat_b.shape[-1]
    return mat_a.new_empty((M, N), dtype=out_dtype)


@register_fake_if_exists("sgl_kernel::mxfp8_scaled_mm")
def _mxfp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype):
    # mat_b is [N, K] row-major here, not [K, N] like the blockwise ops above.
    M = mat_a.shape[-2]
    N = mat_b.shape[-2]
    return mat_a.new_empty((M, N), dtype=out_dtype)


@register_fake_if_exists("sgl_kernel::sgl_per_token_group_quant_8bit_v2_interleaved")
def _sgl_per_token_group_quant_8bit_v2_interleaved_abstract(
    input,
    output_q,
    output_s,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    scale_ue8m0,
):
    return


class SupaFp8GemmRunnerBackend(Enum):
    """FP8 GEMM backend selection for the SUPA runtime."""

    SUPA = "supa"

    def is_auto(self) -> bool:
        return False

    def is_flashinfer_trtllm(self) -> bool:
        return False

    def is_flashinfer_cutlass(self) -> bool:
        return False

    def is_flashinfer_cutedsl(self) -> bool:
        return False

    def is_flashinfer_deepgemm(self) -> bool:
        return False

    def is_cutlass(self) -> bool:
        return False

    def is_deep_gemm(self) -> bool:
        return False

    def is_triton(self) -> bool:
        return False

    def is_aiter(self) -> bool:
        return False

    def is_supa(self) -> bool:
        return self == SupaFp8GemmRunnerBackend.SUPA


class SupaMxfp8DenseGemmBackend(Enum):
    """MXFP8 dense GEMM backend selection for the SUPA runtime."""

    SUPA = "supa"

    def is_flashinfer_cutlass(self) -> bool:
        return False

    def is_flashinfer_cutedsl(self) -> bool:
        return False

    def is_flashinfer_trtllm(self) -> bool:
        return False

    def is_flashinfer(self) -> bool:
        return False

    def is_deep_gemm(self) -> bool:
        return False

    def is_gfx95_dot_scaled(self) -> bool:
        return False

    def is_supa(self) -> bool:
        return self == SupaMxfp8DenseGemmBackend.SUPA

    def is_unsupported(self) -> bool:
        return False


def mxfp8_group_quantize_supa(x):
    """Quantize a contiguous 2D tensor with the SUPA MXFP8 operator."""
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert x.is_contiguous(), "MXFP8 quantization requires a contiguous 2D tensor."
    _, k = x.shape
    assert k % 32 == 0, f"{k=} must be divisible by 32"

    from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8

    quantized = to_mxfp8(x)
    return quantized.data.contiguous(), quantized.scale.contiguous()


def initialize_fp8_gemm_config_around(original_fn):
    """Select the SUPA-only backend when the configured backend is ``supa``."""
    import sglang.srt.layers.quantization.fp8_utils as upstream
    from sglang.srt.runtime_context import get_exec

    backend = get_exec().kernel.fp8_gemm_runner_backend
    if backend == "supa":
        upstream.FP8_GEMM_RUNNER_BACKEND = SupaFp8GemmRunnerBackend.SUPA
        return None
    return original_fn()


def resolve_mxfp8_dense_gemm_backend_around(original_fn, *args, **kwargs):
    """Return the SUPA-only MXFP8 backend for the SUPA GEMM configuration."""
    import sglang.srt.layers.quantization.fp8_utils as upstream

    backend = upstream.get_fp8_gemm_runner_backend()
    is_supa = getattr(backend, "is_supa", None)
    if callable(is_supa) and is_supa():
        if not is_supa_mxfp8_supported():
            raise RuntimeError(
                "MXFP8 dense GEMM requested via --fp8-gemm-backend=supa, but that "
                "kernel requires a SUPA device with MX-scaled FP8 matmul."
            )
        return SupaMxfp8DenseGemmBackend.SUPA
    return original_fn(*args, **kwargs)


def dispatch_w8a8_mxfp8_linear_around(original_fn, *args, **kwargs):
    """Dispatch SUPA MXFP8 to the downstream implementation."""
    import sglang.srt.layers.quantization.fp8_utils as upstream

    backend = upstream.resolve_mxfp8_dense_gemm_backend()
    is_supa = getattr(backend, "is_supa", None)
    if callable(is_supa) and is_supa():
        return supa_w8a8_mxfp8_linear
    return original_fn(*args, **kwargs)


def dispatch_explicit_backend_around(original_fn, backend, *args, **kwargs):
    """Route explicit CUTLASS and SUPA FP8 backends to their implementations."""
    import sglang.srt.layers.quantization.fp8_utils as upstream

    if backend.is_cutlass():
        return upstream.cutlass_w8a8_block_fp8_linear_with_fallback

    is_supa = getattr(backend, "is_supa", None)
    if callable(is_supa) and is_supa():
        return supa_w8a8_block_fp8_linear_with_fallback

    return original_fn(backend, *args, **kwargs)


def register_fp8_utils_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.initialize_fp8_gemm_config",
        initialize_fp8_gemm_config_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.resolve_mxfp8_dense_gemm_backend",
        resolve_mxfp8_dense_gemm_backend_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.dispatch_w8a8_mxfp8_linear",
        dispatch_w8a8_mxfp8_linear_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils._dispatch_explicit_backend",
        dispatch_explicit_backend_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.mxfp8_group_quantize",
        mxfp8_group_quantize_supa,
        HookType.REPLACE,
    )


def supa_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    shape_supported = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    if not shape_supported:
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    # Quantize into row-padded buffers so the sgl-kernel wrapper's per-call
    # pad_tensor() on mat_a / scales_a short-circuits (saves 2x fill + 2x cat
    # kernels per GEMM). weight_scale.T is left as a K-major view because the
    # kernel requires scales_b.stride(0) == 1 and materializes it internally.
    q_input, x_scale = sglang_per_token_group_quant_fp8_row_padded(
        input_2d, block_size[1]
    )
    output = supa_fp8_blockwise_scaled_mm(
        q_input, weight.T, x_scale, weight_scale.T, out_dtype=input_2d.dtype
    )
    if output.shape[0] != input_2d.shape[0]:
        # GEMM ran on the row-padded buffer; drop the padding rows.
        output = output[: input_2d.shape[0]]
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


# sublasLt VEC32_UE8M0 tiles the scale bytes 16 rows at a time: byte (row m,
# group g) sits at (m // 16) * 16 * sfK + g * 16 + m % 16. Both operands of
# mxfp8_scaled_mm must already be in this layout; plain [rows, K // 32] silently
# produces ~100% relative error.
_SUBLAS_UE8M0_ROW_TILE = 16


def to_sublas_interleaved_ue8m0(scale_u8: torch.Tensor) -> torch.Tensor:
    """Interleave a plain 2D UE8M0 scale ``[rows, K // 32]`` for sublasLt."""
    rows, sf_k = scale_u8.shape
    assert (
        rows % _SUBLAS_UE8M0_ROW_TILE == 0
    ), f"sublasLt UE8M0 needs rows % {_SUBLAS_UE8M0_ROW_TILE} == 0, got {rows}"
    return (
        scale_u8.contiguous()
        .view(rows // _SUBLAS_UE8M0_ROW_TILE, _SUBLAS_UE8M0_ROW_TILE, sf_k)
        .transpose(1, 2)
        .contiguous()
        .view(rows, sf_k)
    )


def supa_w8a8_mxfp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """MXFP8 dense linear via the SUPA ``mxfp8_scaled_mm`` sublasLt GEMM.

    ``weight_scale`` must be the sublasLt-interleaved weight scale prepared at load
    time by ``_process_mxfp8_linear_weight_scale``. Activations are quantized here
    with the fused interleaved-emit kernel, which also pads rows to a multiple of 16
    because the GEMM requires ``scales_a.size(0) == mat_a.size(0)``.
    """
    assert weight.dtype == torch.float8_e4m3fn, "MXFP8 weight must be FP8 E4M3."
    assert weight_scale.dtype == torch.uint8, "MXFP8 weight_scale must be UE8M0 uint8."
    # Raised rather than asserted: `python -O` strips asserts, and dropping this one
    # would not fail, it would feed a plain scale to a GEMM that reads it as
    # sublasLt-interleaved and return a ~100% wrong result.
    if input_scale is not None:
        raise RuntimeError(
            "SUPA MXFP8 linear quantizes activations itself; it cannot consume the "
            "plain pre-quantized scales a fused upstream op emits (the GEMM needs "
            "them sublasLt-interleaved)."
        )

    input_2d = input.view(-1, input.shape[-1]).contiguous()
    output_shape = [*input.shape[:-1], weight.shape[0]]
    if output_dtype is None:
        output_dtype = input_2d.dtype

    m, k = input_2d.shape
    assert k == weight.shape[1], f"{k=} does not match weight K={weight.shape[1]}"
    assert k % 32 == 0, f"MXFP8 requires {k=} divisible by 32"

    m_pad = max(ceil_align(m, _SUBLAS_UE8M0_ROW_TILE), _SUBLAS_UE8M0_ROW_TILE)
    q_input = torch.empty((m_pad, k), device=input_2d.device, dtype=fp8_dtype)
    # The interleaved emit writes only the m real rows, scattered within each 16-row
    # tile, so the pad bytes must be zeroed up front rather than sliced off after.
    x_scale = torch.zeros((m_pad, k // 32), device=input_2d.device, dtype=torch.uint8)
    if m_pad != m:
        q_input[m:].zero_()
    sgl_per_token_group_quant_8bit_v2_interleaved(
        input_2d,
        q_input[:m],
        x_scale,
        32,
        1e-10,
        fp8_min,
        fp8_max,
        scale_ue8m0=True,
    )

    output = mxfp8_scaled_mm(q_input, weight, x_scale, weight_scale, output_dtype)
    if m_pad != m:
        output = output[:m]
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)
