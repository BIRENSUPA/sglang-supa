from __future__ import annotations

from sglang_supa.srt.utils.common import is_supa_mxfp8_supported

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

# Current MoE atomic buffer exposed for the downstream cutlass_moe.py path.
moe_atomic_buffer = None


def process_mxfp8_linear_weight_scale_around(original_fn, self, layer, *args, **kwargs):
    backend = self.mxfp8_dense_backend
    is_supa = getattr(backend, "is_supa", None)
    if not callable(is_supa) or not is_supa():
        return original_fn(self, layer, *args, **kwargs)

    from sglang_supa.srt.layers.quantization.fp8_utils import (
        to_sublas_interleaved_ue8m0,
    )

    from sglang.srt.layers.utils import copy_or_rebind_param

    # mxfp8_scaled_mm needs both operands' scales sublasLt-interleaved; the
    # weight side is converted once here rather than on every forward.
    scale_u8 = layer.weight_scale_inv.data
    n = scale_u8.shape[0]
    if n % 16 != 0:
        raise RuntimeError(
            f"SUPA MXFP8 dense GEMM cannot serve weight shape ({n}, "
            f"{layer.weight.shape[1]}): sublasLt UE8M0 scales tile 16 rows at "
            "a time, so N must be a multiple of 16."
        )
    copy_or_rebind_param(
        layer,
        "weight_scale_inv_supa",
        to_sublas_interleaved_ue8m0(scale_u8),
    )


def fp8_linear_apply_around(original_fn, self, layer, x, bias=None):
    """Use the upstream apply path except for SUPA MXFP8 execution."""
    if self.use_marlin or not self.use_mxfp8:
        return original_fn(self, layer, x, bias)

    backend = self.mxfp8_dense_backend
    is_supa = getattr(backend, "is_supa", None)
    if not callable(is_supa) or not is_supa():
        return original_fn(self, layer, x, bias)

    weight_scale = layer.weight_scale_inv_supa

    if isinstance(x, tuple):
        return self.w8a8_mxfp8_linear(
            input=x[0],
            weight=layer.weight,
            weight_scale=weight_scale,
            input_scale=x[1],
            bias=bias,
        )
    return self.w8a8_mxfp8_linear(
        input=x,
        weight=layer.weight,
        weight_scale=weight_scale,
        input_scale=None,
        bias=bias,
    )


def process_mxfp8_moe_weights_around(original_fn, self, layer, *args, **kwargs):
    """Use the SUPA MXFP8 capability check during MoE weight processing."""
    import sglang.srt.layers.quantization.fp8 as upstream

    original_is_sm100_supported = upstream.is_sm100_supported
    upstream.is_sm100_supported = is_supa_mxfp8_supported
    try:
        return original_fn(self, layer, *args, **kwargs)
    finally:
        upstream.is_sm100_supported = original_is_sm100_supported


def ensure_cutlass_buffers_initialized_around(
    original_fn, self, layer, *args, **kwargs
):
    """Initialize upstream buffers and prepare the buffer used by cutlass_moe.py."""
    result = original_fn(self, layer, *args, **kwargs)

    import torch

    from sglang.srt.environ import envs

    num_experts = layer.w13_weight.shape[0]
    self.moe_atomic_buffer = (
        torch.zeros(
            num_experts + 2,
            device=layer.w13_weight.device,
            dtype=torch.int32,
        )
        if envs.SGLANG_OPT_FUSED_MOE_PREPARE.get()
        else None
    )
    return result


def fp8_moe_apply_around(original_fn, self, layer, dispatch_output, *args, **kwargs):
    """Expose this instance's buffer to the downstream cutlass_moe.py path."""
    global moe_atomic_buffer
    moe_atomic_buffer = getattr(self, "moe_atomic_buffer", None)
    return original_fn(self, layer, dispatch_output, *args, **kwargs)


def get_min_capability(_self, *args, **kwargs) -> int:
    """Allow FP8 configuration on SUPA regardless of reported capability."""
    return 0


def register_fp8_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8Config.get_min_capability",
        get_min_capability,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod._process_mxfp8_linear_weight_scale",
        process_mxfp8_linear_weight_scale_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.apply",
        fp8_linear_apply_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8MoEMethod._process_mxfp8_moe_weights",
        process_mxfp8_moe_weights_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8MoEMethod._ensure_cutlass_buffers_initialized",
        ensure_cutlass_buffers_initialized_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8MoEMethod.apply",
        fp8_moe_apply_around,
        HookType.AROUND,
    )
