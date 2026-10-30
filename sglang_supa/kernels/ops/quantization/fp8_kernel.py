"""SUPA patch for the upstream FP8 kernel module."""

from __future__ import annotations

import importlib

from sglang.kernels.ops.quantization import sgl_per_token_group_quant_8bit


def _supa_run_per_token_group_quant_8bit_kernel(
    x,
    x_q,
    x_s,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    *,
    scale_ue8m0,
    fuse_silu_and_mul,
    masked_m,
) -> None:
    """Run the SUPA AOT group-quant operator without changing platform flags."""
    sgl_per_token_group_quant_8bit(
        x,
        x_q,
        x_s,
        group_size,
        eps,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        fuse_silu_and_mul,
        masked_m,
        enable_v2=True,
    )


def register_fp8_kernel_patch() -> None:
    """Patch upstream FP8 dispatch before its first quantization call.

    Lifecycle: plugin-register-immediate. The current vendored FP8 module
    lacks the explicit SUPA AOT branch, so only its group-quant dispatch is
    rebound before runtime dispatch. Manifest: kernels.ops-supa-quantization-fp8.
    """
    module = importlib.import_module("sglang.kernels.ops.quantization.fp8_kernel")
    if getattr(module, "_supa_kernel_patched", False):
        return
    module._run_per_token_group_quant_8bit_kernel = (
        _supa_run_per_token_group_quant_8bit_kernel
    )
    module._V3_KERNEL_SUPPORTED_GROUP_SIZES = module._MUSA_KERNEL_SUPPORTED_GROUP_SIZES
    module._supa_kernel_patched = True
