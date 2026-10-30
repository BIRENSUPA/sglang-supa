from __future__ import annotations

from sglang_supa.srt.layers.quantization import fp8
from sglang_supa.srt.utils.common import is_supa_mxfp8_supported

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _is_sm90_supported_false() -> bool:
    return False


def prepare_moe_input_around(original_fn, *args, **kwargs):
    """Pass the current FP8 MoE atomic buffer to the SUPA kernel wrapper."""
    # Read the module attribute at the upstream API call rather than importing
    # its value once. Each apply call binds its instance buffer before entering
    # this wrapper, so sequential calls from different MoE objects use the
    # current object's buffer instead of a stale imported reference.
    kwargs = dict(kwargs)
    kwargs["atomic_buffer"] = fp8.moe_atomic_buffer
    return original_fn(*args, **kwargs)


def cutlass_fused_experts_fp8_around(original_fn, *args, **kwargs):
    """Use SUPA capability checks during the upstream fused-MoE call."""
    import sglang.srt.layers.moe.cutlass_moe as upstream

    original_is_sm90_supported = upstream.is_sm90_supported
    original_is_sm100_supported = upstream.is_sm100_supported
    # This predicate selects the upstream SM90 kernel branch. SUPA does not
    # support that implementation, so force the branch condition to false.
    upstream.is_sm90_supported = _is_sm90_supported_false

    # This predicate selects the upstream SM100/MXFP8 kernel branch. It is a
    # branch-selection hook here, not literal hardware identification; use the
    # SUPA capability check so the SUPA MXFP8 path can be selected correctly.
    upstream.is_sm100_supported = is_supa_mxfp8_supported
    try:
        return original_fn(*args, **kwargs)
    finally:
        upstream.is_sm90_supported = original_is_sm90_supported
        upstream.is_sm100_supported = original_is_sm100_supported


def register_cutlass_moe_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.moe.cutlass_moe.prepare_moe_input",
        prepare_moe_input_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.cutlass_moe.cutlass_fused_experts_fp8",
        cutlass_fused_experts_fp8_around,
        HookType.AROUND,
    )
