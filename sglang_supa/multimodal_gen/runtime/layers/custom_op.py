"""SUPA dispatch hook for upstream multimodal custom operators."""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_CUSTOM_OP_DISPATCH_TARGET = (
    "sglang.multimodal_gen.runtime.layers.custom_op.CustomOp.dispatch_forward"
)
_hook_registered = False


def _dispatch_forward_supa(original_fn, self):
    """Select ``forward_cuda`` on SUPA and otherwise keep upstream dispatch."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.is_supa():
        return self.forward_cuda
    return original_fn(self)


def register_multimodal_custom_op_hooks() -> None:
    """Register the SUPA dispatch override through upstream HookRegistry."""
    global _hook_registered
    if _hook_registered:
        return

    HookRegistry.register(
        _CUSTOM_OP_DISPATCH_TARGET,
        _dispatch_forward_supa,
        HookType.AROUND,
    )
    _hook_registered = True
