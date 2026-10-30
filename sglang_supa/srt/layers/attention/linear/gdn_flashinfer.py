from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def decode_around(original_fn, self, *args, **kwargs):
    """Run FlashInfer GDN decode through the SUPA state-pool path."""
    use_state_pool_original = self.use_state_pool
    self.use_state_pool = True
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        self.use_state_pool = use_state_pool_original


def register_gdn_flashinfer_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer.FlashInferGDNKernel.decode",
        decode_around,
        HookType.AROUND,
    )
