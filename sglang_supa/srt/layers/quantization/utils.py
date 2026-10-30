from __future__ import annotations

from sglang.srt.layers.quantization.utils import _module_path_match
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def is_layer_skipped_around(original_fn, prefix, ignored_layers, fused_mapping=None):
    from types import MappingProxyType

    from sglang.srt.layers.quantization.utils import _FALLBACK_FUSED_SHARDS

    if fused_mapping is None:
        fused_mapping = MappingProxyType({})

    proj_name = prefix.split(".")[-1]
    effective_fused = (
        fused_mapping if proj_name in fused_mapping else _FALLBACK_FUSED_SHARDS
    )

    if proj_name in effective_fused:
        if any(_module_path_match(ignored, prefix) for ignored in ignored_layers):
            return True

    return original_fn(prefix, ignored_layers, fused_mapping)


def register_quant_utils_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.quantization.utils.is_layer_skipped",
        is_layer_skipped_around,
        HookType.AROUND,
    )
