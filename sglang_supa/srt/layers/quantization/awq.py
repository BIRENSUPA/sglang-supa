from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _get_supported_act_dtypes(self):
    import torch
    return [torch.float16, torch.bfloat16]


def register_awq_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.quantization.awq.AWQConfig.get_supported_act_dtypes",
        _get_supported_act_dtypes,
        HookType.REPLACE,
    )
