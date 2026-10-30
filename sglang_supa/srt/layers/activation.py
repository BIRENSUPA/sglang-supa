from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def jit_act_supported(_out: torch.Tensor) -> bool:
    """Force the upstream JIT activation path to be considered supported."""
    return True


def register_activation_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.activation._jit_act_supported",
        jit_act_supported,
        HookType.REPLACE,
    )
