from __future__ import annotations

from typing import List

from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce as custom_all_reduce_module
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def create_shared_buffer_around(
    original_fn, size_in_bytes: int, group: ProcessGroup | None = None
) -> List[int]:
    """Enable the upstream MUSA initialization path for SUPA buffers.

    The upstream function reads module-level ``_is_musa`` to decide whether to
    zero-initialize the allocation. This temporary mutation is not
    thread-safe, but the state is restored even when the original function
    raises.
    """
    is_musa_original = custom_all_reduce_module._is_musa
    custom_all_reduce_module._is_musa = True
    try:
        return original_fn(size_in_bytes, group=group)
    finally:
        custom_all_reduce_module._is_musa = is_musa_original


def register_custom_all_reduce_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.custom_all_reduce.CustomAllreduce.create_shared_buffer",
        create_shared_buffer_around,
        HookType.AROUND,
    )
