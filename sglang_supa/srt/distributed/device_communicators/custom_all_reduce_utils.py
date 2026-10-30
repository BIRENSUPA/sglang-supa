from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def gpu_p2p_access_check(src: int, tgt: int) -> bool:
    """Check if GPU src can access GPU tgt."""

    try:
        return torch.cuda.can_device_access_peer(src, tgt)
    except Exception as error:
        logger.warning("SUPA P2P access check failed: %r", error)
        return False


def is_full_nvlink(physical_device_ids, world_size):
    """SUPA has no NVML"""
    try:
        for i in physical_device_ids:
            for j in physical_device_ids:
                if i != j and not torch.cuda.can_device_access_peer(i, j):
                    return False
        return True
    except Exception as error:
        logger.warning("SUPA P2P detection failed: %r", error)
        return False


def register_custom_all_reduce_utils_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_utils.gpu_p2p_access_check",
        gpu_p2p_access_check,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_utils.is_full_nvlink",
        is_full_nvlink,
        HookType.REPLACE,
    )
