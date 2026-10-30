import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils import get_device_module


def is_stream_capturing(stream: torch.Stream) -> bool:
    # On SUPA, cuda-python is unavailable, so use the portable torch
    # API (which maps to the SUPA runtime).
    with get_device_module().stream(stream):
        return get_device_module().is_current_stream_capturing()


def register_is_streaming_capturing_hooks():
    HookRegistry.register(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph._is_stream_capturing",
        is_stream_capturing,
        HookType.REPLACE,
    )
