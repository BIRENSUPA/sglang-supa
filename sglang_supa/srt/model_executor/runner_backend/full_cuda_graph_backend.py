from __future__ import annotations

from typing import Any, Callable

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


class _SupaGraphDeviceModule:
    """Adapt SGLang's CUDA keyword to torch_supa's positional graph API."""

    def __init__(self, device_module: Any) -> None:
        self._device_module = device_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._device_module, name)

    def graph(self, cuda_graph, **kwargs):
        return self._device_module.graph(cuda_graph, **kwargs)


def capture_one_around(original_fn: Callable, self, *args, **kwargs):
    """Run upstream capture with the torch_supa graph calling convention.

    Full graph capture is serialized during runner initialization. The temporary
    per-backend proxy is restored even when warmup or capture raises.
    """
    device_module = self._device_module
    self._device_module = _SupaGraphDeviceModule(device_module)
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        self._device_module = device_module


def register_full_cuda_graph_backend_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_one",
        capture_one_around,
        HookType.AROUND,
    )
