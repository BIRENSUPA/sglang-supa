"""SUPA source resource selection for the TRT-LLM LoRA MoE launcher."""

from __future__ import annotations

import importlib
from pathlib import Path


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


class _OverlayCsrcDir:
    def __init__(self, upstream_dir: Path, plugin_dir: Path):
        self._upstream_dir = upstream_dir
        self._plugin_dir = plugin_dir

    def __truediv__(self, name: str) -> Path:
        if name == "trtllm_fused_moe_kernel_launcher.cu":
            return self._plugin_dir / name
        return self._upstream_dir / name


class _OverlayDataDir:
    def __init__(self, upstream_dir: Path, plugin_dir: Path):
        self._upstream_dir = upstream_dir
        self._plugin_dir = plugin_dir

    def __truediv__(self, name: str):
        if name == "csrc":
            return _OverlayCsrcDir(
                self._upstream_dir / name,
                self._plugin_dir / name,
            )
        return self._upstream_dir / name


def register_trtllm_lora_resource_patch() -> None:
    """Use the plugin-owned launcher resource for future FlashInfer builds.

    Lifecycle: plugin-register-immediate. The upstream generator resolves
    ``_data_dir`` when its function is called, so rebinding it also covers the
    generator module imported before plugin discovery. Manifest:
    kernels.ops-supa-trtllm-lora-resource.
    """
    upstream = importlib.import_module("sglang.kernels.ops.moe.trtllm_lora_temp.jit")
    if getattr(upstream, "_supa_resource_patched", False):
        return
    upstream_data_dir = upstream._data_dir()
    upstream._data_dir = lambda: _OverlayDataDir(
        upstream_data_dir,
        _data_dir(),
    )
    upstream._supa_resource_patched = True
