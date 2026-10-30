from __future__ import annotations

import enum
import sys
from functools import lru_cache

import torch
from fastcore.basics import patch_to


def _extend_enum(enum_cls, member_name: str):
    if member_name in enum_cls.__members__:
        return enum_cls
    extended_enum = enum.Enum(
        enum_cls.__name__,
        [*enum_cls.__members__.keys(), member_name],
        module=enum_cls.__module__,
        qualname=enum_cls.__qualname__,
    )
    for name, value in enum_cls.__dict__.items():
        if name.startswith("_") or name in enum_cls.__members__:
            continue
        setattr(extended_enum, name, value)
    return extended_enum


def patch_platform_enums() -> None:
    import sglang.multimodal_gen.runtime.platforms as platforms_module
    import sglang.multimodal_gen.runtime.platforms.interface as interface_module

    attention_backend_enum = _extend_enum(interface_module.AttentionBackendEnum, "SUPA")
    platform_enum = _extend_enum(interface_module.PlatformEnum, "SUPA")

    interface_module.AttentionBackendEnum = attention_backend_enum
    interface_module.PlatformEnum = platform_enum
    platforms_module.AttentionBackendEnum = attention_backend_enum
    platforms_module.PlatformEnum = platform_enum

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("sglang.multimodal_gen"):
            continue
        if getattr(module, "AttentionBackendEnum", None) is not None:
            if getattr(module.AttentionBackendEnum, "__name__", None) == "AttentionBackendEnum":
                module.AttentionBackendEnum = attention_backend_enum
        if getattr(module, "PlatformEnum", None) is not None:
            if getattr(module.PlatformEnum, "__name__", None) == "PlatformEnum":
                module.PlatformEnum = platform_enum


def is_supa(self) -> bool:
    return self.is_supa_static()


def is_supa_static(cls) -> bool:
    import sglang.multimodal_gen.runtime.platforms.interface as interface_module

    return getattr(cls, "_enum", None) == interface_module.PlatformEnum.SUPA

def is_cuda_static(cls) -> bool:
    import sglang.multimodal_gen.runtime.platforms.interface as interface_module

    return getattr(cls, "_enum", None) in (
        interface_module.PlatformEnum.CUDA,
        interface_module.PlatformEnum.SUPA,
    )

def is_cuda_alike(self) -> bool:
    import sglang.multimodal_gen.runtime.platforms.interface as interface_module

    return self._enum in (
        interface_module.PlatformEnum.CUDA,
        interface_module.PlatformEnum.ROCM,
        interface_module.PlatformEnum.MUSA,
        interface_module.PlatformEnum.SUPA,
    )

def is_float64_supported(cls) -> bool:
    import sglang.multimodal_gen.runtime.platforms.interface as interface_module

    if getattr(cls, "_enum", None) == interface_module.PlatformEnum.SUPA:
        return False
    return True


def register_multimodal_platform_interface_hooks() -> None:
    patch_platform_enums()

    from sglang.multimodal_gen.runtime.platforms.interface import Platform

    if getattr(Platform, "_supa_interface_patched", False):
        return

    patch_to(Platform)(is_supa)
    patch_to(Platform, cls_method=True)(is_supa_static)
    patch_to(Platform, cls_method=True)(is_cuda_static)
    patch_to(Platform)(is_cuda_alike)
    patch_to(Platform, cls_method=True)(is_float64_supported)
    Platform._supa_interface_patched = True
