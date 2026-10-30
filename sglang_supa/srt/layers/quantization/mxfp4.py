"""Compatibility fake for the optional FlashInfer TRT-LLM MXFP4 enums."""

from __future__ import annotations

import logging
import sys
import types

logger = logging.getLogger(__name__)

_MODULE_NAME = "flashinfer.tllm_enums"


class _UnsupportedEnumMember:
    def __init__(self, enum_name: str, member_name: str):
        self._enum_name = enum_name
        self._member_name = member_name

    @property
    def value(self):
        message = (
            f"{_MODULE_NAME}.{self._enum_name}.{self._member_name} was used, "
            "but the installed FlashInfer build does not provide TRT-LLM "
            "MXFP4 support on SUPA."
        )
        logger.error(message)
        raise RuntimeError(message)

    def __repr__(self) -> str:
        return f"{self._enum_name}.{self._member_name}"


class _ActivationType:
    Situ = _UnsupportedEnumMember("ActivationType", "Situ")


class _RoutingMethodType:
    TopK = _UnsupportedEnumMember("RoutingMethodType", "TopK")
    DeepSeekV3 = _UnsupportedEnumMember("RoutingMethodType", "DeepSeekV3")


def install_flashinfer_tllm_enums_fake() -> None:
    """Install missing enum names without claiming the kernels are supported."""
    import flashinfer

    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return

    module = types.ModuleType(_MODULE_NAME)
    module.ActivationType = _ActivationType
    module.RoutingMethodType = _RoutingMethodType
    module.__all__ = ["ActivationType", "RoutingMethodType"]
    sys.modules[_MODULE_NAME] = module
    flashinfer.tllm_enums = module
