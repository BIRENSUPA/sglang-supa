from __future__ import annotations

import os


def supa_platform_plugin() -> str | None:
    """Return the SUPA platform only when diffusion selected SUPA or none.

    The upstream diffusion resolver gives
    ``SGLANG_DIFFUSION_PLATFORM_OVERRIDE`` precedence over automatic platform
    detection. Keep the same contract in the out-of-tree SUPA plugin: a
    non-SUPA override must be handled by the upstream resolver instead of
    being replaced by SUPA hardware detection.
    """
    forced_platform = os.environ.get("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "")
    if forced_platform.strip().lower() not in ("", "supa"):
        return None

    try:
        from sglang_supa.entrypoints import detect_supa_hardware

        if detect_supa_hardware():
            return "sglang_supa.multimodal_gen.runtime.platforms.supa.SupaPlatform"
    except Exception:
        return None
    return None


def register_multimodal_platform_init_hooks() -> None:
    import sglang.multimodal_gen.runtime.platforms as platforms_module

    if getattr(platforms_module, "_supa_platform_init_patched", False):
        return

    original_fn = platforms_module.resolve_current_platform_cls_qualname

    def resolve_current_platform_cls_qualname():
        platform_cls_qualname = supa_platform_plugin()
        if platform_cls_qualname is not None:
            return platform_cls_qualname
        return original_fn()

    platforms_module.resolve_current_platform_cls_qualname = (
        resolve_current_platform_cls_qualname
    )
    if (
        platforms_module._current_platform is not None
        and supa_platform_plugin() is not None
    ):
        platforms_module._current_platform = None
    platforms_module._supa_platform_init_patched = True
