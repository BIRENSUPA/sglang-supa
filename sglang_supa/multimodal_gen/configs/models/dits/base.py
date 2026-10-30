from __future__ import annotations

from fastcore.basics import patch_to


def add_supa_backend(self) -> None:
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if hasattr(self, "_supported_attention_backends"):
        self._supported_attention_backends.add(AttentionBackendEnum.SUPA)


def register_multimodal_dit_config_hooks() -> None:
    from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig

    if getattr(DiTArchConfig, "_supa_config_patched", False):
        return

    original_post_init = DiTArchConfig.__post_init__

    @patch_to(DiTArchConfig, nm="__post_init__")
    def dit_arch_config_post_init(self) -> None:
        original_post_init(self)
        add_supa_backend(self)

    DiTArchConfig._supa_config_patched = True
