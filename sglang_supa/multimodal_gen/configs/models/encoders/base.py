from __future__ import annotations

from fastcore.basics import patch_to


def add_supa_backend(self) -> None:
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if hasattr(self, "_supported_attention_backends"):
        self._supported_attention_backends.add(AttentionBackendEnum.SUPA)


def register_multimodal_encoder_config_hooks() -> None:
    from sglang.multimodal_gen.configs.models.encoders.base import EncoderArchConfig

    if getattr(EncoderArchConfig, "_supa_config_patched", False):
        return

    original_init = EncoderArchConfig.__init__

    @patch_to(EncoderArchConfig, nm="__init__")
    def encoder_arch_config_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        add_supa_backend(self)

    EncoderArchConfig._supa_config_patched = True
