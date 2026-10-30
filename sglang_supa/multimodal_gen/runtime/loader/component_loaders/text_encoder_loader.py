from __future__ import annotations

from fastcore.basics import patch_to


def register_multimodal_text_encoder_loader_hooks() -> None:
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        TextEncoderLoader,
    )

    if getattr(TextEncoderLoader, "_supa_text_encoder_loader_patched", False):
        return

    original_get_all_weights = TextEncoderLoader._get_all_weights

    @patch_to(TextEncoderLoader, nm="_get_all_weights")
    def text_encoder_loader_get_all_weights(self, model, model_path, to_cpu: bool):
        return original_get_all_weights(self, model, model_path, to_cpu=True)

    TextEncoderLoader._supa_text_encoder_loader_patched = True
