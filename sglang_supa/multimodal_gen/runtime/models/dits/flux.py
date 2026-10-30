from __future__ import annotations

from fastcore.basics import patch_to


def register_multimodal_flux_hooks() -> None:
    from sglang.multimodal_gen.runtime.models.dits.flux import (
        FluxPosEmbed,
        NDRotaryEmbedding,
    )

    if getattr(FluxPosEmbed, "_supa_pos_embed_patched", False):
        return

    original_init = FluxPosEmbed.__init__

    @patch_to(FluxPosEmbed, nm="__init__")
    def flux_pos_embed_init(self, theta, axes_dim):
        from sglang.multimodal_gen.runtime.platforms import current_platform
        import torch

        original_init(self, theta, axes_dim)
        if not current_platform.is_supa():
            return
        self.rope = NDRotaryEmbedding(
            rope_dim_list=axes_dim,
            rope_theta=theta,
            use_real=False,
            repeat_interleave_real=False,
            dtype=torch.float32,
        )

    FluxPosEmbed._supa_pos_embed_patched = True
