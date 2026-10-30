from __future__ import annotations

from fastcore.basics import patch_to


def register_multimodal_wanvideo_hooks() -> None:
    from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
        NDRotaryEmbedding,
        WanTransformer3DModel,
    )

    if getattr(WanTransformer3DModel, "_supa_rotary_emb_patched", False):
        return

    original_init = WanTransformer3DModel.__init__

    @patch_to(WanTransformer3DModel, nm="__init__")
    def wan_transformer_init(self, *args, **kwargs):
        from sglang.multimodal_gen.runtime.platforms import current_platform
        import torch

        original_init(self, *args, **kwargs)
        if not current_platform.is_supa():
            return
        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=torch.float32,
        )

    WanTransformer3DModel._supa_rotary_emb_patched = True
