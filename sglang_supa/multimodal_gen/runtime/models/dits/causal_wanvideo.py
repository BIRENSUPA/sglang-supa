from __future__ import annotations


def register_multimodal_causal_wanvideo_hooks() -> None:
    import sglang.multimodal_gen.runtime.models.dits.causal_wanvideo as causal_wanvideo_module

    if getattr(causal_wanvideo_module, "_supa_rotary_pos_embed_patched", False):
        return

    original_get_rotary_pos_embed = causal_wanvideo_module.get_rotary_pos_embed

    def get_rotary_pos_embed(*args, **kwargs):
        from sglang.multimodal_gen.runtime.platforms import current_platform
        import torch

        if current_platform.is_supa():
            kwargs["dtype"] = torch.float32
        return original_get_rotary_pos_embed(*args, **kwargs)

    causal_wanvideo_module.get_rotary_pos_embed = get_rotary_pos_embed
    causal_wanvideo_module._supa_rotary_pos_embed_patched = True
