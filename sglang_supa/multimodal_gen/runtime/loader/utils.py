from __future__ import annotations

from fastcore.basics import patch_to


class skip_init_modules:
    def __enter__(self):
        import torch.nn as nn

        self._orig_reset = {}
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None

    def __exit__(self, exc_type, exc_value, traceback):
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig


def register_multimodal_loader_utils_hooks() -> None:
    import torch.nn as nn
    import sglang.multimodal_gen.runtime.loader.utils as loader_utils_module

    if getattr(loader_utils_module, "_supa_loader_utils_patched", False):
        return

    original_get_memory_usage = loader_utils_module.get_memory_usage_of_component

    loader_utils_module.skip_init_modules = skip_init_modules

    def get_memory_usage_of_component(module):
        if not isinstance(module, nn.Module):
            return 0.0
        return original_get_memory_usage(module)

    loader_utils_module.get_memory_usage_of_component = get_memory_usage_of_component
    loader_utils_module._supa_loader_utils_patched = True
