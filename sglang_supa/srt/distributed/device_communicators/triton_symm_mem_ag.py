from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def multimem_all_gatherer_init_around(original_fn, self, *args, **kwargs):
    """Disable the upstream multimem all-gather state during initialization."""
    kwargs = dict(kwargs)
    kwargs["enabled"] = False
    return original_fn(self, *args, **kwargs)


def register_triton_symm_mem_ag_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.triton_symm_mem_ag.MultimemAllGatherer.__init__",
        multimem_all_gatherer_init_around,
        HookType.AROUND,
    )
