from __future__ import annotations

import torch


def init_distributed_environment(
    original_fn,
    world_size: int = 1,
    rank: int = 0,
    distributed_init_method: str = "env://",
    local_rank: int = 0,
    backend: str | None = None,
    device_id: torch.device | None = None,
    timeout: int | None = None,
):
    from sglang.multimodal_gen.runtime.platforms import current_platform
    if current_platform.is_supa():
        device_id = None
    return original_fn(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank,
        backend=backend,
        device_id=device_id,
        timeout=timeout,
    )


def register_multimodal_parallel_state_hooks() -> None:
    import sglang.multimodal_gen.runtime.distributed.parallel_state as parallel_state_module

    if getattr(parallel_state_module, "_supa_parallel_state_patched", False):
        return

    original_fn = parallel_state_module.init_distributed_environment

    def init_distributed_environment_wrapper(*args, **kwargs):
        return init_distributed_environment(original_fn, *args, **kwargs)

    parallel_state_module.init_distributed_environment = init_distributed_environment_wrapper
    parallel_state_module._supa_parallel_state_patched = True
