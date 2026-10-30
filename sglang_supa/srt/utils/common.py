from __future__ import annotations

import sys
from functools import lru_cache, wraps


def register_safe_unpickler_patch() -> None:
    """Allow SUPA multiprocessing tensor reductions during safe unpickling."""
    from sglang.srt.utils.common import SafeUnpickler

    SafeUnpickler.ALLOWED_GLOBALS.update(
        {
            ("torch_supa.multiprocessing.reductions", "rebuild_supa_tensor"),
        }
    )


def direct_register_custom_op_around(original_fn, *args, **kwargs):
    """Reuse the upstream NPU branch to register SUPA ops on PrivateUse1."""
    import sglang.srt.utils.common as upstream

    original_is_npu = upstream.is_npu
    upstream.is_npu = lambda: True
    try:
        return original_fn(*args, **kwargs)
    finally:
        upstream.is_npu = original_is_npu


def register_direct_register_custom_op_patch() -> None:
    """Patch custom-op registration before any eager op decorators run.

    ``register_custom_op`` is eager by default, so a HookRegistry entry is too
    late when an upstream module imports ``parallel_state`` before P4.  Patch
    the function and stale ``from ... import direct_register_custom_op``
    bindings immediately instead.
    """
    import sglang.srt.utils.common as upstream

    original = upstream.direct_register_custom_op

    @wraps(original)
    def direct_register_custom_op_supa(*args, **kwargs):
        return direct_register_custom_op_around(original, *args, **kwargs)

    upstream.direct_register_custom_op = direct_register_custom_op_supa

    for module in list(sys.modules.values()):
        if module is None or module is upstream:
            continue
        try:
            module_globals = vars(module)
        except TypeError:
            continue
        for name, value in list(module_globals.items()):
            if value is original:
                try:
                    setattr(module, name, direct_register_custom_op_supa)
                except (AttributeError, TypeError):
                    pass


@lru_cache(maxsize=1)
def is_supa_mxfp8_supported() -> bool:
    import torch

    return torch.cuda.get_device_capability() == (9, 1)
