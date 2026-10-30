"""SUPA call-path hooks for upstream multimodal normalization kernels."""

from __future__ import annotations

import importlib
import sys
import types

_UPSTREAM_MODULE = (
    "sglang.kernels.ops.diffusion.cutedsl.scale_residual_norm_scale_shift"
)
_SUPA_MODULE = "sglang_supa.kernels.ops.diffusion.supa.scale_residual_norm_scale_shift"
_FUSED_FUNCTIONS = (
    "fused_norm_scale_shift",
    "fused_scale_residual_norm_scale_shift",
)


def register_multimodal_norm_hooks() -> None:
    """Route upstream fused norm calls to the SUPA implementations.

    Multimodal ``CustomOp`` dispatch already selects upstream ``forward_cuda``
    on SUPA. That method imports the fused functions lazily, so replacing the
    imported module at this boundary preserves all upstream control flow while
    avoiding the CUDA/CuTe module's optional ``cutlass`` dependency.
    """

    # upstream forward_cuda() executes
    # ``from sglang.kernels.ops.diffusion.cutedsl.scale_residual_norm_scale_shift
    # import fused_norm_scale_shift``. The real CuTe module imports ``cutlass``
    # and cannot be loaded on SUPA. HookRegistry can wrap forward_cuda(), but
    # cannot intercept this import inside the function. We therefore put a
    # small proxy with the same name in sys.modules. When Python handles the
    # import above, it finds the proxy and gets the SUPA fused function instead.
    # As a result, upstream still calls ``fused_norm_scale_shift(...)``, but
    # that name now points to
    # ``sglang_supa.kernels.ops.diffusion.supa.scale_residual_norm_scale_shift``.
    # The parent package attribute is updated for normal module lookup, and the
    # marker makes this process-wide replacement idempotent.

    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_supa():
        return

    implementation = importlib.import_module(_SUPA_MODULE)
    target = sys.modules.get(_UPSTREAM_MODULE)
    if target is not None and getattr(
        target, "_supa_multimodal_norm_kernel_path_patched", False
    ):
        return
    if target is None:
        target = types.ModuleType(_UPSTREAM_MODULE)
        target.__package__ = _UPSTREAM_MODULE.rsplit(".", 1)[0]
        target.__all__ = list(_FUSED_FUNCTIONS)
        sys.modules[_UPSTREAM_MODULE] = target

    for function_name in _FUSED_FUNCTIONS:
        setattr(target, function_name, getattr(implementation, function_name))

    parent_name, module_name = _UPSTREAM_MODULE.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    setattr(parent, module_name, target)
    target._supa_multimodal_norm_kernel_path_patched = True
