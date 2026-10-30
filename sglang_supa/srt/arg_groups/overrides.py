import logging
from typing import Any

from sglang_supa.srt.utils.common import is_supa_mxfp8_supported

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def moe_runner_backend_quant_constraints_around(original_fn, view: Any):
    """Select the CUTLASS MoE runner for SUPA MXFP8 quantization."""
    moe_runner_backend = view.moe_runner_backend
    if view.quantization == "mxfp8" and is_supa_mxfp8_supported():
        from sglang.srt.server_args import MXFP8_MOE_RUNNER_BACKEND_CHOICES

        allowed = list(MXFP8_MOE_RUNNER_BACKEND_CHOICES)
        mxfp8_default = "cutlass"
        if moe_runner_backend == "auto":
            moe_runner_backend = mxfp8_default
        elif moe_runner_backend not in allowed:
            logger.warning(
                "mxfp8 quantization supports only %s backends. " "Overriding %r.",
                ", ".join(allowed),
                moe_runner_backend,
            )
            moe_runner_backend = mxfp8_default

        if moe_runner_backend != view.moe_runner_backend:
            return {"moe_runner_backend": moe_runner_backend}
        return {}
    return original_fn(view)


def register_overrides_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.arg_groups.overrides._moe_runner_backend_quant_constraints",
        moe_runner_backend_quant_constraints_around,
        HookType.AROUND,
    )
