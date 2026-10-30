from __future__ import annotations

from typing import Optional

import torch
from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils.custom_op import register_custom_op


# TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
# Wrap bmm_fp8 as a custom op so torch.compile does not trace into
# torch.cuda.current_blas_handle() (which returns a non-Tensor).
@register_custom_op(mutates_args=["out"])
def _bmm_fp8_op(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    _bmm_fp8_op(A, B, out, A_scale, B_scale)
    return out


def register_forward_mla_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.bmm_fp8",
        bmm_fp8,
        HookType.REPLACE,
    )
