from typing import Optional, Tuple

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
    num_real_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_8bit,
    )
    from sglang.srt.layers import deep_gemm_wrapper

    use_ue8m0 = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    return sglang_per_token_group_quant_8bit(
        x=gateup_output,
        dst_dtype=torch.float8_e4m3fn,
        group_size=group_size,
        masked_m=masked_m,
        column_major_scales=use_ue8m0,
        scale_tma_aligned=use_ue8m0,
        scale_ue8m0=use_ue8m0,
        fuse_silu_and_mul=True,
    )


def register_deep_gemm_moe_runner_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.deep_gemm._varlen_deep_gemm_silu_mul_quant",
        _varlen_deep_gemm_silu_mul_quant,
        HookType.REPLACE,
    )
