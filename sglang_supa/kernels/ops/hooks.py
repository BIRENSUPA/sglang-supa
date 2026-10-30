"""Monkey-patches for SUPA-specific settings in upstream ``kernels.ops``.

The upstream ops modules remain the source of truth. This module only changes
runtime flags and kernel/function selection required by the SUPA device; JIT
calls still enter through the existing ``sglang.kernels.jit`` mirror.
"""

from __future__ import annotations

import importlib
import sys
from functools import wraps


def register_kernels_ops_patches(*, jit_overlay_enabled: bool) -> None:
    """Install the current ``sglang_supa/kernels/ops`` adaptations.

    Lifecycle: plugin-register-immediate. This groups only the current
    owner-local kernels/ops patches; pre-existing compatibility hooks in
    ``hooks.py`` remain outside this function. The multimodal diffusion
    scale-residual path is owned by its multimodal module proxy and is not
    installed by this kernels/ops group.
    """
    from sglang_supa.kernels.ops.attention.fla.fused_recurrent import (
        register_fused_recurrent_patch,
    )
    from sglang_supa.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        register_fused_qk_rmsnorm_rope_gate_patch,
    )
    from sglang_supa.kernels.ops.attention.triton_gdn_fused_proj import (
        register_triton_gdn_fused_proj_patch,
    )
    from sglang_supa.kernels.ops.layernorm.mhc import register_mhc_patch
    from sglang_supa.kernels.ops.mamba.causal_conv1d_triton import (
        register_causal_conv1d_patch,
    )
    from sglang_supa.kernels.ops.moe.moe_fused_gate import (
        register_moe_fused_gate_patch,
    )
    from sglang_supa.kernels.ops.moe.trtllm_lora_temp.jit import (
        register_trtllm_lora_resource_patch,
    )
    from sglang_supa.kernels.ops.quantization.fp8_kernel import (
        register_fp8_kernel_patch,
    )

    register_fp8_kernel_patch()
    if not jit_overlay_enabled:
        register_kernels_ops_jit_flag_hooks()
    register_fused_recurrent_patch()
    register_fused_qk_rmsnorm_rope_gate_patch()
    register_triton_gdn_fused_proj_patch()
    register_mhc_patch()
    register_causal_conv1d_patch()
    register_moe_fused_gate_patch()
    register_trtllm_lora_resource_patch()
    _register_marlin_stubs()


def _register_marlin_stubs() -> None:
    """Replace marlin JIT kernels with stubs that raise on SUPA.

    Marlin uses cp.async PTX instructions unsupported by the SUPA MIRA
    backend. Until a native SUPA implementation is available, intercept
    the public entry points so that loading a GPTQ/AWQ-marlin quantized
    model produces a clear error instead of a JIT compilation failure.
    """
    _UNSUPPORTED = (
        ("sglang.kernels.ops.quantization.gptq_marlin", "gptq_marlin_gemm"),
        ("sglang.kernels.ops.quantization.gptq_marlin_repack", "gptq_marlin_repack"),
        ("sglang.kernels.ops.quantization.awq_marlin_repack", "awq_marlin_repack"),
        ("sglang.kernels.ops.quantization.awq_marlin_repack", "awq_marlin_moe_repack"),
    )

    for module_name, func_name in _UNSUPPORTED:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        original = getattr(mod, func_name, None)
        if original is None:
            continue

        @wraps(original)
        def _stub(*args, _name=f"{module_name}.{func_name}", **kwargs):
            raise NotImplementedError(
                f"{_name} is not supported on SUPA. "
                f"Marlin kernels require cp.async PTX instructions that the "
                f"SUPA MIRA backend does not support. Use a non-marlin "
                f"quantization method (e.g. --quantization fp8)."
            )

        setattr(mod, func_name, _stub)


_MIRA_FLAG_MODULES = (
    "sglang.kernels.ops.attention.concat_mla",
    "sglang.kernels.ops.kvcache.set_mla_kv_buffer",
    "sglang.kernels.ops.quantization.awq_dequantize",
)


def _patch_module_load_jit_with_mira(module) -> None:
    """Add the SUPA Mira mode to this ops module's JIT calls only."""
    original = getattr(module, "load_jit", None)
    if original is None or getattr(original, "_supa_mira_flag_patch", False):
        return

    @wraps(original)
    def load_jit_with_mira(*args, **kwargs):
        extra = [
            flag
            for flag in (kwargs.pop("extra_cuda_cflags", None) or ())
            if flag not in ("-use-mira=true", "-use-mira=false")
        ]
        extra.append("-use-mira=true")
        kwargs["extra_cuda_cflags"] = extra
        return original(*args, **kwargs)

    load_jit_with_mira._supa_mira_flag_patch = True
    module.load_jit = load_jit_with_mira


def register_kernels_ops_jit_flag_hooks() -> None:
    """Inject per-op SUPA Mira flags into the three alias-JIT modules.

    Lifecycle: plugin-register-immediate, after the JIT package alias and before
    any kernels.ops JIT cache is populated. The target modules may be imported
    already; their module-global ``load_jit`` binding is replaced in place.
    Manifest: kernels.ops-supa-jit-mira-flags.
    """
    for name in _MIRA_FLAG_MODULES:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError:
                continue
        _patch_module_load_jit_with_mira(module)
