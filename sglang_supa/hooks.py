from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    """Register every active SUPA adaptation for the current process.

    This function runs in lifecycle phase P3. Direct patches, module aliases,
    and registry writes below take effect immediately; HookRegistry-based
    registrations are only applied in P4 when ``load_plugins()`` invokes
    ``apply_hooks()``.

    The complete contracts and ordering constraints are tracked in
    ``docs/hooks/hooks_manifest.yaml``.
    """
    # Install FlashInfer compatibility symbols before any registration helper
    # imports quantization modules. Manifest: bootstrap.flashinfer-import-guard.
    _patch_incompatible_flashinfer_imports()

    # This direct patch must precede imports that define eager SGLang custom
    # ops. Manifest: custom-op.privateuse1-registration.
    from sglang_supa.srt.utils.common import register_direct_register_custom_op_patch

    register_direct_register_custom_op_patch()

    # Select the JIT package alias or overlay before any lazy JIT compilation,
    # then install the DSV4-specific loader arguments on that selected loader.
    _patch_kernels_jit()

    # Install owner-local kernel patches after JIT setup and before SRT hooks.
    # The optional diffusion patch only applies when its multimodal lazy proxy
    # already exists; the multimodal subsystem is registered below.
    from sglang_supa.kernels.ops.hooks import register_kernels_ops_patches

    register_kernels_ops_patches(jit_overlay_enabled=True)

    # SRT hooks are kept in their mirror-owned orchestrator. Their direct
    # patches run now; HookRegistry registrations remain deferred until P4.
    from sglang_supa.srt.hooks import register_srt_hooks

    register_srt_hooks()

    # Multimodal uses immediate patch_to/direct assignments rather than the SRT
    # HookRegistry. Keep the whole subsystem optional at plugin load because
    # its dependencies are not present in every SRT installation.
    try:
        _register_multimodal_gen_hooks()
    except Exception:
        logger.warning(
            "Multimodal hooks registration failed (non-fatal)", exc_info=True
        )


def _patch_incompatible_flashinfer_imports() -> None:
    """Keep optional 0.5.18 FlashInfer modules from blocking SUPA imports.

    The SUPA image currently carries FlashInfer 0.4.x, while upstream 0.5.18
    imports ``bmm_fp8`` unconditionally. Install that missing symbol and the
    missing TRT-LLM enum module before quantization imports. The enum fake does
    not claim kernel support: accessing its values logs and raises.

    Lifecycle: P3 plugin-register-immediate; it must precede quantization imports.
    Import contract: the FlashInfer module may be pre-imported, but dependent
    quantization modules may not; already-cached missing bindings and
    failed/partial imports are not repaired.
    Failure policy: required when the compatibility symbols are absent.
    Manifest: bootstrap.flashinfer-import-guard.
    """
    import flashinfer

    if not hasattr(flashinfer, "bmm_fp8"):

        def bmm_fp8(*args, **kwargs):
            raise RuntimeError(
                "flashinfer.bmm_fp8 is unavailable in the installed FlashInfer "
                "build; the SUPA backend does not support this FlashInfer FP8 "
                "BMM path."
            )

        flashinfer.bmm_fp8 = bmm_fp8

    from sglang_supa.srt.layers.quantization.mxfp4 import (
        install_flashinfer_tllm_enums_fake,
    )

    install_flashinfer_tllm_enums_fake()


def _patch_kernels_jit() -> None:
    """Install the SUPA JIT overlay before the first JIT load.

    The overlay keeps the upstream JIT package and patches its loader inputs
    (include paths, source resolution, link flags). The DSV4 metadata loader
    wrapper is then installed on top. Lifecycle: P3 plugin-register-immediate.
    """
    from sglang_supa.kernels.jit_overlay.hooks import patch_kernels_jit_overlay

    patch_kernels_jit_overlay()


def _register_multimodal_gen_hooks() -> None:
    """Apply optional multimodal P3 patches in dependency order.

    Platform enums and capability methods must be installed before config and
    model hooks inspect them. Each registration function has a separate
    ``multimodal.*`` contract in hooks_manifest.yaml.
    """
    from sglang_supa.multimodal_gen.configs.models.adapter.base import (
        register_multimodal_adapter_config_hooks,
    )
    from sglang_supa.multimodal_gen.configs.models.dits.base import (
        register_multimodal_dit_config_hooks,
    )
    from sglang_supa.multimodal_gen.configs.models.encoders.base import (
        register_multimodal_encoder_config_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.debug_utils.tensor_dump_patch import (
        register_multimodal_tensor_dump_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.distributed.device_communicators.pynccl_wrapper import (
        register_multimodal_pynccl_wrapper_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.distributed.parallel_state import (
        register_multimodal_parallel_state_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.layers.custom_op import (
        register_multimodal_custom_op_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.layers.layernorm import (
        register_multimodal_norm_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        register_multimodal_text_encoder_loader_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.loader.utils import (
        register_multimodal_loader_utils_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.models.dits.causal_wanvideo import (
        register_multimodal_causal_wanvideo_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.models.dits.flux import (
        register_multimodal_flux_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.models.dits.flux_2 import (
        register_multimodal_flux_2_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.models.dits.wanvideo import (
        register_multimodal_wanvideo_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.platforms import (
        register_multimodal_platform_init_hooks,
    )
    from sglang_supa.multimodal_gen.runtime.platforms.interface import (
        register_multimodal_platform_interface_hooks,
    )
    from sglang_supa.multimodal_gen.utils import register_multimodal_utils_hooks

    register_multimodal_platform_interface_hooks()
    register_multimodal_platform_init_hooks()
    register_multimodal_adapter_config_hooks()
    register_multimodal_dit_config_hooks()
    register_multimodal_encoder_config_hooks()
    register_multimodal_utils_hooks()
    register_multimodal_parallel_state_hooks()
    register_multimodal_pynccl_wrapper_hooks()
    register_multimodal_loader_utils_hooks()
    register_multimodal_custom_op_hooks()
    register_multimodal_norm_hooks()
    register_multimodal_text_encoder_loader_hooks()
    register_multimodal_tensor_dump_hooks()
    register_multimodal_flux_hooks()
    register_multimodal_flux_2_hooks()
    register_multimodal_wanvideo_hooks()
    register_multimodal_causal_wanvideo_hooks()
