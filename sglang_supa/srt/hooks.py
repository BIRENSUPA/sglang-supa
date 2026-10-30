from __future__ import annotations


def register_srt_hooks() -> None:
    """Register the active adaptations owned by ``sglang_supa.srt``.

    This is an explicit orchestration layer. Registration order matters because
    some helpers patch module state immediately while HookRegistry helpers only
    take effect later during ``HookRegistry.apply_hooks()``.
    """
    from sglang_supa.srt.environ import register_envs_patch
    from sglang_supa.srt.utils.common import register_safe_unpickler_patch

    # These mutate process configuration and must precede imports that may read it.
    register_envs_patch()
    register_safe_unpickler_patch()

    from sglang_supa.srt.arg_groups.overrides import register_overrides_hooks
    from sglang_supa.srt.distributed.device_communicators.cuda_wrapper import (
        register_cuda_wrapper_hooks,
    )
    from sglang_supa.srt.distributed.device_communicators.custom_all_reduce import (
        register_custom_all_reduce_hooks,
    )
    from sglang_supa.srt.distributed.device_communicators.custom_all_reduce_utils import (
        register_custom_all_reduce_utils_hooks,
    )
    from sglang_supa.srt.distributed.device_communicators.pynccl_wrapper import (
        register_pynccl_wrapper_hooks,
    )
    from sglang_supa.srt.distributed.device_communicators.triton_symm_mem_ag import (
        register_triton_symm_mem_ag_hooks,
    )
    from sglang_supa.srt.layers.activation import register_activation_hooks
    from sglang_supa.srt.layers.attention.linear.gdn_flashinfer import (
        register_gdn_flashinfer_hooks,
    )
    from sglang_supa.srt.layers.moe.cutlass_moe import register_cutlass_moe_hooks
    from sglang_supa.srt.layers.moe.moe_runner.deep_gemm import (
        register_deep_gemm_moe_runner_hooks,
    )
    from sglang_supa.srt.layers.quantization.awq import register_awq_hooks
    from sglang_supa.srt.layers.quantization.fp8 import register_fp8_hooks
    from sglang_supa.srt.layers.quantization.fp8_utils import (
        register_fp8_utils_hooks,
    )
    from sglang_supa.srt.layers.quantization.utils import register_quant_utils_hooks
    from sglang_supa.srt.model_executor.model_runner_components.load_model_utils import (
        register_load_model_utils_patch,
    )
    from sglang_supa.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        register_full_cuda_graph_backend_hooks,
    )
    from sglang_supa.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        register_is_streaming_capturing_hooks,
    )
    from sglang_supa.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        register_forward_mla_hooks,
    )
    from sglang_supa.srt.models.deepseek_common.utils import (
        register_deepseek_common_utils_hooks,
    )
    from sglang_supa.srt.server_args import register_server_args_patch
    from sglang_supa.srt.speculative.eagle_utils import register_eagle_utils_hooks

    # Quantization hooks must be registered before the model and MoE hooks that
    # resolve their backend functions.
    register_awq_hooks()
    register_fp8_utils_hooks()
    register_quant_utils_hooks()
    register_fp8_hooks()

    # Communication and common runtime hooks.
    register_pynccl_wrapper_hooks()
    register_cuda_wrapper_hooks()
    register_custom_all_reduce_hooks()
    register_custom_all_reduce_utils_hooks()
    register_triton_symm_mem_ag_hooks()
    # The custom-op PrivateUse1 patch is intentionally installed by the
    # top-level plugin before this orchestrator imports eager custom ops.

    # Model, attention, speculative-decoding, and execution hooks.
    register_deepseek_common_utils_hooks()
    register_forward_mla_hooks()
    register_gdn_flashinfer_hooks()
    register_overrides_hooks()
    register_activation_hooks()
    register_cutlass_moe_hooks()
    register_deep_gemm_moe_runner_hooks()
    register_eagle_utils_hooks()
    register_load_model_utils_patch()
    register_full_cuda_graph_backend_hooks()
    register_is_streaming_capturing_hooks()

    # Immediate server-argument registry updates.
    register_server_args_patch()
