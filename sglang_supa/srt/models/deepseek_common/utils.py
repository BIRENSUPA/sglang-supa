from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def awq_dequantize_func():
    from sglang.kernels.ops.quantization.awq_dequantize import awq_dequantize

    return awq_dequantize


def register_deepseek_common_utils_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.models.deepseek_common.utils.awq_dequantize_func",
        awq_dequantize_func,
        HookType.REPLACE,
    )
