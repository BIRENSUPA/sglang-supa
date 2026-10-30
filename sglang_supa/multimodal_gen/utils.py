from __future__ import annotations


def _find_nccl_library() -> str:
    return "libbccl.so.2"


def register_multimodal_utils_hooks() -> None:
    import sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl_wrapper as pynccl_module

    pynccl_module._find_nccl_library = _find_nccl_library
