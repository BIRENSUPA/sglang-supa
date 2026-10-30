from __future__ import annotations

from fastcore.basics import patch_to


def register_multimodal_pynccl_wrapper_hooks() -> None:
    from sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl_wrapper import (
        NCCLLibrary,
    )

    if getattr(NCCLLibrary, "_supa_pynccl_wrapper_patched", False):
        return

    original_init = NCCLLibrary.__init__

    @patch_to(NCCLLibrary, nm="__init__")
    def nccl_library_init(self, so_file=None):
        cls = type(self)
        if not getattr(cls, "_supa_exported_functions_patched", False):
            cls.exported_functions = [
                type(func)(
                    "bccl" + func.name[len("nccl") :]
                    if func.name.startswith("nccl")
                    else func.name,
                    func.restype,
                    func.argtypes,
                )
                for func in cls.exported_functions
            ]
            cls._supa_exported_functions_patched = True
        return original_init(self, so_file or "libbccl.so.2")

    NCCLLibrary._supa_pynccl_wrapper_patched = True
