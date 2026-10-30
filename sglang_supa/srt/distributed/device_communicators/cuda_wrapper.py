from __future__ import annotations

import ctypes
from typing import Optional

from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    find_loaded_library,
)
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def supa_rt_library_init(self, so_file: Optional[str] = None):
    """Initialize the upstream wrapper with SUPA runtime symbols."""
    if so_file is None:
        so_file = find_loaded_library("libsupart")
        assert so_file is not None, "libsupart is not loaded in the current process"

    if so_file not in CudaRTLibrary.path_to_library_cache:
        CudaRTLibrary.path_to_library_cache[so_file] = ctypes.CDLL(so_file)
    self.lib = CudaRTLibrary.path_to_library_cache[so_file]

    if so_file not in CudaRTLibrary.path_to_dict_mapping:
        funcs = {}
        for func in CudaRTLibrary.exported_functions:
            f = getattr(self.lib, func.name.replace("cuda", "supa"))
            f.restype = func.restype
            f.argtypes = func.argtypes
            funcs[func.name] = f
        CudaRTLibrary.path_to_dict_mapping[so_file] = funcs
    self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]


def register_cuda_wrapper_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.cuda_wrapper.CudaRTLibrary.__init__",
        supa_rt_library_init,
        HookType.REPLACE,
    )
