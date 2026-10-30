from __future__ import annotations

import ctypes
import logging
import platform
from typing import Any, Dict, Optional

from sglang.srt.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)


def find_nccl_library():
    return "libbccl.so.2"


def bccl_library_init(self, so_file: Optional[str] = None):
    """Initialize the upstream NCCL wrapper against the SUPA BCCL library."""
    so_file = so_file or find_nccl_library()

    try:
        if so_file not in NCCLLibrary.path_to_dict_mapping:
            NCCLLibrary.path_to_library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = NCCLLibrary.path_to_library_cache[so_file]
    except Exception as e:
        logger.error(
            "Failed to load BCCL library from %s on platform %s.",
            so_file,
            platform.platform(),
        )
        raise e

    if so_file not in NCCLLibrary.path_to_dict_mapping:
        funcs: Dict[str, Any] = {}
        exported_functions = list(NCCLLibrary.exported_functions)
        if hasattr(self.lib, "bcclCommWindowRegister"):
            exported_functions.extend(NCCLLibrary.exported_functions_symm_mem)
        for func in exported_functions:
            f = getattr(self.lib, func.name.replace("nccl", "bccl"))
            f.restype = func.restype
            f.argtypes = func.argtypes
            funcs[func.name] = f
        NCCLLibrary.path_to_dict_mapping[so_file] = funcs
    self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]


def register_pynccl_wrapper_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.__init__",
        bccl_library_init,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.pynccl_wrapper.find_nccl_library",
        find_nccl_library,
        HookType.REPLACE,
    )
