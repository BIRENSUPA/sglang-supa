"""SUPA-only overlay for the upstream ``sglang.kernels.jit`` package.

The upstream JIT sources remain owned and imported from SGLang.  This module
patches Python-level build infrastructure (arch detection, include paths,
link flags, source resolution) so upstream kernel sources compile and link
correctly under the SUPA gstub nvcc toolchain.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from functools import wraps

import torch

_OVERLAY_ROOT = pathlib.Path(__file__).resolve().parent
_OVERLAY_CSRC = _OVERLAY_ROOT / "csrc"
_OVERLAY_INCLUDE = _OVERLAY_ROOT / "include"
_COMPAT_HEADER = _OVERLAY_INCLUDE / "sgl_kernel" / "device_compat.h"

_PATCHED = "_sglang_supa_jit_overlay_applied"


def _is_supa_runtime() -> bool:
    return hasattr(torch, "supa") or getattr(torch.version, "supa", None) is not None


def _repair_bindings(original, replacement, *, skip_modules=()):
    """Replace all module-level references to *original* with *replacement*."""
    skip = set(skip_modules)
    for module in list(sys.modules.values()):
        if module is None or module in skip:
            continue
        try:
            values = vars(module)
        except TypeError:
            continue
        for name, value in list(values.items()):
            if value is original:
                setattr(module, name, replacement)


def _patch_common():
    """Inject ``is_supa_runtime`` into the upstream common module."""
    from sglang.kernels.jit.utils.common import cache_once

    common = importlib.import_module("sglang.kernels.jit.utils.common")
    if hasattr(common, "is_supa_runtime"):
        return
    common.is_supa_runtime = cache_once(_is_supa_runtime)


def _patch_arch():
    """Force sm_90 target — gstub nvcc does not support sm_91a."""
    arch = importlib.import_module("sglang.kernels.jit.utils.arch")
    original_init = arch._init_jit_cuda_arch_once

    def _supa_init():
        if _is_supa_runtime():
            arch._CUDA_ARCH = arch.ArchInfo(major=9, minor=0, suffix="")
            return
        original_init()

    arch._init_jit_cuda_arch_once = _supa_init


def _patch_source_resolution():
    """Prefer overlay csrc files (SUPA-only kernels) over upstream."""
    spec_mod = importlib.import_module("sglang.kernels.jit.utils.compile.spec")
    original_resolve = spec_mod.resolve_sources
    upstream_csrc = spec_mod.KERNEL_PATH / "csrc"

    def _supa_resolve(files):
        if not _is_supa_runtime() or not files:
            return original_resolve(files)
        resolved = []
        for f in files:
            path = pathlib.Path(f)
            if path.is_absolute():
                try:
                    rel = path.resolve().relative_to(upstream_csrc.resolve())
                    overlay = _OVERLAY_CSRC / rel
                    if overlay.exists():
                        resolved.append(str(overlay.resolve()))
                        continue
                except ValueError:
                    pass
                resolved.append(str(path.resolve()))
            else:
                overlay = _OVERLAY_CSRC / path
                if overlay.exists():
                    resolved.append(str(overlay.resolve()))
                else:
                    resolved.append(str((upstream_csrc / path).resolve()))
        return tuple(resolved)

    spec_mod.resolve_sources = _supa_resolve
    _repair_bindings(original_resolve, _supa_resolve, skip_modules=(spec_mod,))


def _patch_loader():
    """Inject SUPA header include paths and device_compat.h into every JIT build."""
    loader = importlib.import_module("sglang.kernels.jit.utils.compile.loader")
    compile_pkg = importlib.import_module("sglang.kernels.jit.utils.compile")
    paths = importlib.import_module("sglang.kernels.jit.utils.compile.paths")

    if not _COMPAT_HEADER.is_file():
        raise FileNotFoundError(
            f"Missing required SUPA JIT overlay header: {_COMPAT_HEADER}"
        )

    upstream_include = str(pathlib.Path(paths.KERNEL_PATH) / "include")
    paths.DEFAULT_INCLUDE[:] = [str(_OVERLAY_INCLUDE), upstream_include]
    loader.DEFAULT_INCLUDE = paths.DEFAULT_INCLUDE

    force_include = ["-include", str(_COMPAT_HEADER)]
    original_load_jit = loader.load_jit

    @wraps(original_load_jit)
    def _supa_load_jit(*args, **kwargs):
        if _is_supa_runtime():
            extra = list(kwargs.pop("extra_cuda_cflags", None) or ())
            for flag in force_include:
                if flag not in extra:
                    extra.append(flag)
            kwargs["extra_cuda_cflags"] = extra
        return original_load_jit(*args, **kwargs)

    loader.load_jit = _supa_load_jit
    compile_pkg.load_jit = _supa_load_jit
    _repair_bindings(
        original_load_jit, _supa_load_jit, skip_modules=(loader, compile_pkg)
    )


def _patch_exports():
    """Ensure the upstream utils package exports SUPA-added symbols."""
    utils_pkg = importlib.import_module("sglang.kernels.jit.utils")
    common = importlib.import_module("sglang.kernels.jit.utils.common")
    if hasattr(common, "is_supa_runtime"):
        utils_pkg.is_supa_runtime = common.is_supa_runtime


def apply() -> None:
    """Patch upstream JIT modules once, before the first kernel load."""
    paths = importlib.import_module("sglang.kernels.jit.utils.compile.paths")
    if getattr(paths, _PATCHED, False):
        return
    if not _is_supa_runtime():
        return

    _patch_common()
    _patch_arch()
    _patch_source_resolution()
    _patch_loader()
    _patch_exports()

    setattr(paths, _PATCHED, True)
