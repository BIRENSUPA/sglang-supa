import logging

logger = logging.getLogger(__name__)


def patch_kernels_jit_overlay() -> None:
    """Patch the upstream JIT loader with SUPA build inputs.

    Lifecycle: plugin-register-immediate; the overlay must be installed before
    the first lazy kernel load. Import contract: pre-imported loader bindings
    are repaired by the overlay implementation. Manifest: kernels.jit-overlay.
    """
    from sglang_supa.kernels.jit_overlay.patch import apply

    apply()
