"""Correctness test for JIT fused_add_rmsnorm on SUPA.

Verifies that the JIT-compiled fused_add_rmsnorm kernel produces results
matching a pure-torch reference implementation.

Requires a functional SUPA/CUDA GPU. Skips gracefully when unavailable.
Plugins are loaded by test/conftest.py before this module runs.

Run:
    pytest test/test_fused_add_rmsnorm.py -v
"""

from __future__ import annotations

import sys

import pytest
import torch

_DEVICE = "cuda"


_stream_registered = False


def _ensure_tvm_ffi_stream():
    """Register the current torch CUDA/SUPA stream with tvm-ffi for kDLCUDA."""
    global _stream_registered
    if _stream_registered:
        return
    import tvm_ffi

    kDLCUDA = 2
    device_id = torch.cuda.current_device()
    stream_ptr = torch.cuda.current_stream().cuda_stream
    tvm_ffi.use_raw_stream(tvm_ffi.device(kDLCUDA, device_id), stream_ptr).__enter__()
    _stream_registered = True


def _has_functional_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device=_DEVICE)
        (t + 1).item()
    except Exception:
        return False
    return True


def _ref_fused_add_rmsnorm(input, residual, weight, eps):
    """torch reference: input += residual, then rmsnorm(input, weight)."""
    x = input.float() + residual.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps) * weight.float()
    return normed.to(input.dtype), x.to(input.dtype)


@pytest.mark.parametrize("hidden_size", [256, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_add_rmsnorm_correctness(hidden_size, dtype):
    if not _has_functional_gpu():
        pytest.skip("No functional GPU for kernel execution")
    _ensure_tvm_ffi_stream()
    from sglang.kernels.ops.layernorm.norm import (
        _jit_fused_add_rmsnorm_module,
        is_supported_jit_fused_add_rmsnorm_hidden_size,
    )

    if not is_supported_jit_fused_add_rmsnorm_hidden_size(hidden_size):
        pytest.skip(f"hidden_size={hidden_size} not supported")

    module = _jit_fused_add_rmsnorm_module(dtype, False)
    B, eps = 4, 1e-6
    x = torch.randn(B, hidden_size, dtype=dtype, device=_DEVICE)
    residual = torch.randn(B, hidden_size, dtype=dtype, device=_DEVICE)
    weight = torch.randn(hidden_size, dtype=dtype, device=_DEVICE)

    normed_ref, updated_ref = _ref_fused_add_rmsnorm(x, residual, weight, eps)

    x_k, residual_k = x.clone(), residual.clone()
    module.fused_add_rmsnorm(x_k, residual_k, weight, eps)

    torch.testing.assert_close(residual_k, updated_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_k, normed_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
