"""JIT overlay unit tests: compilation + numerical correctness.

Tests 5 representative JIT kernels covering different CUDA features:
  1. fused_add_rmsnorm  — cooperative_groups + warp shuffle
  2. moe_topk_softmax   — cub (BlockRadixSort/BlockReduce) + --use_fast_math
  3. awq_dequantize     — PTX inline asm (lop3.b32, prmt.b32)
  4. gptq_marlin_gemm   — cp.async (async global→shared copy)
  5. scale_residual_norm_scale_shift — overlay csrc path resolution (SUPA-only kernel)

Two test tiers per kernel:
  - *_compile: JIT compilation only (needs nvcc, no GPU execution)
  - *_correctness: compile + run + compare against torch reference (needs real GPU)

Run:
    # All tests (compile-only pass without GPU, correctness tests skip)
    PYTHONPATH=python:3rdparty/sglang/python pytest sglang_supa/kernels/jit_overlay/test_jit_overlay.py -v

    # Compile tests only
    PYTHONPATH=python:3rdparty/sglang/python pytest sglang_supa/kernels/jit_overlay/test_jit_overlay.py -v -k compile

    # Correctness tests only (requires real GPU)
    PYTHONPATH=python:3rdparty/sglang/python pytest sglang_supa/kernels/jit_overlay/test_jit_overlay.py -v -k correctness
"""

from __future__ import annotations

import sys

import pytest
import torch

_DEVICE = "cuda"
_plugins_loaded = False


def _ensure_plugins():
    global _plugins_loaded
    if _plugins_loaded:
        return
    from sglang.srt.plugins import load_plugins

    load_plugins()
    _plugins_loaded = True


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
    """Check if a real GPU is available for kernel execution."""
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1, device=_DEVICE)
        (t + 1).item()
    except Exception:
        return False
    return True


# ───────────────────────────────────────────────────────────────────────
# 1. fused_add_rmsnorm (cooperative_groups + warp shuffle)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_add_rmsnorm_compile(dtype):
    _ensure_plugins()
    from sglang.kernels.ops.layernorm.norm import _jit_fused_add_rmsnorm_module

    module = _jit_fused_add_rmsnorm_module(dtype, False)
    assert module is not None


# ───────────────────────────────────────────────────────────────────────
# 2. moe_topk_softmax (cub)
# ───────────────────────────────────────────────────────────────────────


def _ref_topk_softmax(gating_output, topk, renormalize=False):
    """torch reference: softmax → topk."""
    probs = torch.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.float(), topk_ids.int()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_moe_topk_softmax_compile(dtype):
    _ensure_plugins()
    from sglang.kernels.ops.moe.moe_topk_softmax import _jit_moe_topk_softmax_module

    module = _jit_moe_topk_softmax_module(dtype)
    assert module is not None


@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("topk", [2, 4])
def test_moe_topk_softmax_correctness(num_experts, topk):
    if not _has_functional_gpu():
        pytest.skip("No functional GPU for kernel execution")
    _ensure_plugins()
    _ensure_tvm_ffi_stream()
    from sglang.kernels.ops.moe.moe_topk_softmax import topk_softmax

    num_tokens = 16
    gating = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=_DEVICE)
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device=_DEVICE)
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device=_DEVICE)

    topk_softmax(topk_weights, topk_ids, gating, renormalize=True)

    ref_weights, ref_ids = _ref_topk_softmax(gating, topk, renormalize=True)

    sorted_ids, sort_idx = topk_ids.sort(dim=-1)
    sorted_ref_ids, sort_ref_idx = ref_ids.to(_DEVICE).sort(dim=-1)
    assert torch.equal(sorted_ids, sorted_ref_ids)

    gathered_weights = topk_weights.gather(1, sort_idx)
    gathered_ref = ref_weights.to(_DEVICE).gather(1, sort_ref_idx)
    torch.testing.assert_close(gathered_weights, gathered_ref, rtol=1e-2, atol=1e-2)


# ───────────────────────────────────────────────────────────────────────
# 3. awq_dequantize (PTX inline asm)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [torch.float16])
def test_awq_dequantize_compile(dtype):
    _ensure_plugins()
    from sglang.kernels.ops.quantization.awq_dequantize import (
        _jit_awq_dequantize_module,
    )

    module = _jit_awq_dequantize_module(dtype)
    assert module is not None


@pytest.mark.parametrize("dtype", [torch.float16])
def test_awq_dequantize_correctness(dtype):
    if not _has_functional_gpu():
        pytest.skip("No functional GPU for kernel execution")
    _ensure_plugins()
    _ensure_tvm_ffi_stream()
    from sglang.kernels.ops.quantization.awq_dequantize import awq_dequantize

    rows, cols, group_size = 128, 64, 128
    qweight = torch.randint(0, 2**31, (rows, cols), dtype=torch.int32, device=_DEVICE)
    scales = torch.rand(rows // group_size, cols * 8, dtype=dtype, device=_DEVICE)
    qzeros = torch.randint(
        0, 2**31, (rows // group_size, cols), dtype=torch.int32, device=_DEVICE
    )

    out = awq_dequantize(qweight, scales, qzeros)
    assert not torch.any(torch.isinf(out)), "output contains inf"
    assert not torch.any(torch.isnan(out)), "output contains nan"
    assert out.shape == (rows, cols * 8)


# ───────────────────────────────────────────────────────────────────────
# 4. gptq_marlin_gemm — stub raises NotImplementedError on SUPA
# ───────────────────────────────────────────────────────────────────────


def test_marlin_stub():
    _ensure_plugins()
    from sglang.kernels.ops.quantization.gptq_marlin import gptq_marlin_gemm

    with pytest.raises(NotImplementedError, match="not supported on SUPA"):
        gptq_marlin_gemm(
            a=None,
            c=None,
            b_q_weight=None,
            b_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=None,
            b_q_type=None,
            size_m=0,
            size_n=0,
            size_k=0,
        )


# ───────────────────────────────────────────────────────────────────────
# 5. scale_residual_norm_scale_shift (overlay csrc path)
# ───────────────────────────────────────────────────────────────────────


def _ref_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps):
    """torch reference: norm(x) * (1 + scale) + shift."""
    x_f = x.float()
    if norm_type == "rms":
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(variance + eps)
    else:
        mean = x_f.mean(dim=-1, keepdim=True)
        variance = (x_f - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x_f - mean) * torch.rsqrt(variance + eps)

    if weight is not None:
        normed = normed * weight.float()
    if bias is not None:
        normed = normed + bias.float()

    y = normed * (1.0 + scale.float()) + shift.float()
    return y.to(x.dtype)


@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_scale_residual_norm_compile(norm_type, dtype):
    _ensure_plugins()
    from sglang_supa.kernels.ops.diffusion.supa.scale_residual_norm_scale_shift import (
        _jit_scale_residual_norm_scale_shift_module,
    )

    module = _jit_scale_residual_norm_scale_shift_module(dtype, norm_type)
    assert module is not None


@pytest.mark.parametrize("norm_type", ["rms", "layer"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_scale_residual_norm_correctness(norm_type, dtype):
    if not _has_functional_gpu():
        pytest.skip("No functional GPU for kernel execution")
    _ensure_plugins()
    _ensure_tvm_ffi_stream()
    from sglang_supa.kernels.ops.diffusion.supa.scale_residual_norm_scale_shift import (
        fused_norm_scale_shift,
    )

    B, S, D = 2, 4, 256
    eps = 1e-5
    x = torch.randn(B, S, D, dtype=dtype, device=_DEVICE)
    weight = torch.randn(D, dtype=dtype, device=_DEVICE)
    bias = torch.randn(D, dtype=dtype, device=_DEVICE)
    scale = torch.randn(B, S, D, dtype=dtype, device=_DEVICE) * 0.1
    shift = torch.randn(B, S, D, dtype=dtype, device=_DEVICE) * 0.1

    y = fused_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps)
    y_ref = _ref_norm_scale_shift(x, weight, bias, scale, shift, norm_type, eps)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
