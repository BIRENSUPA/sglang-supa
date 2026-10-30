"""SUPA contiguous GDN projection kernel and upstream binding."""

from __future__ import annotations

import importlib
import math

import torch
import triton
import triton.language as tl

# =============================================================================
# SUPA-optimized contiguous kernel (SUPA platform only; CUDA path unchanged)
#
# Same CONTIGUOUS layout as the kernel above, but each program handles
# GROUPS_PER_PROGRAM head groups. Since q/k/v/z/b/a are each stored contiguously
# across groups, adjacent groups form one span, so the per-group copies collapse
# into a single vectorized load/store — shrinking grid[1] by GROUPS_PER_PROGRAM.
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_supa_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    i_bs, i_group_block = tl.program_id(0), tl.program_id(1)
    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
    TOTAL_QKVZ: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V + TOTAL_V
    TOTAL_BA: tl.constexpr = NUM_HEADS_V * 2
    QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V
    QK_ELEMS: tl.constexpr = GROUPS_PER_PROGRAM * HEAD_QK
    V_ELEMS: tl.constexpr = GROUPS_PER_PROGRAM * V_PER_GROUP * HEAD_V
    BA_ELEMS: tl.constexpr = GROUPS_PER_PROGRAM * V_PER_GROUP

    base_in = mixed_qkvz + i_bs * TOTAL_QKVZ
    qk_group = i_group_block * GROUPS_PER_PROGRAM
    qk_base = qk_group * HEAD_QK
    v_base = qk_group * V_PER_GROUP * HEAD_V
    ba_group_base = qk_group * V_PER_GROUP
    qk_idx = tl.arange(0, QK_ELEMS)
    v_idx = tl.arange(0, V_ELEMS)
    ba_idx = tl.arange(0, BA_ELEMS)

    q = tl.load(base_in + qk_base + qk_idx)
    k = tl.load(base_in + TOTAL_Q + qk_base + qk_idx)
    v = tl.load(base_in + TOTAL_Q + TOTAL_K + v_base + v_idx)
    z_ld = tl.load(base_in + TOTAL_Q + TOTAL_K + TOTAL_V + v_base + v_idx)
    ba_base = mixed_ba + i_bs * TOTAL_BA
    b_ld = tl.load(ba_base + ba_group_base + ba_idx)
    a_ld = tl.load(ba_base + NUM_HEADS_V + ba_group_base + ba_idx)

    base_out = mixed_qkv + i_bs * QKV_DIM_T
    tl.store(base_out + qk_base + qk_idx, q)
    tl.store(base_out + NUM_HEADS_QK * HEAD_QK + qk_base + qk_idx, k)
    tl.store(base_out + NUM_HEADS_QK * HEAD_QK * 2 + v_base + v_idx, v)
    tl.store(z + i_bs * NUM_HEADS_V * HEAD_V + v_base + v_idx, z_ld)
    tl.store(b + i_bs * NUM_HEADS_V + ba_group_base + ba_idx, b_ld)
    tl.store(a + i_bs * NUM_HEADS_V + ba_group_base + ba_idx, a_ld)


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """SUPA implementation of the upstream contiguous-layout entry point."""
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    groups_per_program = math.gcd(16, num_heads_qk)
    grid = (batch * seq_len, triton.cdiv(num_heads_qk, groups_per_program))
    fused_qkvzba_split_reshape_cat_contiguous_supa_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        groups_per_program,
        num_warps=8,
        num_stages=1,
    )
    return mixed_qkv, z, b, a


def register_triton_gdn_fused_proj_patch() -> None:
    """Bind the SUPA contiguous GDN entry point.

    Lifecycle: plugin-register-immediate. The patch must be installed before GDN
    callers import and cache this entry point.
    Manifest: kernels.ops-supa-attention-gdn.
    """
    upstream = importlib.import_module(
        "sglang.kernels.ops.attention.triton_gdn_fused_proj"
    )
    if getattr(upstream, "_supa_kernel_patched", False):
        return
    upstream.fused_qkvzba_split_reshape_cat_contiguous = (
        fused_qkvzba_split_reshape_cat_contiguous
    )
    upstream._supa_kernel_patched = True
