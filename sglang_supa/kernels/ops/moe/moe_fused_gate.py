"""SUPA router kernel and monkey-patch for upstream MoE gating."""

from __future__ import annotations

import importlib

import triton
import triton.language as tl
from sglang_supa.kernels.ops.utils import SupaApiWrapper


@triton.jit
def _router_triton_supa_kernel(
    scores_ptr,  # [M, N] fp32, GEMM output (raw logits)
    bias_ptr,  # [N]    fp32
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    BLOCK_M: tl.constexpr,  # rows processed per program (row tiling)
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    USE_PDL: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # prefetch bias before PDL wait
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )  # [BLOCK_N]

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    # Unmasked scores load when the full tile is in-bounds (constexpr-all-true mask).
    if BLOCK_N == N and (M % BLOCK_M) == 0:
        scores = tl.load(row_ptr).to(tl.float32)  # [BLOCK_M, BLOCK_N]
    else:
        mask2d = mask_m[:, None] & mask_n[None, :]
        scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(tl.float32)

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        # sqrt(softplus(x)) = sqrt(log1p(exp(x))); guard against overflow when x is large.
        sp = tl.where(scores > 20.0, scores, tl.log(1.0 + tl.exp(scores)))
        activated = tl.sqrt(sp)
        biased = activated + bias[None, :]
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias[None, :]
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]  # [BLOCK_M, 1]
        # OOB lanes carry -inf -> exp yields 0 exactly; no clamp needed.
        exp_row = tl.exp(biased - row_max)
        row_sum = tl.sum(exp_row, axis=1)[:, None]  # [BLOCK_M, 1]
        activated = exp_row / row_sum

    # softmax path already masked `biased` above; only sigmoid/sqrtsoftplus need it.
    if SCORING_FUNC != 2:
        biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    # NaN floor for sigmoid/sqrtsoftplus (exp/log can NaN at extremes); softmax can't.
    if SCORING_FUNC != 2:
        biased = tl.where(biased == biased, biased, -1e30)  # [BLOCK_M, BLOCK_N]

    # Grouped routing (DeepSeek-V3 noaux_tc): per-group score = sum of the top-2
    # biased values; keep TOPK_GROUP groups (lowest group id wins ties); mask the
    # experts of dropped groups to -inf before the top-k below. Weight is still the
    # bias-free `activated`. Constexpr N_GROUP <= 1 skips this entirely (ungrouped).
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
        group_of_n = offs_n // EXPERTS_PER_GROUP  # [BLOCK_N]
        group_score = tl.full([BLOCK_M, BLOCK_G], -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float("inf"))
            top1 = tl.max(vals, axis=1)[:, None]  # [BLOCK_M, 1]
            vals2 = tl.where(vals >= top1, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float("inf"), gcur)
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)
    routed_sum = tl.zeros([BLOCK_M, 1], dtype=tl.float32)

    # Manual max+argmin butterfly top-k; consume winner by masking to -inf.
    cur = biased  # [BLOCK_M, BLOCK_N]
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=1)[:, None]  # [BLOCK_M, 1]
        is_max = cur == max_val
        lane_id = tl.where(is_max, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)  # [BLOCK_M, 1]
        # softmax path: winner's biased value == max_val, so activated[win] = exp(max_val - row_max) / row_sum.
        if SCORING_FUNC == 2:
            win_activated = tl.exp(max_val - row_max) / row_sum  # [BLOCK_M,1]
        else:
            win_activated = tl.sum(
                tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
            )[
                :, None
            ]  # [BLOCK_M, 1]
        slot = offs_k[None, :] == k  # [1, BLOCK_K]
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        routed_sum += win_activated
        cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = routed_sum / routed_scaling_factor  # [BLOCK_M, 1]
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
        inv_norm = 1.0 / norm  # precompute reciprocal
        selected_vals = selected_vals * inv_norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def _adapt_router_launch_args(_args, kwargs):
    # The upstream caller still computes the grid and all shape-dependent
    # constexpr arguments. Only the SUPA launch configuration is overridden.
    kwargs["BLOCK_M"] = 8
    kwargs["num_warps"] = 4


def _adapt_router_grid(grid, args, kwargs):
    if len(grid) != 1:
        raise ValueError(f"Expected a 1D Triton grid, got {grid!r}")
    try:
        M = args[4]
    except IndexError:
        M = kwargs["M"]
    return (triton.cdiv(M, 8),)


def register_moe_fused_gate_patch() -> None:
    """Bind the SUPA router kernel into the upstream MoE caller.

    Lifecycle: plugin-register-immediate. The upstream public caller remains
    responsible for validation, fast paths, output allocation, and grid
    construction. Manifest: kernels.ops-supa-moe-fused-gate.
    """
    upstream = importlib.import_module("sglang.kernels.ops.moe.moe_fused_gate")
    if getattr(upstream, "_supa_kernel_patched", False):
        return
    replacement = SupaApiWrapper(
        _router_triton_supa_kernel,
        adapt_grid=_adapt_router_grid,
        adapt_args=_adapt_router_launch_args,
    )
    upstream._router_triton_kernel = replacement
    upstream._supa_kernel_patched = True
