"""SUPA fused QK RMSNorm/RoPE kernel and upstream binding patch."""

from __future__ import annotations

import importlib

import triton
import triton.language as tl
from sglang_supa.kernels.ops.utils import SupaApiWrapper

# =============================================================================
# SUPA-optimized variant (SUPA platform only; CUDA path unchanged)
#
# Each program tiles HEADS_PER_BLOCK heads (default 4) with vectorized
# max_contiguous/multiple_of loads, eviction_policy hints, precomputed
# INV_HEAD_DIM, and a BASE_POS offset for mrope positions — shrinking grid[1]
# by HEADS_PER_PROGRAM.
# =============================================================================


@triton.jit
def _fused_qk_rmsnorm_rope_gate_supa_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    INV_HEAD_DIM: tl.constexpr,
    FP16: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head_block = tl.program_id(1)
    head = head_block * HEADS_PER_BLOCK
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)
    out_dtype = tl.float16 if FP16 else tl.bfloat16

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
    else:
        q_head_stride = HEAD_DIM * 2 if HAS_GATE else HEAD_DIM
        in_base = q_gate_ptr + token * stride_qg_t + local_head * q_head_stride
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    head_raw = tl.arange(0, HEAD_BLOCK)
    head_offs = tl.max_contiguous(tl.multiple_of(head_raw, 64), 64)
    rot_raw = tl.arange(0, ROT_HALF_BLOCK)
    rot_offs = tl.max_contiguous(tl.multiple_of(rot_raw, 32), 32)

    if (
        HEADS_PER_BLOCK == 4
        and HEAD_BLOCK == HEAD_DIM
        and ROT_HALF_BLOCK == HALF_ROTARY
    ):
        pos = tl.load(positions_ptr + token).to(tl.int64)
        cache_off = pos * stride_cos_t
        head1 = head + 1
        head2 = head + 2
        head3 = head + 3
        total_heads = NUM_Q_HEADS + NUM_KV_HEADS
        valid2 = head2 < total_heads
        valid3 = head3 < total_heads
        local_head1 = tl.where(is_k, head1 - NUM_Q_HEADS, head1)
        local_head2 = tl.where(is_k, head2 - NUM_Q_HEADS, head2)
        local_head3 = tl.where(is_k, head3 - NUM_Q_HEADS, head3)

        if is_k:
            in_base1 = k_ptr + token * stride_k_t + local_head1 * HEAD_DIM
            out_base1 = k_out_ptr + token * stride_ko_t + local_head1 * HEAD_DIM
            in_base2 = k_ptr + token * stride_k_t + local_head2 * HEAD_DIM
            out_base2 = k_out_ptr + token * stride_ko_t + local_head2 * HEAD_DIM
            in_base3 = k_ptr + token * stride_k_t + local_head3 * HEAD_DIM
            out_base3 = k_out_ptr + token * stride_ko_t + local_head3 * HEAD_DIM
        else:
            q_head_stride = HEAD_DIM * 2 if HAS_GATE else HEAD_DIM
            in_base1 = q_gate_ptr + token * stride_qg_t + local_head1 * q_head_stride
            out_base1 = q_out_ptr + token * stride_qo_t + local_head1 * HEAD_DIM
            in_base2 = q_gate_ptr + token * stride_qg_t + local_head2 * q_head_stride
            out_base2 = q_out_ptr + token * stride_qo_t + local_head2 * HEAD_DIM
            in_base3 = q_gate_ptr + token * stride_qg_t + local_head3 * q_head_stride
            out_base3 = q_out_ptr + token * stride_qo_t + local_head3 * HEAD_DIM

        x1 = tl.load(in_base + rot_offs, eviction_policy="evict_first").to(tl.float32)
        x2 = tl.load(
            in_base + HALF_ROTARY + rot_offs, eviction_policy="evict_first"
        ).to(tl.float32)
        x1_b = tl.load(in_base1 + rot_offs, eviction_policy="evict_first").to(
            tl.float32
        )
        x2_b = tl.load(
            in_base1 + HALF_ROTARY + rot_offs, eviction_policy="evict_first"
        ).to(tl.float32)
        x1_c = tl.load(
            in_base2 + rot_offs,
            mask=valid2,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x2_c = tl.load(
            in_base2 + HALF_ROTARY + rot_offs,
            mask=valid2,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x1_d = tl.load(
            in_base3 + rot_offs,
            mask=valid3,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x2_d = tl.load(
            in_base3 + HALF_ROTARY + rot_offs,
            mask=valid3,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w1 = tl.load(w_ptr + rot_offs, eviction_policy="evict_last").to(tl.float32)
        w2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, eviction_policy="evict_last").to(
            tl.float32
        )

        pass_mask = head_offs >= ROTARY_DIM
        w_pass = tl.load(
            w_ptr + head_offs,
            mask=pass_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        cos = tl.load(
            cos_sin_cache_ptr + cache_off + rot_offs,
            eviction_policy="evict_last",
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x_pass = tl.load(
            in_base + head_offs,
            mask=pass_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x_pass_b = tl.load(
            in_base1 + head_offs,
            mask=pass_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x_pass_c = tl.load(
            in_base2 + head_offs,
            mask=pass_mask & valid2,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x_pass_d = tl.load(
            in_base3 + head_offs,
            mask=pass_mask & valid3,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        var = (
            tl.sum(x1 * x1, axis=0)
            + tl.sum(x2 * x2, axis=0)
            + tl.sum(x_pass * x_pass, axis=0)
        ) * INV_HEAD_DIM
        var_b = (
            tl.sum(x1_b * x1_b, axis=0)
            + tl.sum(x2_b * x2_b, axis=0)
            + tl.sum(x_pass_b * x_pass_b, axis=0)
        ) * INV_HEAD_DIM
        var_c = (
            tl.sum(x1_c * x1_c, axis=0)
            + tl.sum(x2_c * x2_c, axis=0)
            + tl.sum(x_pass_c * x_pass_c, axis=0)
        ) * INV_HEAD_DIM
        var_d = (
            tl.sum(x1_d * x1_d, axis=0)
            + tl.sum(x2_d * x2_d, axis=0)
            + tl.sum(x_pass_d * x_pass_d, axis=0)
        ) * INV_HEAD_DIM
        inv_rms = tl.rsqrt(var + EPS)
        inv_rms_b = tl.rsqrt(var_b + EPS)
        inv_rms_c = tl.rsqrt(var_c + EPS)
        inv_rms_d = tl.rsqrt(var_d + EPS)

        x_pass_norm = (x_pass * inv_rms * (w_pass + 1.0)).to(out_dtype).to(tl.float32)
        x_pass_norm_b = (
            (x_pass_b * inv_rms_b * (w_pass + 1.0)).to(out_dtype).to(tl.float32)
        )
        x_pass_norm_c = (
            (x_pass_c * inv_rms_c * (w_pass + 1.0)).to(out_dtype).to(tl.float32)
        )
        x_pass_norm_d = (
            (x_pass_d * inv_rms_d * (w_pass + 1.0)).to(out_dtype).to(tl.float32)
        )
        if HAS_PASS:
            tl.store(
                out_base + head_offs,
                x_pass_norm,
                mask=pass_mask,
                eviction_policy="evict_first",
            )
            tl.store(
                out_base1 + head_offs,
                x_pass_norm_b,
                mask=pass_mask,
                eviction_policy="evict_first",
            )
            tl.store(
                out_base2 + head_offs,
                x_pass_norm_c,
                mask=pass_mask & valid2,
                eviction_policy="evict_first",
            )
            tl.store(
                out_base3 + head_offs,
                x_pass_norm_d,
                mask=pass_mask & valid3,
                eviction_policy="evict_first",
            )

        xr1 = (x1 * inv_rms * (w1 + 1.0)).to(out_dtype).to(tl.float32)
        xr2 = (x2 * inv_rms * (w2 + 1.0)).to(out_dtype).to(tl.float32)
        xr1_b = (x1_b * inv_rms_b * (w1 + 1.0)).to(out_dtype).to(tl.float32)
        xr2_b = (x2_b * inv_rms_b * (w2 + 1.0)).to(out_dtype).to(tl.float32)
        xr1_c = (x1_c * inv_rms_c * (w1 + 1.0)).to(out_dtype).to(tl.float32)
        xr2_c = (x2_c * inv_rms_c * (w2 + 1.0)).to(out_dtype).to(tl.float32)
        xr1_d = (x1_d * inv_rms_d * (w1 + 1.0)).to(out_dtype).to(tl.float32)
        xr2_d = (x2_d * inv_rms_d * (w2 + 1.0)).to(out_dtype).to(tl.float32)

        tl.store(
            out_base + rot_offs, xr1 * cos - xr2 * sin, eviction_policy="evict_first"
        )
        tl.store(
            out_base + HALF_ROTARY + rot_offs,
            xr2 * cos + xr1 * sin,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base1 + rot_offs,
            xr1_b * cos - xr2_b * sin,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base1 + HALF_ROTARY + rot_offs,
            xr2_b * cos + xr1_b * sin,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base2 + rot_offs,
            xr1_c * cos - xr2_c * sin,
            mask=valid2,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base2 + HALF_ROTARY + rot_offs,
            xr2_c * cos + xr1_c * sin,
            mask=valid2,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base3 + rot_offs,
            xr1_d * cos - xr2_d * sin,
            mask=valid3,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base3 + HALF_ROTARY + rot_offs,
            xr2_d * cos + xr1_d * sin,
            mask=valid3,
            eviction_policy="evict_first",
        )

        # Gate copy for the remaining heads of the block (head0 handled by the
        # shared section below).
        if HAS_GATE and not is_k:
            tl.store(
                gate_out_ptr
                + token * stride_gate_t
                + local_head1 * HEAD_DIM
                + head_offs,
                tl.load(
                    in_base1 + HEAD_DIM + head_offs,
                    eviction_policy="evict_first",
                ),
            )
            tl.store(
                gate_out_ptr
                + token * stride_gate_t
                + local_head2 * HEAD_DIM
                + head_offs,
                tl.load(
                    in_base2 + HEAD_DIM + head_offs,
                    mask=valid2,
                    other=0.0,
                    eviction_policy="evict_first",
                ),
                mask=valid2,
            )
            tl.store(
                gate_out_ptr
                + token * stride_gate_t
                + local_head3 * HEAD_DIM
                + head_offs,
                tl.load(
                    in_base3 + HEAD_DIM + head_offs,
                    mask=valid3,
                    other=0.0,
                    eviction_policy="evict_first",
                ),
                mask=valid3,
            )
    else:
        head_mask = head_offs < HEAD_DIM
        x = tl.load(
            in_base + head_offs,
            mask=head_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(
            w_ptr + head_offs,
            mask=head_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        var = tl.sum(x * x, axis=0) * INV_HEAD_DIM
        inv_rms = tl.rsqrt(var + EPS)
        x_norm = (x * inv_rms * (w + 1.0)).to(out_dtype).to(tl.float32)

        if HAS_PASS:
            pass_mask = (head_offs < HEAD_DIM) & (head_offs >= ROTARY_DIM)
            tl.store(
                out_base + head_offs,
                x_norm,
                mask=pass_mask,
                eviction_policy="evict_first",
            )

        rot_mask = rot_offs < HALF_ROTARY
        xr1 = tl.load(
            in_base + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        xr2 = tl.load(
            in_base + HALF_ROTARY + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        wr1 = tl.load(
            w_ptr + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        wr2 = tl.load(
            w_ptr + HALF_ROTARY + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        xr1 = (xr1 * inv_rms * (wr1 + 1.0)).to(out_dtype).to(tl.float32)
        xr2 = (xr2 * inv_rms * (wr2 + 1.0)).to(out_dtype).to(tl.float32)

        pos = tl.load(positions_ptr + token).to(tl.int64)
        cache_off = pos * stride_cos_t
        cos = tl.load(
            cos_sin_cache_ptr + cache_off + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs,
            mask=rot_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        tl.store(
            out_base + rot_offs,
            xr1 * cos - xr2 * sin,
            mask=rot_mask,
            eviction_policy="evict_first",
        )
        tl.store(
            out_base + HALF_ROTARY + rot_offs,
            xr2 * cos + xr1 * sin,
            mask=rot_mask,
            eviction_policy="evict_first",
        )

    # Gate copy (Q heads only)
    if HAS_GATE and not is_k:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        head_mask = head_offs < HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _adapt_fused_qk_grid(grid, _args, _kwargs):
    heads_per_block = 4
    if len(grid) != 2:
        raise ValueError(f"Expected a 2D Triton grid, got {grid!r}")
    return (grid[0], triton.cdiv(grid[1], heads_per_block))


def _adapt_fused_qk_args(_args, kwargs):
    kwargs["INV_HEAD_DIM"] = 1.0 / kwargs["HEAD_DIM"]
    kwargs["HEADS_PER_BLOCK"] = 4
    kwargs["num_warps"] = 2
    kwargs["num_stages"] = 3


def register_fused_qk_rmsnorm_rope_gate_patch() -> None:
    """Bind the SUPA kernel into the upstream QK/RoPE kernel global.

    Lifecycle: plugin-register-immediate. The upstream caller resolves its
    kernel through the defining module global. Manifest:
    kernels.ops-supa-attention-qk.
    """
    upstream = importlib.import_module(
        "sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate"
    )
    if getattr(upstream, "_supa_kernel_patched", False):
        return

    upstream._fused_qk_rmsnorm_rope_gate_kernel = SupaApiWrapper(
        _fused_qk_rmsnorm_rope_gate_supa_kernel,
        _adapt_fused_qk_grid,
        _adapt_fused_qk_args,
    )
    upstream._ENABLE_PDL = False
    upstream._supa_kernel_patched = True
