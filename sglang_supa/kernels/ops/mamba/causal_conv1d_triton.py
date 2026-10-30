"""SUPA causal-convolution kernels and upstream runtime bindings."""

from __future__ import annotations

import importlib

import triton
import triton.language as tl
from sglang_supa.kernels.ops.utils import SupaApiWrapper


@triton.jit()
def _causal_conv1d_fwd_supa_kernel(
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_feats = tl.max_contiguous(tl.multiple_of(idx_feats, BLOCK_N), BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col0 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col1 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col0 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens
                col3 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col2 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col1 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok
                col0 = tl.load(
                    conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first"
                )
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0, eviction_policy="evict_first")
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            # tl.debug_barrier()  # NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0, eviction_policy="evict_first")

                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(
                    x_ptrs, mask_x, 0.0, eviction_policy="evict_first"
                )
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, eviction_policy="evict_first")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(
                        x_ptrs_1d, mask=mask_x_1d, eviction_policy="evict_first"
                    )
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(
                        x_ptrs_1d, mask=mask_x_1d, eviction_policy="evict_first"
                    )
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(
                        x_ptrs_1d, mask=mask_x_1d, eviction_policy="evict_first"
                    )

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            # acc = acc / (1 + tl.exp(-acc))
            acc = acc * tl.sigmoid(acc)
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token).to(tl.int64)
            * stride_o_token
            + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d, eviction_policy="evict_first")


@triton.jit()
def _causal_conv1d_update_supa_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accept_tokens_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    retrieve_parent_token_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices: tl.constexpr,
    stride_retrieve_next_token_seq: tl.constexpr,
    stride_retrieve_next_token_token: tl.constexpr,
    stride_retrieve_next_sibling_seq: tl.constexpr,
    stride_retrieve_next_sibling_token: tl.constexpr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    NP2_SEQLEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    # ruff: noqa: E501
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        if SAVE_INTERMEDIATE:
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accept_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim
    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    # R26: 2-D merge of conv_state cols. Instead of 3 (or state_len) separate
    # 1-D tl.load's each with stride_conv_state_dim (poor coalescing, 33% line
    # util), issue ONE [BLOCK_N, NP2_STATELEN] 2-D load. The state_len cols for
    # a given dim are contiguous in memory (stride_conv_state_tok=1), so the 2-D
    # load packs 64 dims * state_len cols into contiguous cachelines at 100%
    # util. col0/col1/col2 become register slices of cs_tile. L1 access count
    # drops 9 -> 3 per warp; coalescing improves; traffic spreads across banks.
    if KERNEL_WIDTH == 4:
        k_idx = tl.arange(0, NP2_STATELEN)  # [NP2_STATELEN]
        cs_ptrs_2d = (
            prior_tokens[:, None] + k_idx[None, :] * stride_conv_state_tok
        )  # [BLOCK_N, NP2_STATELEN]
        cs_mask_2d = mask_w[:, None] & (k_idx[None, :] < state_len)
        cs_tile = tl.load(cs_ptrs_2d, cs_mask_2d, other=0.0)  # [BLOCK_N, NP2_STATELEN]
        # Extract columns via masked-sum (proven-compiling pattern from R5;
        # tl.split / constexpr indexing of a 2-D tile do not lower on this
        # triton). Each is a select, not a real reduction.
        col0 = tl.sum(
            tl.where((k_idx == 0)[None, :], cs_tile, tl.zeros_like(cs_tile)), axis=1
        )
        col1 = tl.sum(
            tl.where((k_idx == 1)[None, :], cs_tile, tl.zeros_like(cs_tile)), axis=1
        )
        col2 = tl.sum(
            tl.where((k_idx == 2)[None, :], cs_tile, tl.zeros_like(cs_tile)), axis=1
        )
    else:
        if KERNEL_WIDTH >= 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH >= 3:
            conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH >= 4:
            conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: update conv_state (sliding-window shift-left by seqlen).
    # R15 fast path for the common decode/prefill-1 case (seqlen==1, no spec
    # decoding): write the shift-left result with `state_len` independent 1-D
    # tl.store's instead of materializing a [NP2_STATELEN, BLOCK_N] tile via a
    # nested tl.where broadcast + 2-D store. Numerically identical to R14
    # (row0<-col1, row1<-col2, row2<-x_tok0); the [4,256] live tile and its
    # nested where control flow are removed, cutting the dominant register-
    # spill -> GSM read path identified by PFC. Store byte count unchanged.
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    if not IS_SPEC_DECODING and seqlen == 1:
        x_tok0 = tl.load(x_base + 0 * stride_x_token, mask_w, 0.0)  # [BLOCK_N]
        # 1-D shift-left stores. Semantics: row i <- old col (i+1) for
        # i in [0, state_len-2]; row (state_len-1) <- x_tok0. Only cols
        # actually defined in STEP 1 are referenced. Rows >= state_len
        # (the NP2 padding slots) are simply not written, matching the
        # R14 (idx_tokens < state_len) row mask. Store byte count
        # unchanged (state_len * BLOCK_N).
        if KERNEL_WIDTH == 2:
            # state_len=1: [x_tok0]
            tl.store(conv_states_base + 0 * stride_conv_state_tok, x_tok0, mask_w)
        elif KERNEL_WIDTH == 3:
            # state_len=2: [col1, x_tok0]
            tl.store(conv_states_base + 0 * stride_conv_state_tok, col1, mask_w)
            tl.store(conv_states_base + 1 * stride_conv_state_tok, x_tok0, mask_w)
        elif KERNEL_WIDTH == 4:
            # state_len=3: [col1, col2, x_tok0]
            tl.store(conv_states_base + 0 * stride_conv_state_tok, col1, mask_w)
            tl.store(conv_states_base + 1 * stride_conv_state_tok, col2, mask_w)
            tl.store(conv_states_base + 2 * stride_conv_state_tok, x_tok0, mask_w)
        elif KERNEL_WIDTH == 5:
            # state_len=4: [col1, col2, col3, x_tok0]
            tl.store(conv_states_base + 0 * stride_conv_state_tok, col1, mask_w)
            tl.store(conv_states_base + 1 * stride_conv_state_tok, col2, mask_w)
            tl.store(conv_states_base + 2 * stride_conv_state_tok, col3, mask_w)
            tl.store(conv_states_base + 3 * stride_conv_state_tok, x_tok0, mask_w)
    else:
        # General path (seqlen>1 or spec decoding): original sliding-window merge.
        # The tl.debug_barrier was redundant (RAW dependency already orders the
        # tl.where), so it is dropped here too.
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_state_batch_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[None, :]
            + (
                (idx_tokens + (1 if IS_SPEC_DECODING else seqlen))
                * stride_conv_state_tok
            )[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)
        VAL = state_len - seqlen
        x_ptrs = (
            x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
        )
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)

        new_conv_state = tl.where(mask, conv_state, loaded_x)

        # R14: reuse conv_states_base (identical expr) instead of recomputing.
        conv_state_ptrs_target = (
            conv_states_base + (idx_tokens * stride_conv_state_tok)[:, None]
        )  # [BLOCK_M, BLOCK_N]
        mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        idx_tokens = tl.arange(0, NP2_SEQLEN)  # [BLOCK_M]
        # Update parent mapping for all tokens at once using vectorized operations
        mask_retrieve = idx_tokens < seqlen
        retrieve_next_token_base = (
            retrieve_next_token_ptr
            + (idx_seq * stride_retrieve_next_token_seq)
            + idx_tokens * stride_retrieve_next_token_token
        )
        retrieve_next_tokens = tl.load(retrieve_next_token_base, mask_retrieve)
        retrieve_next_sibling_base = (
            retrieve_next_sibling_ptr
            + (idx_seq * stride_retrieve_next_sibling_seq)
            + idx_tokens * stride_retrieve_next_sibling_token
        )
        retrieve_next_siblings = tl.load(retrieve_next_sibling_base, mask_retrieve)
        parent_idx_tokens = tl.zeros((NP2_SEQLEN,), dtype=tl.int32)

    # R14 pointer-hoist: reuse `mask_w` (defined once in STEP 1) instead of
    # recomputing `idx_feats < dim`; drop the `0 * stride_w_width` dead multiply
    # and the intermediate `w_ptrs` rebinds.
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    # R26: 2-D merge of weight cols. Instead of 4 separate 1-D tl.load's each
    # with stride_w_dim=4 (25% line util), issue ONE [BLOCK_N, KERNEL_WIDTH]
    # 2-D load. The KW cols for a given dim are contiguous (stride_w_width=1),
    # so 64 dims * 4 cols = 512B = 4 cachelines at 100% util. w_col0..3 become
    # register slices of w_tile. L1 access count drops 16 -> 4 per warp.
    if KERNEL_WIDTH == 4:
        w_jidx = tl.arange(0, KERNEL_WIDTH)  # [KW]
        w_ptrs_2d = w_base[:, None] + w_jidx[None, :] * stride_w_width  # [BLOCK_N, KW]
        w_mask_2d = mask_w[:, None]
        w_tile = tl.load(w_ptrs_2d, w_mask_2d, other=0.0)  # [BLOCK_N, KW]
        # Extract columns via masked-sum (proven-compiling pattern from R5).
        w_col0 = tl.sum(
            tl.where((w_jidx == 0)[None, :], w_tile, tl.zeros_like(w_tile)), axis=1
        )
        w_col1 = tl.sum(
            tl.where((w_jidx == 1)[None, :], w_tile, tl.zeros_like(w_tile)), axis=1
        )
        w_col2 = tl.sum(
            tl.where((w_jidx == 2)[None, :], w_tile, tl.zeros_like(w_tile)), axis=1
        )
        w_col3 = tl.sum(
            tl.where((w_jidx == 3)[None, :], w_tile, tl.zeros_like(w_tile)), axis=1
        )
    else:
        if KERNEL_WIDTH >= 2:
            w_col0 = tl.load(w_base, mask_w, other=0.0)
            w_col1 = tl.load(w_base + stride_w_width, mask_w, other=0.0)
        if KERNEL_WIDTH >= 3:
            w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
        if KERNEL_WIDTH >= 4:
            w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = mask_w  # R14: reuse mask_w (== idx_feats < dim)

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # set the parent index of the next token in the eagle tree
            # next token's parent is the current token
            retrieve_next_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_tokens, 0)
            )
            if retrieve_next_token_idx != -1:  # pad slot id
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_next_token_idx,
                    idx_token,
                    parent_idx_tokens,
                )
            # next token's parent is the parent of the current token
            retrieve_sibling_token_idx = tl.sum(
                tl.where(idx_tokens == idx_token, retrieve_next_siblings, 0)
            )
            if retrieve_sibling_token_idx != -1:  # pad slot id
                parent_idx_token = tl.sum(
                    tl.where(idx_tokens == idx_token, parent_idx_tokens, 0)
                )
                parent_idx_tokens = tl.where(
                    idx_tokens == retrieve_sibling_token_idx,
                    parent_idx_token,
                    parent_idx_tokens,
                )
            # tl.device_print("am", parent_idx_tokens)

            _idx_token = idx_token
            x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
            matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            # convolution operation: itself * wcol[-1] + parent * wcol[-2] + grand-parent * wcol[-3] + ...
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 0:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 3:
                    if j == 0:
                        matrix_w = w_col2
                    elif j == 1:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0
                elif KERNEL_WIDTH == 4:
                    if j == 0:
                        matrix_w = w_col3
                    elif j == 1:
                        matrix_w = w_col2
                    elif j == 2:
                        matrix_w = w_col1
                    else:
                        matrix_w = w_col0

                if SAVE_INTERMEDIATE:
                    # Save the window state after consuming this token
                    # Layout: [seq(cache line), step, dim, win(K-1)]
                    base_ptr = (
                        intermediate_conv_window_ptr
                        + intermediate_state_batch_coord * stride_inter_seq
                        + idx_token * stride_inter_step
                        + idx_feats * stride_inter_dim
                    )

                    # store itself in KERNEL_WIDTH-2 slot, parent in KERNEL_WIDTH-3 slot, grand-parent in KERNEL_WIDTH-4 slot, ...
                    if KERNEL_WIDTH - j - 2 >= 0:
                        tl.store(
                            base_ptr + (KERNEL_WIDTH - j - 2) * stride_inter_win,
                            matrix_x,
                            mask=mask_w,
                        )

                acc += matrix_x * matrix_w

                # move to parent for next iteration
                if _idx_token > 0:
                    _idx_token = tl.sum(
                        tl.where(idx_tokens == _idx_token, parent_idx_tokens, 0)
                    )
                    x_ptrs_1d = x_base_1d + _idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                else:
                    # no parent within the current chunk, load from prev conv state: col[-1] (idx 0's parent), col[-2] (idx 0's grand parent), ...
                    if KERNEL_WIDTH == 2:
                        if _idx_token == 0:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 3:
                        if _idx_token == 0:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    elif KERNEL_WIDTH == 4:
                        if _idx_token == 0:
                            matrix_x = col2
                        elif _idx_token == -1:
                            matrix_x = col1
                        else:
                            matrix_x = col0
                    _idx_token = _idx_token - 1
        else:
            matrix_w = w_col0
            matrix_x = col0

            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 2:
                    if j == 1:  # KERNEL_WIDTH-1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        # R16: reuse x_tok0 (loaded in STEP2) instead of
                        # re-loading x_base + idx_token*stride_x_token.
                        # seqlen==1 -> idx_token=0 -> same address as x_tok0.
                        matrix_x = x_tok0

                acc += matrix_x * matrix_w  # [BLOCK_N]

            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x

            if SAVE_INTERMEDIATE:
                # Save the window state after consuming this token
                # Layout: [seq(cache line), step, dim, win(K-1)]
                base_ptr = (
                    intermediate_conv_window_ptr
                    + intermediate_state_batch_coord * stride_inter_seq
                    + idx_token * stride_inter_step
                    + idx_feats * stride_inter_dim
                )
                if KERNEL_WIDTH >= 2:
                    tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
                if KERNEL_WIDTH >= 3:
                    tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
                if KERNEL_WIDTH >= 4:
                    tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        # fuse: store calculated retrieve_parent_token to tensor
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            tl.store(
                retrieve_parent_token_ptr
                + idx_seq * stride_retrieve_parent_token_seq
                + idx_tokens * stride_retrieve_parent_token_token,
                parent_idx_tokens,
                mask=mask_retrieve,
            )

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def _adapt_fwd_launch_args(_args, kwargs):
    kwargs["BLOCK_M"] = 256
    kwargs["BLOCK_N"] = 512


def _adapt_update_launch_args(_args, kwargs):
    kwargs["BLOCK_N"] = 512
    kwargs["num_stages"] = 1


def register_causal_conv1d_patch() -> None:
    """Bind SUPA kernels into upstream Mamba call sites.

    Lifecycle: plugin-register-immediate. Upstream public callers remain the
    source of API validation, buffer setup, and grid construction. Manifest:
    kernels.ops-supa-mamba-conv.
    """
    upstream = importlib.import_module("sglang.kernels.ops.mamba.causal_conv1d_triton")
    if getattr(upstream, "_supa_kernel_patched", False):
        return

    upstream._causal_conv1d_fwd_kernel = SupaApiWrapper(
        _causal_conv1d_fwd_supa_kernel,
        adapt_args=_adapt_fwd_launch_args,
    )
    upstream._causal_conv1d_update_kernel = SupaApiWrapper(
        _causal_conv1d_update_supa_kernel,
        adapt_args=_adapt_update_launch_args,
    )
    upstream._supa_kernel_patched = True
