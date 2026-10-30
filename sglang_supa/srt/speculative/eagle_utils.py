from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyInput


def eagle_sample_around(
    original_fn,
    verify_input: EagleVerifyInput,
    batch: ScheduleBatch,
    logits_output: LogitsProcessorOutput,
    grammar_mask: Optional[GrammarMask] = None,
):
    """Reuse the upstream NPU sampling branch for SUPA."""
    import sglang.srt.speculative.eagle_utils as upstream

    original_is_npu = upstream._is_npu
    upstream._is_npu = True
    try:
        return original_fn(verify_input, batch, logits_output, grammar_mask)
    finally:
        upstream._is_npu = original_is_npu


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    """Run greedy tree verification with the upstream Triton implementation."""
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_triton

    verify_tree_greedy_triton(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_predict=target_predict,
    )
    return predicts, accept_index, accept_token_num


def register_eagle_utils_hooks() -> None:
    HookRegistry.register(
        "sglang.srt.speculative.eagle_utils.eagle_sample",
        eagle_sample_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_utils.verify_tree_greedy_func",
        verify_tree_greedy_func,
        HookType.REPLACE,
    )
