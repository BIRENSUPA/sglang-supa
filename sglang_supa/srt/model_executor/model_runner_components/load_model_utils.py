"""SUPA model-loading utility patches."""

UNBALANCED_MODEL_LOADING_TIMEOUT_S = 1200


def register_load_model_utils_patch() -> None:
    """Extend the upstream post-load distributed barrier timeout for SUPA."""
    import sglang.srt.model_executor.model_runner_components.load_model_utils as upstream

    upstream.UNBALANCED_MODEL_LOADING_TIMEOUT_S = UNBALANCED_MODEL_LOADING_TIMEOUT_S
