def register_server_args_patch() -> None:
    """Add the SUPA FP8 GEMM backend to the upstream server argument choices."""
    from sglang.srt import server_args as upstream

    if "supa" not in upstream.FP8_GEMM_RUNNER_BACKEND_CHOICES:
        upstream.FP8_GEMM_RUNNER_BACKEND_CHOICES.append("supa")
