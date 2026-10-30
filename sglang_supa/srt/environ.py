def register_envs_patch():
    from sglang.srt.environ import EnvBool, Envs

    # Fuse the 4 separate prepare_moe_input kernels (count + offsets + argsort)
    # into a single kernel launch with inter-block atomic sync.
    setattr(Envs, "SGLANG_OPT_FUSED_MOE_PREPARE", EnvBool(False))

    # SUPA post_reorder_deepgemm Triton kernel: fall back to fp32 accumulation
    # (default False = bf16 accumulator, lower register pressure). Set to 1 if a
    # dataset hits precision issues with bf16 accumulation.
    setattr(Envs, "SGLANG_SUPA_POST_REORDER_FP32_ACCUM", EnvBool(False))
