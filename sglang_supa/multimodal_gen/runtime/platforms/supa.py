from __future__ import annotations

import torch
from functools import lru_cache

from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    Platform,
    PlatformEnum,
)


class SupaPlatform(Platform):
    _enum = PlatformEnum.SUPA
    device_name: str = "supa"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        import sglang.multimodal_gen.envs as envs

        return torch.device(f"cuda:{envs.LOCAL_RANK}")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_name(device_id))

    @classmethod
    @lru_cache(maxsize=1)
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        if enforce_eager:
            return False
        return True

    @classmethod
    def log_warnings(cls) -> None:
        pass

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        return False

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return float(torch.cuda.max_memory_allocated(device))

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group=None,
    ) -> float:
        if empty_cache:
            torch.cuda.empty_cache()

        free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)
        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())
        return free_gpu_memory / (1 << 30)

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        if selected_backend == AttentionBackendEnum.TORCH_SDPA:
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        if selected_backend == AttentionBackendEnum.SUPA:
            return "sglang_supa.multimodal_gen.runtime.layers.attention.backends.supa_attn.SupaAttentionBackend"
        if selected_backend in (AttentionBackendEnum.FA, None):
            pass
        elif selected_backend:
            raise ValueError(f"Invalid attention backend for {cls.device_name}: {selected_backend}")

        target_backend = AttentionBackendEnum.FA
        if dtype not in (torch.float16, torch.bfloat16):
            target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.FA:
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (  # noqa: F401
                    FlashAttentionBackend,
                )

                supported_sizes = FlashAttentionBackend.get_supported_head_sizes()
                if head_size not in supported_sizes:
                    target_backend = AttentionBackendEnum.TORCH_SDPA
            except ImportError:
                target_backend = AttentionBackendEnum.TORCH_SDPA

        if target_backend == AttentionBackendEnum.TORCH_SDPA:
            return "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
        return "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.FlashAttentionBackend"

    @classmethod
    def get_torch_distributed_backend_str(cls) -> str:
        return "bccl"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.cuda_communicator.CudaCommunicator"
