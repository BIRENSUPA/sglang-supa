from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)


class SupaDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "supa"
    device_type = "cuda"

    def is_cuda_alike(self) -> bool:
        return True

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.supa.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        return float(torch.supa.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("supa", local_rank)

    def set_device(self, device: torch.device) -> None:
        torch.supa.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.supa.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.supa.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.supa.get_device_capability(device_id)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        torch.supa.empty_cache()

    def synchronize(self) -> None:
        torch.supa.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        free, total = torch.supa.mem_get_info(device_id)
        return int(free), int(total)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    def get_torch_distributed_backend_str(self) -> str:
        return "bccl"

    def get_torch_profiler_activity_str(self) -> str:
        return "SUPA"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.SUPA
