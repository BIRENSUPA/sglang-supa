from __future__ import annotations

from sglang_supa.srt.platforms.supa import SupaDeviceMixin

from sglang.srt.platforms.interface import SRTPlatform


class SupaPlatform(SupaDeviceMixin, SRTPlatform):
    def get_default_attention_backend(self) -> str:
        return "fa3"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

        return DecodeCudaGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    def get_dsa_kv_pool_cls(self) -> type:
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return DSATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator

    def get_piecewise_backend_cls(self) -> type:
        from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend

        return CUDAPiecewiseBackend

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True

    def supports_fp8(self) -> bool:
        return True

    def get_dispatch_key_name(self) -> str:
        # SUPA is CUDA-like and reuses BaseFusedOp.forward_cuda implementations.
        return "cuda"
