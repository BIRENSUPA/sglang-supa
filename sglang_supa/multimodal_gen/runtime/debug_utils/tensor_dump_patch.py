from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def _ensure_server_args_fields(ServerArgs) -> None:
    if not hasattr(ServerArgs, "debug_tensor_dump_output_folder"):
        ServerArgs.debug_tensor_dump_output_folder = None
    if not hasattr(ServerArgs, "debug_tensor_dump_modules"):
        ServerArgs.debug_tensor_dump_modules = None


def _patch_server_args() -> None:
    import sglang.multimodal_gen.runtime.server_args as server_args_module

    if getattr(server_args_module, "_supa_tensor_dump_server_args_patched", False):
        return

    ServerArgs = server_args_module.ServerArgs
    _ensure_server_args_fields(ServerArgs)

    original_adjust_warmup = ServerArgs._adjust_warmup
    original_add_cli_args = ServerArgs.add_cli_args
    original_from_dict = ServerArgs.from_dict
    original_from_kwargs = ServerArgs.from_kwargs

    def _adjust_warmup(self):
        original_adjust_warmup(self)
        if (
            getattr(self, "debug_tensor_dump_output_folder", None) is not None
            and self.warmup_mode != "off"
        ):
            logger.warning("Warmup is disabled because of using tensor dump mode")
            self.warmup_mode = "off"
            self.warmup_resolutions = None

    def add_cli_args(parser):
        result = original_add_cli_args(parser)
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping diffusion tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-modules",
            type=str,
            nargs="+",
            default=ServerArgs.debug_tensor_dump_modules,
            help="The pipeline module names to dump. Dumps common model modules if not specified.",
        )
        return result

    @classmethod
    def from_dict(cls, kwargs):
        server_args = original_from_dict(kwargs)
        if "debug_tensor_dump_output_folder" in kwargs:
            server_args.debug_tensor_dump_output_folder = kwargs[
                "debug_tensor_dump_output_folder"
            ]
        if "debug_tensor_dump_modules" in kwargs:
            server_args.debug_tensor_dump_modules = kwargs["debug_tensor_dump_modules"]
        server_args._adjust_warmup()
        return server_args

    @classmethod
    def from_kwargs(cls, **kwargs):
        tensor_dump_output_folder = kwargs.pop("debug_tensor_dump_output_folder", None)
        tensor_dump_modules = kwargs.pop("debug_tensor_dump_modules", None)
        server_args = original_from_kwargs(**kwargs)
        server_args.debug_tensor_dump_output_folder = tensor_dump_output_folder
        server_args.debug_tensor_dump_modules = tensor_dump_modules
        server_args._adjust_warmup()
        return server_args

    ServerArgs._adjust_warmup = _adjust_warmup
    ServerArgs.add_cli_args = staticmethod(add_cli_args)
    ServerArgs.from_dict = classmethod(from_dict.__func__)
    ServerArgs.from_kwargs = classmethod(from_kwargs.__func__)
    server_args_module._supa_tensor_dump_server_args_patched = True


def _execute_with_tensor_dump(worker, forward_fn: Callable):
    tensor_dumper = getattr(worker, "tensor_dumper", None)
    if tensor_dumper is None:
        return forward_fn()

    tensor_dumper.reset_current_tensors()
    try:
        result = forward_fn()
    finally:
        tensor_dumper.dump_current_tensors()
    return result


def _maybe_register_tensor_dump_hooks(worker) -> None:
    if getattr(worker, "tensor_dumper", None) is not None:
        return

    dump_folder = getattr(worker.server_args, "debug_tensor_dump_output_folder", None)
    if dump_folder is None:
        return

    from sglang_supa.multimodal_gen.runtime.debug_utils.tensor_dump_forward_hook import (
        register_forward_hook_for_pipeline,
    )

    worker.tensor_dumper = register_forward_hook_for_pipeline(
        worker.pipeline,
        dump_folder,
        getattr(worker.server_args, "debug_tensor_dump_modules", None),
        worker.rank,
        worker.local_rank,
    )


def _patch_gpu_worker() -> None:
    import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module

    if getattr(gpu_worker_module, "_supa_tensor_dump_gpu_worker_patched", False):
        return

    GPUWorker = gpu_worker_module.GPUWorker
    original_init = GPUWorker.__init__
    original_init_device_and_model = GPUWorker.init_device_and_model
    original_execute_forward_common = GPUWorker._execute_forward_common

    def __init__(self, *args, **kwargs):
        self.tensor_dumper = None
        original_init(self, *args, **kwargs)

    def init_device_and_model(self):
        original_init_device_and_model(self)
        _maybe_register_tensor_dump_hooks(self)

    def _execute_forward_common(self, req, *, forward_fn, **kwargs):
        def wrapped_forward_fn():
            return _execute_with_tensor_dump(self, forward_fn)

        return original_execute_forward_common(
            self,
            req,
            forward_fn=wrapped_forward_fn,
            **kwargs,
        )

    GPUWorker.__init__ = __init__
    GPUWorker.init_device_and_model = init_device_and_model
    GPUWorker._execute_forward_common = _execute_forward_common
    gpu_worker_module._supa_tensor_dump_gpu_worker_patched = True


def register_multimodal_tensor_dump_hooks() -> None:
    _patch_server_args()
    _patch_gpu_worker()
