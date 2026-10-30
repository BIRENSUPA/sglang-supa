from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_TENSOR_DUMP_MODULES = (
    "transformer",
    "transformer_2",
    "video_dit",
    "video_dit_2",
    "audio_dit",
    "vae",
    "video_vae",
    "audio_vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "image_encoder",
    "vision_language_encoder",
    "dual_tower_bridge",
    "diffusers_pipeline",
)


class DiffusionTensorDumper:
    def __init__(
        self,
        dump_dir: str,
        module_names: list[str] | None,
        rank: int,
        local_rank: int,
    ):
        self._module_names = module_names
        self._forward_pass_id = 0
        self._pid = os.getpid()
        self._current_tensors: dict[str, Any] = {}
        self._handles: list[Any] = []
        self._base_dir = Path(dump_dir)
        self._process_dir = (
            self._base_dir / f"Rank{rank}_LocalRank{local_rank}_pid{self._pid}"
        )
        self._process_dir.mkdir(parents=True, exist_ok=True)

    def get_dump_dir(self) -> str:
        return str(self._process_dir)

    def reset_current_tensors(self) -> None:
        self._current_tensors = {}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def add_tensor(self, name: str, value: Any) -> None:
        self._add_tensor_recursive(name, value)

    def dump_current_tensors(self) -> None:
        if len(self._current_tensors) == 0:
            return
        tensor_file_for_pass = self._process_dir / f"Pass{self._forward_pass_id:05d}.pt"
        logger.info("Dump %05dth pass to %s", self._forward_pass_id, tensor_file_for_pass)
        torch.save(self._current_tensors, str(tensor_file_for_pass))
        self.reset_current_tensors()
        self._forward_pass_id += 1

    def register_pipeline(self, pipeline: Any) -> None:
        selected_modules = self._select_modules(pipeline)
        if len(selected_modules) == 0:
            requested = self._module_names or list(DEFAULT_TENSOR_DUMP_MODULES)
            raise ValueError(
                "No torch.nn.Module instances found for tensor dump modules: "
                + ", ".join(requested)
            )

        seen_module_ids: set[int] = set()
        for component_name, module in selected_modules.items():
            for module_name, submodule in self._iter_hook_modules(component_name, module):
                module_id = id(submodule)
                if module_id in seen_module_ids:
                    continue
                seen_module_ids.add(module_id)
                self._handles.append(
                    submodule.register_forward_hook(self._dump_hook(module_name))
                )
        logger.info(
            "Registered diffusion tensor dump hooks for modules: %s. Output folder: %s",
            list(selected_modules.keys()),
            self.get_dump_dir(),
        )

    def _select_modules(self, pipeline: Any) -> dict[str, nn.Module]:
        requested = self._module_names or list(DEFAULT_TENSOR_DUMP_MODULES)
        selected: dict[str, nn.Module] = {}
        for name in requested:
            module = self._get_pipeline_module(pipeline, name)
            if isinstance(module, nn.Module):
                selected[name] = module
            elif self._module_names is not None:
                logger.warning(
                    "Skip tensor dump module %s because it is not a torch.nn.Module: %s",
                    name,
                    type(module),
                )
        return selected

    @staticmethod
    def _get_pipeline_module(pipeline: Any, name: str) -> Any:
        if hasattr(pipeline, "get_module"):
            return pipeline.get_module(name, None)
        return getattr(pipeline, "modules", {}).get(name, None)

    @staticmethod
    def _iter_hook_modules(component_name: str, module: nn.Module):
        children = list(module.named_children())
        if len(children) == 0:
            yield component_name, module
            return

        for name, submodule in module.named_modules():
            if name == "":
                continue
            if len(list(submodule.children())) == 0:
                yield f"{component_name}.{name}", submodule

    def _dump_hook(self, tensor_name: str):
        def inner_dump_hook(module, input, output):
            if output is not None:
                self.add_tensor(tensor_name, output)

        return inner_dump_hook

    def _add_tensor_recursive(self, name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self._current_tensors[name] = value.detach().cpu()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                self._add_tensor_recursive(f"{name}.{i}", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                self._add_tensor_recursive(f"{name}.{key}", item)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                self._add_tensor_recursive(f"{name}.{field.name}", getattr(value, field.name))


def register_forward_hook_for_pipeline(
    pipeline: Any,
    dump_dir: str,
    module_names: list[str] | None,
    rank: int,
    local_rank: int,
) -> DiffusionTensorDumper:
    tensor_dumper = DiffusionTensorDumper(
        dump_dir=dump_dir,
        module_names=module_names,
        rank=rank,
        local_rank=local_rank,
    )
    tensor_dumper.register_pipeline(pipeline)
    return tensor_dumper
