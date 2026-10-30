# -*- coding: utf-8 -*-
# Copyright©2020-2026 Shanghai Biren Technology Co., Ltd. All rights reserved.

"""Detect the SUPA runtime environment.

A lightweight, upgrade-stable smoke test that verifies the torch backend
(SUPA on Biren systems, CUDA as a fallback) is usable. It deliberately
avoids any sglang-internal API so that an upstream sglang upgrade cannot
break environment detection at collection time.

Run:
    pytest test/test_supa_env.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch

# Side-effect import: registers the SUPA PrivateUse1 backend on Biren systems.
try:
    import torch_br  # noqa: F401

    _ = torch_br
except ImportError:
    pass


# ----------------------------------------------------------- backend abstraction
@dataclass(frozen=True)
class Backend:
    name: str
    mod: Any

    def is_available(self) -> bool:
        return bool(self.mod.is_available())

    def device_str(self, device_id: int) -> str:
        return f"{self.name}:{device_id}"

    def current_device(self) -> int:
        return int(self.mod.current_device())

    def device_count(self) -> int:
        return int(self.mod.device_count())


def _pick_backend() -> Optional[Backend]:
    """Prefer SUPA when available, fall back to CUDA."""
    if getattr(torch, "supa", None) is not None and torch.supa.is_available():
        return Backend(name="supa", mod=torch.supa)
    if torch.cuda.is_available():
        return Backend(name="cuda", mod=torch.cuda)
    return None


# ----------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def backend() -> Backend:
    b = _pick_backend()
    if b is None:
        pytest.skip("neither torch.supa nor torch.cuda is available")
    return b


# ----------------------------------------------------------- tests
@pytest.mark.full
@pytest.mark.regression
@pytest.mark.stress
@pytest.mark.sanity
@pytest.mark.ci_mini
class TestSupaEnv:
    """Verify the SUPA/CUDA backend is correctly initialized."""

    def test_backend_available(self, backend: Backend):
        """At least one compute backend is available and reports a device."""
        assert backend.is_available(), f"torch.{backend.name} reported unavailable"
        assert backend.device_count() >= 1, (
            f"torch.{backend.name}.device_count() = {backend.device_count()}, expected >= 1"
        )

    def test_current_device(self, backend: Backend):
        """current_device() returns a valid device index."""
        dev = backend.current_device()
        assert 0 <= dev < backend.device_count(), (
            f"current_device()={dev} out of range [0, {backend.device_count()})"
        )

    def test_tensor_alloc_and_copy(self, backend: Backend):
        """A tensor can be allocated on the backend and moved to CPU.

        This exercises the basic dispatch path (memory allocation + DtoH copy)
        without depending on any kernel-launch details.
        """
        device_id = backend.current_device()
        x = torch.randn(8, 8, device=backend.device_str(device_id))
        assert x.device.type == backend.name or x.device.type == "cuda", (
            f"tensor device type={x.device.type}, expected {backend.name} (or cuda alias)"
        )

        y = x.cpu()
        assert y.device.type == "cpu"
        assert y.shape == (8, 8)

    def test_device_name_nonempty(self, backend: Backend):
        """get_device_name returns a non-empty identifier."""
        device_id = backend.current_device()
        name = backend.mod.get_device_name(device_id)
        assert isinstance(name, str) and name, (
            f"get_device_name({device_id}) returned empty/non-str: {name!r}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
