from __future__ import annotations

import importlib.util
import os

SUPA_AVAILABLE_ENV = "SUPA_AVAILABLE"


def detect_supa_hardware() -> bool:
    """Lightweight probe used by the platform activator.

    Replace this with your real vendor-runtime detection logic.
    """
    env_value = os.getenv(SUPA_AVAILABLE_ENV)
    if env_value is not None:
        return env_value.lower() in {"1", "true", "yes", "on"}

    return importlib.util.find_spec("supa_runtime") is not None


def activate_supa() -> str | None:
    if not detect_supa_hardware():
        return None
    return "sglang_supa.platform.SupaPlatform"
