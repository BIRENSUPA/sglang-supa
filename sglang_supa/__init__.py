from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang_supa.entrypoints import activate_supa
    from sglang_supa.platform import SupaPlatform

__all__ = ["activate_supa", "SupaPlatform"]


def __getattr__(name: str) -> Any:
    if name == "activate_supa":
        from sglang_supa.entrypoints import activate_supa

        return activate_supa
    if name == "SupaPlatform":
        from sglang_supa.platform import SupaPlatform

        return SupaPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
