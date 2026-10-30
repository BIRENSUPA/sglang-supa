"""Load SUPA hooks at interpreter startup for bootstrapped process trees.

SUPA entrypoints add this directory to ``PYTHONPATH`` and set
``SGLANG_AUTO_LOAD_PLUGINS=1`` before spawning child processes.
"""

from __future__ import annotations

import os

if os.environ.get("SGLANG_AUTO_LOAD_PLUGINS") == "1":
    from sglang.srt.plugins import load_plugins

    load_plugins()
