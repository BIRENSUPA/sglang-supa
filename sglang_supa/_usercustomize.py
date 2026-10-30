"""Auto-activate SUPA hooks for Python processes.

This file is installed to site-packages as usercustomize.py via
``sglang-supa-init``. When ``SGLANG_AUTO_LOAD_PLUGINS=1`` is set by a SUPA
bootstrap entrypoint or the caller, it loads the same hooks in each process.
"""

import os

if os.environ.get("SGLANG_AUTO_LOAD_PLUGINS") == "1":
    from sglang.srt.plugins import load_plugins

    load_plugins()
