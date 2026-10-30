"""Post-install setup for sglang-supa.

Installs usercustomize.py to site-packages so that subprocesses
spawned by SGLang can automatically activate SUPA hooks.
"""

import site
import shutil
from pathlib import Path


def install_usercustomize():
    src = Path(__file__).parent / "_usercustomize.py"
    dst = Path(site.getsitepackages()[0]) / "usercustomize.py"
    shutil.copy2(src, dst)
    print(f"Installed: {dst}")
