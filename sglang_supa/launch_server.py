from __future__ import annotations

import os
import pathlib
import runpy
import sys


def _enable_spawn_bootstrap() -> None:
    bootstrap_path = pathlib.Path(__file__).with_name("bootstrap")
    pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    bootstrap_str = str(bootstrap_path)

    if bootstrap_str not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [bootstrap_str, *[entry for entry in pythonpath_entries if entry]]
        )
    if bootstrap_str not in sys.path:
        sys.path.insert(0, bootstrap_str)

    os.environ["SGLANG_AUTO_LOAD_PLUGINS"] = "1"


def main() -> None:
    _enable_spawn_bootstrap()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
