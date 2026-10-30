from __future__ import annotations

import os
import pathlib
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

    from sglang.srt.plugins import load_plugins

    load_plugins()

    import argparse
    import logging

    from sglang.bench_one_batch import BenchArgs, main as bench_main
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        bench_main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
