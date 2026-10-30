from __future__ import annotations


def _load_plugins() -> None:
    from sglang.srt.plugins import load_plugins

    load_plugins()


def main() -> None:
    _load_plugins()

    from sglang.cli.main import main as sglang_main

    sglang_main()
