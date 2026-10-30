#!/usr/bin/env python3
"""Check that the kernels/ops adaptation matrix covers an upstream diff exactly."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "docs/hooks/kernels_ops_adaptation_matrix.yaml"


def _upstream_diff(repo: Path, parent: str, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "diff", "--unified=0", parent, commit, "--", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _diff_inventory(diff: str) -> dict[str, int]:
    inventory: dict[str, int] = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current = line.split()[3][2:]
            inventory[current] = 0
        elif line.startswith("@@") and current is not None:
            inventory[current] += 1
    return inventory


def validate_matrix(matrix_path: Path = DEFAULT_MATRIX) -> list[str]:
    data: dict[str, Any] = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    repo = Path(data["upstream_repo"])
    commit = data["upstream_commit"]
    parent = data.get("upstream_parent", f"{commit}^")
    path = data["upstream_path"]
    actual = _diff_inventory(_upstream_diff(repo, parent, commit, path))

    entries = data.get("entries", [])
    expected: dict[str, int] = {}
    for entry in entries:
        file = f"{path}/{entry['file']}"
        if file in expected:
            errors.append(f"duplicate matrix entry: {file}")
        expected[file] = int(entry["upstream_hunks"])
        if entry.get("status") not in {
            "settings_patch",
            "pending_plugin_impl",
            "pending_integration",
            "covered",
            "intentionally_dropped",
        }:
            errors.append(f"{file}: invalid status {entry.get('status')!r}")

    for file in sorted(set(actual) - set(expected)):
        errors.append(f"missing matrix entry: {file}")
    for file in sorted(set(expected) - set(actual)):
        errors.append(f"matrix entry is not changed by upstream diff: {file}")
    for file in sorted(set(actual) & set(expected)):
        if actual[file] != expected[file]:
            errors.append(
                f"{file}: matrix says {expected[file]} hunks, upstream has {actual[file]}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    errors = validate_matrix(args.matrix)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validated {args.matrix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
