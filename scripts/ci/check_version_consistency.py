#!/usr/bin/env python3
"""CI version consistency check.

Verifies that three sources of truth are in sync:
  1. pyproject.toml [tool.sglang-supa] supa-major/minor/patch
  2. 3rdparty/sglang submodule current tag
  3. sglang_supa/version.py __version__

Exit code 0 = consistent. Exit code 1 = mismatch (prints details).

Usage:
  python scripts/ci/check_version_consistency.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _get_submodule_tag() -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=ROOT / "3rdparty" / "sglang",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            cwd=ROOT / "3rdparty" / "sglang",
            capture_output=True,
            text=True,
        )
    return result.stdout.strip().lstrip("v")


def _normalize(version: str) -> str:
    version = version.strip().lstrip("v")
    match = re.match(r"^(\d+\.\d+\.\d+(?:\.post\d+)?)", version)
    return match.group(1) if match else version


def _read_supa_fields() -> tuple:
    content = (ROOT / "pyproject.toml").read_text()
    section_match = re.search(r"\[tool\.sglang-supa\](.*?)(?=\[|\Z)", content, re.DOTALL)
    if not section_match:
        return None, None, None
    section = section_match.group(1)

    def _get_int(key):
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", section, re.MULTILINE)
        return int(m.group(1)) if m else None

    return _get_int("supa-major"), _get_int("supa-minor"), _get_int("supa-patch")


def _read_version_py() -> str:
    content = (ROOT / "sglang_supa" / "version.py").read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else ""


def main() -> int:
    errors = []

    sglang_tag = _get_submodule_tag()
    supa_major, supa_minor, supa_patch = _read_supa_fields()
    version_py = _read_version_py()

    print(f"submodule tag  : {sglang_tag}")
    print(f"pyproject supa : {supa_major}.{supa_minor}.{supa_patch}")
    print(f"version.py     : {version_py}")

    if supa_major is None:
        errors.append("FAIL: [tool.sglang-supa] section not found in pyproject.toml")
    else:
        expected = f"{sglang_tag}+sglang-supa_v{supa_major}.{supa_minor}.{supa_patch}"
        if version_py != expected:
            errors.append(
                f"FAIL: version.py mismatch\n"
                f"  expected : {expected}\n"
                f"  actual   : {version_py}\n"
                f"  fix      : run 'python3 setup.py --version' to regenerate version.py"
            )

        sglang_in_version = version_py.split("+")[0] if "+" in version_py else ""
        if sglang_in_version and _normalize(sglang_in_version) != _normalize(sglang_tag):
            errors.append(
                f"FAIL: sglang version in version.py ({sglang_in_version}) "
                f"does not match submodule tag ({sglang_tag})"
            )

    if errors:
        print("\n" + "\n".join(errors))
        return 1

    print("\nOK: all version sources are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
