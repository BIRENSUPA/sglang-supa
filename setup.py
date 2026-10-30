import contextlib
import os
import re
import shutil
import site
import sys
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

ROOT = Path(__file__).resolve().parent
TARGET_VERSION_FILE = ROOT / "sglang_supa" / "version.py"


def _read_project_version(pyproject: Path) -> str:
    for line in pyproject.read_text().splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return os.getenv("SGLANG_SUPA_BASE_VERSION", "0.1.0")


def _normalize_version(version: str) -> str:
    version = version.strip().removeprefix("v")
    match = re.match(r"^(\d+\.\d+\.\d+(?:\.post\d+)?)", version)
    if match:
        return match.group(1)
    return "0.0.0"


def _get_sglang_version() -> str:
    generated_version = (
        ROOT / "3rdparty" / "sglang" / "python" / "sglang" / "_version.py"
    )
    if generated_version.exists():
        match = re.search(
            r"__version__\s*=\s*version\s*=\s*['\"]([^'\"]+)['\"]",
            generated_version.read_text(),
        )
        if match:
            return _normalize_version(match.group(1))

    version_file = ROOT / "3rdparty" / "sglang" / "python" / "sglang" / "version.py"
    if version_file.exists():
        sys.path.insert(0, str(ROOT / "3rdparty" / "sglang" / "python"))
        try:
            from sglang.version import __version__

            return _normalize_version(__version__)
        except Exception:
            pass
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(ROOT / "3rdparty" / "sglang" / "python"))
    try:
        import sglang

        return _normalize_version(sglang.__version__)
    except Exception:
        return "0.0.0"


def _version_as_number(version: str) -> int:
    base = re.sub(r"\.post\d+$", "", _normalize_version(version))
    major, minor, patch = [int(v) for v in base.split(".")]
    return major * 10000 + minor * 100 + patch


def _read_supa_version(pyproject: Path) -> tuple:
    """Read supa-major/minor/patch from [tool.sglang-supa] in pyproject.toml."""
    content = pyproject.read_text()
    section_match = re.search(
        r"\[tool\.sglang-supa\](.*?)(?=\[|\Z)", content, re.DOTALL
    )
    if not section_match:
        return (0, 1, 0)
    section = section_match.group(1)

    def _get_int(key: str) -> int:
        m = re.search(rf"^\s*{key}\s*=\s*(\d+)", section, re.MULTILINE)
        return int(m.group(1)) if m else 0

    return _get_int("supa-major"), _get_int("supa-minor"), _get_int("supa-patch")


def _get_arch_label() -> str:
    silicon_n_lable_dict = {
        "arch_20": "br20x",
        "arch_21": "br21x",
        "arch_20,arch_21": "br2xx",
        "arch_20,arch_21,arch_22": "br2xx",
        # Product-name aliases used by build.sh (-a / default supa_arch).
        "br200": "br20x",
        "br210": "br21x",
        "br299": "br2xx",
    }
    build_param = os.getenv("supa_arch", default="arch_20,arch_21,arch_22")
    local_version_label = silicon_n_lable_dict[build_param]
    return f"{local_version_label}"


def _get_full_version() -> str:
    supa_major, supa_minor, supa_patch = _read_supa_version(ROOT / "pyproject.toml")
    sglang_version = _get_sglang_version()
    arch_label = _get_arch_label()
    TARGET_VERSION_FILE.write_text(
        f'__version__ = "{sglang_version}+sglang-supa_v{supa_major}.{supa_minor}.{supa_patch}"\n',
        encoding="utf-8",
    )
    return f"{sglang_version}+supa{supa_major}.{supa_minor}.{supa_patch}.{arch_label}"


def _install_usercustomize():
    """Copy _usercustomize.py to site-packages as usercustomize.py."""
    src = ROOT / "sglang_supa" / "_usercustomize.py"
    if not src.exists():
        return
    dst_dir = site.getsitepackages()[0]
    dst = Path(dst_dir) / "usercustomize.py"
    shutil.copy2(src, dst)
    print(f"Installed usercustomize.py -> {dst}")


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        _install_usercustomize()


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        _install_usercustomize()


if __name__ == "__main__":
    setup(
        name="sglang_supa",
        version=_get_full_version(),
        packages=find_packages(include=["sglang_supa", "sglang_supa.*"]),
        include_package_data=True,
        package_data={
            "sglang_supa": [
                "kernels/jit/include/**/*",
                "kernels/jit/csrc/**/*",
                "kernels/jit_overlay/include/**/*",
            ],
        },
        cmdclass={
            "install": PostInstallCommand,
            "develop": PostDevelopCommand,
        },
    )
