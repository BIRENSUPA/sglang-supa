#!/usr/bin/env bash

# -------------------------------------------------
# sglang_supa build and packaging script
# -------------------------------------------------
set -eo pipefail

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DIST_DIR="${CUR_DIR}/dist"
BUILD_DIR="${CUR_DIR}/build"
SGLANG_DIR="${CUR_DIR}/3rdparty/sglang"
SGLANG_PYTHON_DIR="${SGLANG_DIR}/python"

parallel_jobs="${ENV_CI_cpu_limit:-16}"
build_type="Release"
supa_arch="br200"
build_python="OFF"
build_kernel="OFF"
build_gateway="OFF"
build_gateway_rust="OFF"
setup_env="OFF"
clean_only="OFF"
no_install="OFF"

### Usage: bash build.sh [option]
### Options:
###   -h, --help                 Print help.
###   -c, --clean                Clean build outputs.
###   -a, --arch <arg>           SUPA GPU arch, default br200.
###   -b, --build-type <arg>     Build type: Debug, Release, RelWithDebInfo.
###   -j, --jobs <arg>           Parallel compile jobs.
###   --setup-env                Install upstream SGLang, sglang_supa, and sgl-kernel in editable mode.
###   --build-python             Build the sglang_supa Python wheel.
###   --build-kernel             Build the sgl-kernel wheel.
###   --build-gateway            Build model gateway Python wheel when sources are present.
###   --build-gateway-rust       Build model gateway Rust binary when sources are present.
###   --all                      Build all available wheel artifacts.
###   --no-install               Do not install wheels after building kernel/gateway.
###
### Default with no build selector: build both sglang_supa and sgl-kernel wheels.

function print_help() {
  sed -rn 's/^### ?//;T;p;' "$0"
}

function clean_outputs() {
  rm -rf "${BUILD_DIR}" "${DIST_DIR}" "${CUR_DIR}"/*.egg-info
  rm -rf "${CUR_DIR}"/sglang_supa-test "${CUR_DIR}"/sglang_supa-test.tar.gz
  rm -rf "${CUR_DIR}"/sglang_br-test "${CUR_DIR}"/sglang_br-test.tar.gz
}

function get_build_metadata() {
  local timestamp git_revision git_commit_count build_id
  timestamp=$(date +%Y%m%d%H%M%S)
  git_revision=$(git -C "${CUR_DIR}" rev-parse HEAD | cut -c -8)
  git_commit_count=$(git -C "${CUR_DIR}" log --oneline | wc -l)
  build_id="${BUILD_ID:-0}"
  echo "${git_commit_count}.${build_id}.${git_revision}.${timestamp}"
}

function patch_upstream_sglang_pyproject() {
  local pyproject="${SGLANG_PYTHON_DIR}/pyproject.toml"
  if [[ ! -f "${pyproject}" ]]; then
    echo "[WARN] Upstream SGLang pyproject not found: ${pyproject}"
    return 0
  fi

  python3 - "${pyproject}" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines(keepends=True)
out = []
skip = False
for line in lines:
    if 'setuptools-rust' in line:
        line = line.replace('"setuptools-rust>=1.10", ', '')
    if line.startswith('dependencies = ['):
        out.append('dependencies = []\n')
        skip = not line.rstrip().endswith(']')
        continue
    if skip:
        if line.startswith(']'):
            skip = False
        continue
    if line.startswith('[[tool.setuptools-rust'):
        skip = True
        continue
    out.append(line)
p.write_text(''.join(out))
PY
}

function install_upstream_sglang() {
  patch_upstream_sglang_pyproject
  if [[ ! -d "${SGLANG_PYTHON_DIR}" ]]; then
    echo "[ERROR] Upstream SGLang python package not found: ${SGLANG_PYTHON_DIR}"
    exit 1
  fi
  python3 -m pip install --no-deps -e "${SGLANG_PYTHON_DIR}"
}

function setup_development_env() {
  install_upstream_sglang
  python3 -m pip install -e "${CUR_DIR}[dev,all]"
}

function build_python_wheel() {
  install_upstream_sglang
  mkdir -p "${DIST_DIR}"
  if python3 -m build --version >/dev/null 2>&1; then
    python3 -m build "${CUR_DIR}" --wheel --outdir "${DIST_DIR}"
  else
    python3 -m pip install build wheel
    python3 -m build "${CUR_DIR}" --wheel --outdir "${DIST_DIR}"
  fi
}

function build_gateway_python_wheel() {
  local gateway_dir="${CUR_DIR}/sgl-model-gateway"
  if [[ -f "${gateway_dir}/Makefile" ]]; then
    (cd "${gateway_dir}" && make python-build)
    cp -af "${gateway_dir}"/target/wheels/*.whl "${DIST_DIR}"/ 2>/dev/null || true
  else
    echo "[WARN] sgl-model-gateway Makefile not found, skipping gateway Python build"
  fi
}

function build_gateway_rust_binary() {
  local gateway_dir="${CUR_DIR}/sgl-model-gateway"
  if [[ -f "${gateway_dir}/Makefile" ]]; then
    (cd "${gateway_dir}" && make build)
  else
    echo "[WARN] sgl-model-gateway Makefile not found, skipping gateway Rust build"
  fi
}

function package_tests() {
  local pkg_name="sglang_supa-test"
  local pkg_dir="${CUR_DIR}/${pkg_name}"
  rm -rf "${pkg_dir}"
  mkdir -p "${pkg_dir}"

  for path in requirements.txt requirements-no-deps.txt pytest.ini test benchmarks; do
    if [[ -e "${CUR_DIR}/${path}" ]]; then
      cp -af "${CUR_DIR}/${path}" "${pkg_dir}/"
    fi
  done

  if [[ -d "${CUR_DIR}/env_setup" ]]; then
    cp -af "${CUR_DIR}/env_setup" "${pkg_dir}/env_setup"
  fi

  tar -C "${CUR_DIR}" -zcf "${CUR_DIR}/${pkg_name}.tar.gz" "${pkg_name}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -c|--clean)
      clean_only="ON"
      shift
      ;;
    -i|--internal)
      for_internal=true
      shift
      ;;
    -e|--external)
      for_external=true
      shift
      ;;
    -t|--build-test)
      enable_build_test="true"
      test_level=$2
      shift 2
      ;;
    -a|--arch)
      supa_arch="$2"
      shift 2
      ;;
    -b|--build-type)
      build_type="$2"
      shift 2
      ;;
    --enable-coverage)
      enable_coverage=ON
      build_type=Debug
      shift
      ;;
    -j|--jobs)
      parallel_jobs="$2"
      shift 2
      ;;
    --setup-env)
      setup_env="ON"
      shift
      ;;
    --build-python)
      build_python="ON"
      shift
      ;;
    --build-kernel)
      build_kernel="ON"
      shift
      ;;
    --build-gateway)
      build_gateway="ON"
      shift
      ;;
    --build-gateway-rust)
      build_gateway_rust="ON"
      shift
      ;;
    --all)
      build_python="ON"
      build_kernel="ON"
      build_gateway="ON"
      build_gateway_rust="ON"
      shift
      ;;
    --no-install)
      no_install="ON"
      shift
      ;;
    *)
      echo "[ERROR] Invalid input param: $1"
      print_help
      exit 1
      ;;
  esac
done

export BUILD_METADATA="${BUILD_METADATA:-$(get_build_metadata)}"
export supa_arch
export DEBUG="0"
if [[ "${build_type}" == "Debug" ]]; then
  export DEBUG="1"
elif [[ "${build_type}" == "RelWithDebInfo" ]]; then
  export REL_WITH_DEB_INFO="1"
fi

if [[ "${clean_only}" == "ON" ]]; then
  clean_outputs
  exit 0
fi

if [[ "${setup_env}" == "ON" ]]; then
  setup_development_env
  exit 0
fi

if [[ "${build_python}" == "OFF" && "${build_kernel}" == "OFF" && "${build_gateway}" == "OFF" && "${build_gateway_rust}" == "OFF" ]]; then
  build_python="ON"
  build_kernel="ON"
fi

mkdir -p "${DIST_DIR}"

if [[ "${build_python}" == "ON" ]]; then
  build_python_wheel
fi
if [[ "${build_gateway}" == "ON" ]]; then
  build_gateway_python_wheel
fi
if [[ "${build_gateway_rust}" == "ON" ]]; then
  build_gateway_rust_binary
fi

package_tests
