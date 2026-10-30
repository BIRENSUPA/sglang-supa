#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_CMD_DEFAULT='bash ./build.sh -a arch_20,arch_21,arch_22 -j 8 -b Debug --enable-coverage -t regression -i'
BUILD_CMD="${BUILD_CMD:-${BUILD_CMD_DEFAULT}}"
RUN_CMD="${RUN_CMD:-}"
PYTHON_COV_CMD="${PYTHON_COV_CMD:-}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CLEAN_COVERAGE="${CLEAN_COVERAGE:-1}"
MODE="${MODE:-python}"
HTML_DIR="${HTML_DIR:-html}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/qemu_coverage_validate.sh [options]

Options:
  --repo-dir DIR          Repository root. Default: parent of this script.
  --build-cmd CMD         C/C++ coverage build command.
                          Default: bash ./build.sh -a arch_20,arch_21,arch_22 -j 8 -b Debug --enable-coverage -t regression -i
  --run-cmd CMD           Command that runs tests/examples after build to generate .gcda files.
  --python-cov-cmd CMD    Python coverage command, for example: python3 -m pytest -sv test/test_cast.py
  --mode python           Coverage mode. Default: python. Python mode requires .coveragerc.
  --skip-build            Do not run build command.
  --no-install-deps       Do not pip install gcovr/coverage.
  --no-clean              Keep old .gcda/.gcno/coverage outputs before running.
  -h, --help              Print this help.

Environment variables with the same names can also be used:
  REPO_DIR BUILD_CMD RUN_CMD PYTHON_COV_CMD MODE SKIP_BUILD INSTALL_DEPS CLEAN_COVERAGE HTML_DIR

Examples:
  # Python coverage.
  bash scripts/qemu_coverage_validate.sh --mode python \
    --python-cov-cmd 'python3 -m pytest -sv test/test_cast.py'
EOF
}

log() { echo "[coverage-qemu] $*"; }
warn() { echo "[coverage-qemu][WARN] $*" >&2; }
fail() { echo "[coverage-qemu][ERROR] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"; shift 2 ;;
    --build-cmd)
      BUILD_CMD="$2"; shift 2 ;;
    --run-cmd)
      RUN_CMD="$2"; shift 2 ;;
    --python-cov-cmd)
      PYTHON_COV_CMD="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --skip-build)
      SKIP_BUILD=1; shift ;;
    --no-install-deps)
      INSTALL_DEPS=0; shift ;;
    --no-clean)
      CLEAN_COVERAGE=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      fail "Unknown argument: $1" ;;
  esac
done

[[ "${MODE}" == "python" ]] || fail "--mode must be python"
[[ -d "${REPO_DIR}" ]] || fail "repo dir not found: ${REPO_DIR}"
cd "${REPO_DIR}"

log "repo: ${REPO_DIR}"
log "mode: ${MODE}"

check_supa_env() {
  log "checking SUPA Python packages"
  python3 - <<'PY_CHECK_SUPA'
import importlib.util
for name in ("torch", "torch_supa", "torch_br", "sglang", "sglang_supa"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else None}")
PY_CHECK_SUPA

  if command -v brcc >/dev/null 2>&1; then
    log "brcc: $(command -v brcc)"
    brcc --version | head -n 3 || true
  else
    warn "brcc is not in PATH. If build.sh needs brcc, source brsw_set_env.sh first."
  fi
}

install_cpp_deps() {
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    log "installing gcovr==7.2"
    python3 -m pip install gcovr==7.2
  fi
}

install_python_deps() {
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    log "installing coverage==7.10.7"
    python3 -m pip install coverage==7.10.7
  fi
}

clean_cpp_outputs() {
  [[ "${CLEAN_COVERAGE}" == "1" ]] || return 0
  log "cleaning previous C/C++ coverage outputs"
  find . \( -name '*.gcda' -o -name '*.gcno' \) -type f -delete
  rm -f coverage_gcc.json coverage_brcc.json
  rm -rf "${HTML_DIR}"
}

run_cpp_build_and_tests() {
  [[ -f gcc_cov.cfg || -f brcc_cov.cfg ]] || fail "gcc_cov.cfg/brcc_cov.cfg not found"

  if [[ "${SKIP_BUILD}" != "1" ]]; then
    log "building with coverage: ${BUILD_CMD}"
    eval "${BUILD_CMD}"
  else
    log "skip build"
  fi

  if [[ -n "${RUN_CMD}" ]]; then
    log "running coverage workload: ${RUN_CMD}"
    eval "${RUN_CMD}"
  else
    warn "no --run-cmd provided. .gcda files are generated only after running tests/examples; report may be empty."
  fi
}

report_cpp() {
  mkdir -p "${HTML_DIR}"
  local json_args=()

  if [[ -f gcc_cov.cfg ]]; then
    log "gcovr terminal report: gcc_cov.cfg"
    python3 -m gcovr -r ./ --gcov-ignore-parse-errors --gcov-ignore-errors=all --config ./gcc_cov.cfg
    log "gcovr json: coverage_gcc.json"
    python3 -m gcovr -r ./ --gcov-ignore-parse-errors --gcov-ignore-errors all --config ./gcc_cov.cfg --json ./coverage_gcc.json
    json_args+=( -a coverage_gcc.json )
  fi

  if [[ -f brcc_cov.cfg ]]; then
    log "gcovr terminal report: brcc_cov.cfg"
    python3 -m gcovr -r ./ --gcov-ignore-parse-errors --gcov-ignore-errors=all --config ./brcc_cov.cfg
    log "gcovr json: coverage_brcc.json"
    python3 -m gcovr -r ./ --gcov-ignore-parse-errors --gcov-ignore-errors all --config ./brcc_cov.cfg --json ./coverage_brcc.json
    json_args+=( -a coverage_brcc.json )
  fi

  if (( ${#json_args[@]} > 0 )); then
    log "gcovr html: ${HTML_DIR}/coverage_details.html"
    python3 -m gcovr "${json_args[@]}" -r ./ --gcov-ignore-errors all --html-details -o "${HTML_DIR}/coverage_details.html"
  fi

  log "coverage data files: $(find . -name '*.gcda' | wc -l) .gcda, $(find . -name '*.gcno' | wc -l) .gcno"
}

run_python_coverage() {
  [[ -f .coveragerc ]] || fail "Python coverage requested but .coveragerc does not exist"

  if [[ -z "${PYTHON_COV_CMD}" ]]; then
    if [[ -n "${RUN_CMD}" ]]; then
      warn "--python-cov-cmd is empty; reuse --run-cmd for Python coverage"
      PYTHON_COV_CMD="${RUN_CMD}"
    else
      fail "Python coverage requested but --python-cov-cmd is empty"
    fi
  fi

  install_python_deps
  mkdir -p "${HTML_DIR}"
  export COVERAGE_RCFILE=./.coveragerc
  log "running Python coverage workload: ${PYTHON_COV_CMD}"
  python3 -m coverage run -a --branch -m ${PYTHON_COV_CMD#python3 -m }
  log "coverage report"
  python3 -m coverage report
  log "coverage html: ${HTML_DIR}/index.html"
  python3 -m coverage html -d "${HTML_DIR}"
}

run_python_coverage

log "done"
