#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(${PYTHON_BIN} --version)"

if [[ "${CLEAN:-1}" != "0" ]]; then
  echo "Cleaning previous build artifacts..."
  rm -rf build dist *.egg-info
fi

if [[ "${RUN_TESTS:-0}" == "1" ]]; then
  echo "Running tests..."
  "${PYTHON_BIN}" -m pytest
fi

if ! "${PYTHON_BIN}" -c "import build" >/dev/null 2>&1; then
  echo "Missing Python package: build"
  echo "Install it with: ${PYTHON_BIN} -m pip install build"
  exit 1
fi

echo "Building source distribution and wheel..."
"${PYTHON_BIN}" -m build

echo "Build artifacts:"
ls -lh dist
